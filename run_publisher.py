import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

# Disable CUDA graphs for torch.compile in multi-threaded environment
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")

import json
import queue
import signal
import cv2
import numpy as np
import torch
from loguru import logger

_LOG_LEVEL = "INFO"
logger.remove()
logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)

from scipy.spatial.transform import Rotation

from mocap.core.gravity_alignment import build_camera_to_world_rotation
from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner
from mocap.realtime.interpolator import PoseInterpolator
from mocap.realtime.offline_prediction_player import OfflinePredictionPublisher
from mocap.realtime.publisher import ZMQPublisher
from mocap.core.setup_estimator import build_default_estimator
from mocap.utils.pose_protocol import prepare_publish_pose
from mocap.utils.rerun_visualizer import RerunVisualizer
from mocap.utils.upright_leveler import InitialUprightLeveler
from mocap.utils.video_source import create_video_source


FOV_MODEL_SIZE = "s"
FOV_RESOLUTION_LEVEL = 0
FOV_FIXED_SIZE = 512
FOV_FAST_MODE = True
YOLO_MODEL_PATH = "checkpoints/yolo/yolo11m-pose.engine"
MHR_MODEL_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"


class RealtimePublisher:
    def __init__(
        self,
        video_source,
        publish_hz,
        interpolate_lag_ms,
        smpl_model_path,
        multiview_model_dir,
        mhr2smpl_mapping_path,
        mhr_mesh_path=None,
        mhr_model_path=MHR_MODEL_PATH,
        smoother_dir=None,
        addr="tcp://*:5556",
        image_size=512,
        yolo_model_path=YOLO_MODEL_PATH,
        record=False,
        record_dir="output/records",
        min_person_confidence=0.75,
        rerun=False,
        rerun_session_name="fast_sam_3d_body_publisher",
        rerun_spawn=True,
        rerun_connect=None,
        rrd_output="",
        rerun_log_stride=2,
        rerun_max_image_side=720,
        rerun_mesh_overlay_stride=1,
        rerun_mesh_overlay=True,
        zmq_protocol_version=3,
        imu_level_init_frames=15,
        body_orient_source="joint_alignment",
    ):
        logger.info("Initializing Realtime Publisher...")
        self.min_person_confidence = min_person_confidence
        self.body_orient_source = str(body_orient_source)
        self._prev_body_quat_xyzw = None

        logger.info("Loading SAM 3D model...")
        self.estimator = build_default_estimator(
            image_size=image_size,
            yolo_model_path=yolo_model_path,
            fov_model_size=FOV_MODEL_SIZE,
            fov_resolution_level=FOV_RESOLUTION_LEVEL,
            fov_fixed_size=FOV_FIXED_SIZE,
            fov_fast_mode=FOV_FAST_MODE,
        )

        self.video_source = video_source
        frame_size = self.video_source.get_frame_size()
        if frame_size is None:
            self.frame_width, self.frame_height = 640, 480
        else:
            self.frame_width, self.frame_height = frame_size

        self.rerun_viz = None
        self._rerun_frame_idx = 0
        self.rerun_log_stride = max(1, int(rerun_log_stride))

        logger.info("Warming up model...")
        self._warmup()

        cam_intrinsics_np = self.video_source.get_camera_intrinsics()
        if cam_intrinsics_np is not None:
            self.cam_intrinsics = torch.from_numpy(
                np.asarray(cam_intrinsics_np, dtype=np.float32)
            )
            logger.info(
                f"Using camera intrinsics: fx={cam_intrinsics_np[0,0,0]:.2f}, fy={cam_intrinsics_np[0,1,1]:.2f}"
            )
        else:
            self.cam_intrinsics = None
            logger.warning("No camera intrinsics provided, will use FOV estimator")

        if rerun:
            logger.info("Initializing Rerun visualization...")
            if rrd_output:
                Path(rrd_output).parent.mkdir(parents=True, exist_ok=True)
            self.rerun_viz = RerunVisualizer(
                width=self.frame_width,
                height=self.frame_height,
                session_name=rerun_session_name,
                cam_intrinsics=cam_intrinsics_np[0] if cam_intrinsics_np is not None else None,
                spawn=rerun_spawn,
                connect=rerun_connect,
                rrd_output=rrd_output,
                max_image_side=rerun_max_image_side,
                mesh_overlay_stride=rerun_mesh_overlay_stride,
                enable_mesh_overlay=rerun_mesh_overlay,
            )
            logger.info(
                "Rerun visualization enabled "
                f"(log_stride={self.rerun_log_stride}, "
                f"max_side={rerun_max_image_side}, "
                f"mesh_overlay_stride={rerun_mesh_overlay_stride}, "
                f"mesh_overlay={'on' if rerun_mesh_overlay else 'off'})"
            )

        self.gravity_direction = self.video_source.get_gravity_direction()
        logger.info(
            f"Using gravity-aligned world frame: gravity=[{self.gravity_direction[0]:.3f}, {self.gravity_direction[1]:.3f}, {self.gravity_direction[2]:.3f}]"
        )
        self.R_world_cam = build_camera_to_world_rotation(self.gravity_direction)
        R_zup_adjustment = np.array(
            [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64
        )
        self.R_world_cam = R_zup_adjustment @ self.R_world_cam

        self.record = record
        self.record_dir = record_dir
        if self.record:
            os.makedirs(self.record_dir, exist_ok=True)
            session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir = os.path.join(self.record_dir, session_id)
            os.makedirs(self.session_dir, exist_ok=True)
            self.video_out_path = os.path.join(self.session_dir, "raw_video.mp4")
            self.smpl_out_path = os.path.join(self.session_dir, "smpl_data.npz")
            logger.info(f"Recording enabled. Saving to: {self.session_dir}")

            intrinsics_data = {
                "width": 848,
                "height": 480,
                "gravity": self.gravity_direction.tolist(),
            }
            if cam_intrinsics_np is not None:
                intrinsics_data["fx"] = float(cam_intrinsics_np[0, 0, 0])
                intrinsics_data["fy"] = float(cam_intrinsics_np[0, 1, 1])
                intrinsics_data["cx"] = float(cam_intrinsics_np[0, 0, 2])
                intrinsics_data["cy"] = float(cam_intrinsics_np[0, 1, 2])
                intrinsics_data["camera_matrix"] = cam_intrinsics_np[0].tolist()

            frame_size = self.video_source.get_frame_size()
            if frame_size is not None:
                intrinsics_data["width"] = frame_size[0]
                intrinsics_data["height"] = frame_size[1]

            with open(os.path.join(self.session_dir, "camera.json"), "w") as f:
                json.dump(intrinsics_data, f, indent=2)

            self.video_queue = queue.Queue(maxsize=300)
            self.smpl_queue = queue.Queue(maxsize=300)
            self.smpl_data_list = []

        logger.info(f"Loading multiview fusion runner model={smpl_model_path}...")
        self.fusion_runner = MultiViewFusionRunner(
            smpl_model_path=smpl_model_path,
            model_dir=multiview_model_dir,
            mapping_path=mhr2smpl_mapping_path,
            mhr_mesh_path=mhr_mesh_path,
            mhr_model_path=mhr_model_path,
            smoother_dir=smoother_dir,
        )

        self.publish_hz = publish_hz
        self.publish_dt = 1.0 / publish_hz
        self.interpolate_lag_s = interpolate_lag_ms / 1000.0

        self.interpolator = PoseInterpolator()
        self.zmq_protocol_version = int(zmq_protocol_version)
        self.publisher = ZMQPublisher(addr, protocol_version=self.zmq_protocol_version)
        logger.info(
            f"ZMQ publisher ready at {addr} using protocol v{self.zmq_protocol_version}"
        )
        self.upright_leveler = InitialUprightLeveler(imu_level_init_frames)
        self._last_level_log_count = 0
        if self.upright_leveler.enabled:
            logger.info(
                "Initial IMU-based upright leveling enabled "
                f"({self.upright_leveler.calibration_frames} valid pose frames)"
            )
        logger.info(f"Body orientation source: {self.body_orient_source}")

        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        self.running = False
        self.video_ended = False
        self._final_stats_logged = False
        self._publish_enabled = True
        self._closed = False

        self._capture_wall_base = None
        self._capture_ts_base = None
        self._pose_clock_lock = threading.Lock()
        self._latest_pose_source_ts = None
        self._latest_pose_perf_ts = None

        self.capture_thread = None
        self.inference_thread = None
        self.publish_thread = None
        self.recording_thread = None

        self.first_capture_ts = None
        self.first_infer_ts = None

        self.stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "infer_count": 0,
            "inference_times": deque(maxlen=100),
            "infer_total_time_s": 0.0,
            "convert_times": deque(maxlen=100),
            "convert_total_time_s": 0.0,
            "publish_count": 0,
            "publish_intervals": deque(maxlen=500),
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
        }

        self._live_log_interval_s = 2.0
        self._live_last_log_perf = time.perf_counter()
        self._live_prev_stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "infer_count": 0,
            "infer_total_time_s": 0.0,
            "convert_total_time_s": 0.0,
            "publish_count": 0,
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
        }

        logger.success("Publisher ready")

    def _warmup(self):
        frame_size = self.video_source.get_frame_size()
        if frame_size is None:
            width, height = 640, 480
        else:
            width, height = frame_size

        dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        warmup_bbox = np.array(
            [[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32
        )
        for _ in range(2):
            _ = self.estimator.process_one_image(
                dummy_img,
                bboxes=warmup_bbox,
                hand_box_source="body_decoder",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _capture_loop(self):
        while self.running:
            try:
                frame, frame_timestamp = self.video_source.get_frame()
            except Exception as exc:
                if self.running:
                    logger.warning(f"Capture loop stopped due to source error: {exc}")
                self.video_ended = True
                self._frame_event.set()
                break

            if frame is None:
                self.video_ended = True
                self._frame_event.set()
                break

            if frame_timestamp is None:
                continue

            if self.first_capture_ts is None:
                self.first_capture_ts = frame_timestamp

            if self._capture_wall_base is None:
                self._capture_wall_base = time.perf_counter()
                self._capture_ts_base = frame_timestamp
            else:
                target_wall = self._capture_wall_base + (
                    frame_timestamp - self._capture_ts_base
                )
                now_wall = time.perf_counter()
                delay = target_wall - now_wall
                if delay > 0:
                    time.sleep(delay)

            self.stats["capture_count"] += 1

            if self.record:
                try:
                    self.video_queue.put_nowait((frame_timestamp, frame))
                except queue.Full:
                    logger.warning(
                        "Video recording queue full, dropping frame for recording"
                    )

            with self._latest_frame_lock:
                if self._latest_frame is not None:
                    self.stats["dropped_capture_count"] += 1
                self._latest_frame = (frame, frame_timestamp)
            self._frame_event.set()

    def _inference_loop(self):
        while self.running:
            self._frame_event.wait(timeout=0.05)
            self._frame_event.clear()

            with self._latest_frame_lock:
                item = self._latest_frame
                self._latest_frame = None

            if item is None:
                if self.video_ended:
                    break
                continue

            frame, frame_timestamp = item

            if self.first_infer_ts is None:
                self.first_infer_ts = frame_timestamp

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t_infer_start = time.perf_counter()
            outputs = self.estimator.process_one_image(
                frame_rgb,
                cam_int=self.cam_intrinsics,
                hand_box_source="yolo_pose",
            )
            infer_dt = time.perf_counter() - t_infer_start

            self.stats["inference_times"].append(infer_dt)
            self.stats["infer_total_time_s"] += infer_dt
            self.stats["infer_count"] += 1
            self._maybe_log_rerun(frame, frame_rgb, outputs)

            num_persons = len(outputs)
            if num_persons != 1:
                if self._publish_enabled:
                    logger.warning(
                        "Pose publishing paused (not exactly 1 confident person detected)."
                    )
                    self._publish_enabled = False
                continue

            if not self._publish_enabled:
                logger.info("Exactly 1 person confirmed; resuming pose publishing.")
                self._publish_enabled = True

            t0 = time.perf_counter()
            out = outputs[0]
            required = ("pred_vertices", "pred_cam_t", "global_rot")
            missing = [k for k in required if k not in out]
            if missing:
                logger.warning(f"Skipping frame missing keys: missing={missing}")
                continue
            pred_vertices = np.asarray(out["pred_vertices"], dtype=np.float32)
            pred_cam_t = np.asarray(out["pred_cam_t"], dtype=np.float32)
            smpl_pose, canonical_joints, _betas, _weights = self.fusion_runner.infer(
                [(pred_vertices, pred_cam_t)]
            )
            body_quat_xyzw = self._estimate_body_quat_xyzw(out, canonical_joints)
            body_quat, smpl_joints, smpl_pose = self._prepare_publish_pose(
                body_quat_xyzw, canonical_joints, smpl_pose
            )

            if self.upright_leveler.enabled and not self.upright_leveler.ready:
                self.upright_leveler.update(body_quat)
                collected = self.upright_leveler.num_collected
                if collected != self._last_level_log_count and (
                    collected == 1
                    or collected == self.upright_leveler.calibration_frames
                    or collected % 5 == 0
                ):
                    logger.info(
                        "Collecting initial upright calibration "
                        f"({collected}/{self.upright_leveler.calibration_frames})"
                    )
                    self._last_level_log_count = collected
                if self.upright_leveler.ready:
                    logger.info(
                        "Initial upright calibration locked "
                        f"(removed ~{self.upright_leveler.estimated_tilt_deg:.2f} deg startup tilt)"
                    )
                else:
                    continue

            if self.upright_leveler.ready and self.upright_leveler.enabled:
                body_quat = self.upright_leveler.apply(body_quat)

            convert_dt = time.perf_counter() - t0
            self.stats["convert_times"].append(convert_dt)
            self.stats["convert_total_time_s"] += convert_dt

            self.interpolator.add_pose(
                frame_timestamp, body_quat, smpl_joints, smpl_pose
            )

            if self.record:
                try:
                    self.smpl_queue.put_nowait(
                        (frame_timestamp, body_quat, smpl_joints, smpl_pose)
                    )
                except queue.Full:
                    logger.warning(
                        "SMPL recording queue full, dropping pose for recording"
                    )

            with self._pose_clock_lock:
                self._latest_pose_source_ts = frame_timestamp
                self._latest_pose_perf_ts = time.perf_counter()

    def _maybe_log_rerun(self, frame_bgr, frame_rgb, outputs):
        if self.rerun_viz is None:
            return
        if self._rerun_frame_idx % self.rerun_log_stride != 0:
            self._rerun_frame_idx += 1
            return

        try:
            self.rerun_viz.log_frame(
                frame_idx=self._rerun_frame_idx,
                frame_bgr=frame_bgr,
                frame_rgb=frame_rgb,
                outputs=outputs,
                faces=self.estimator.faces,
            )
            self._rerun_frame_idx += 1
        except Exception as exc:
            logger.warning(f"Disabling Rerun visualization after logging failure: {exc}")
            self.rerun_viz.close()
            self.rerun_viz = None

    def _compute_body_quat(self, global_rot):
        global_rot = np.asarray(global_rot, dtype=np.float64).reshape(3)
        rot = Rotation.from_euler("ZYX", global_rot)
        x180 = Rotation.from_euler("x", 180.0, degrees=True)
        return (x180 * rot).as_quat().astype(np.float64)

    def _estimate_body_quat_xyzw(self, out, canonical_joints):
        if self.body_orient_source == "joint_alignment":
            try:
                quat_xyzw = self.fusion_runner.estimate_body_quat_xyzw_from_joint_alignment(
                    canonical_joints,
                    out["pred_vertices"],
                    out["pred_cam_t"],
                    prev_quat_xyzw=self._prev_body_quat_xyzw,
                )
                self._prev_body_quat_xyzw = quat_xyzw
                return quat_xyzw
            except Exception as exc:
                logger.warning(
                    "Joint-alignment body orientation failed; "
                    f"falling back to stage1_global_rot: {type(exc).__name__}: {exc}"
                )

        quat_xyzw = self._compute_body_quat(out["global_rot"])
        self._prev_body_quat_xyzw = quat_xyzw
        return quat_xyzw

    def _prepare_publish_pose(self, body_quat_xyzw, canonical_joints, smpl_pose):
        return prepare_publish_pose(
            body_quat_xyzw,
            canonical_joints,
            smpl_pose,
            self.R_world_cam,
        )

    def _maybe_log_live_stats(self, now_perf):
        elapsed = now_perf - self._live_last_log_perf
        if elapsed < self._live_log_interval_s:
            return

        curr = {
            "capture_count": self.stats["capture_count"],
            "dropped_capture_count": self.stats["dropped_capture_count"],
            "infer_count": self.stats["infer_count"],
            "infer_total_time_s": self.stats["infer_total_time_s"],
            "convert_total_time_s": self.stats["convert_total_time_s"],
            "publish_count": self.stats["publish_count"],
            "publish_interpolated_count": self.stats["publish_interpolated_count"],
            "publish_fallback_count": self.stats["publish_fallback_count"],
        }
        prev = self._live_prev_stats

        d_capture = curr["capture_count"] - prev["capture_count"]
        d_drop = curr["dropped_capture_count"] - prev["dropped_capture_count"]
        d_infer = curr["infer_count"] - prev["infer_count"]
        d_infer_time = curr["infer_total_time_s"] - prev["infer_total_time_s"]
        d_convert_time = curr["convert_total_time_s"] - prev["convert_total_time_s"]
        d_publish = curr["publish_count"] - prev["publish_count"]
        d_interp = (
            curr["publish_interpolated_count"] - prev["publish_interpolated_count"]
        )
        d_fallback = curr["publish_fallback_count"] - prev["publish_fallback_count"]

        infer_fps = d_infer / elapsed
        infer_ms = (d_infer_time / d_infer * 1000.0) if d_infer > 0 else float("nan")
        convert_ms = (
            (d_convert_time / d_infer * 1000.0) if d_infer > 0 else float("nan")
        )
        publish_hz = d_publish / elapsed
        fallback_pct = (d_fallback / d_publish * 100.0) if d_publish > 0 else 0.0

        infer_ms_str = f"{infer_ms:.1f}" if np.isfinite(infer_ms) else "n/a"
        convert_ms_str = f"{convert_ms:.1f}" if np.isfinite(convert_ms) else "n/a"
        logger.info(
            "Live: "
            f"capture={d_capture/elapsed:.1f}fps drop+={d_drop}, "
            f"infer throughput={infer_fps:.1f}fps model={infer_ms_str}ms, "
            f"convert={convert_ms_str}ms, "
            f"publish={publish_hz:.1f}Hz, "
            f"interp+={d_interp}, fallback+={d_fallback} ({fallback_pct:.1f}%)"
        )

        self._live_prev_stats = curr
        self._live_last_log_perf = now_perf

    def _recording_loop(self):
        video_writer = None
        fps = self.video_source.fps
        while self.running:
            try:
                frame_ts, frame = self.video_queue.get(timeout=0.05)
                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    video_writer = cv2.VideoWriter(
                        self.video_out_path, fourcc, fps, (w, h)
                    )
                if video_writer is not None:
                    video_writer.write(frame)
            except queue.Empty:
                pass

            try:
                while True:
                    smpl_ts, body_quat, smpl_joints, smpl_pose = (
                        self.smpl_queue.get_nowait()
                    )
                    self.smpl_data_list.append(
                        {
                            "timestamp": smpl_ts,
                            "body_quat": body_quat,
                            "smpl_joints": smpl_joints,
                            "smpl_pose": smpl_pose,
                        }
                    )
            except queue.Empty:
                pass

        # Drain queues on exit
        if self.record:
            logger.info("Flushing recording queues to disk. Please wait...")
            while not self.video_queue.empty():
                try:
                    frame_ts, frame = self.video_queue.get_nowait()
                    if video_writer is not None:
                        video_writer.write(frame)
                except queue.Empty:
                    break

            while not self.smpl_queue.empty():
                try:
                    smpl_ts, body_quat, smpl_joints, smpl_pose = (
                        self.smpl_queue.get_nowait()
                    )
                    self.smpl_data_list.append(
                        {
                            "timestamp": smpl_ts,
                            "body_quat": body_quat,
                            "smpl_joints": smpl_joints,
                            "smpl_pose": smpl_pose,
                        }
                    )
                except queue.Empty:
                    break

            if video_writer is not None:
                video_writer.release()
                logger.info(f"Finished writing video to {self.video_out_path}")

            if self.smpl_data_list:
                timestamps = np.array([d["timestamp"] for d in self.smpl_data_list])
                body_quats = np.array([d["body_quat"] for d in self.smpl_data_list])
                smpl_joints = np.array([d["smpl_joints"] for d in self.smpl_data_list])
                smpl_poses = np.array([d["smpl_pose"] for d in self.smpl_data_list])
                np.savez(
                    self.smpl_out_path,
                    timestamps=timestamps,
                    body_quats=body_quats,
                    smpl_joints=smpl_joints,
                    smpl_poses=smpl_poses,
                )
                logger.info(f"Finished writing SMPL data to {self.smpl_out_path}")

    def _publish_loop(self):
        last_publish = None
        next_publish = time.perf_counter()

        while self.running:
            now_perf = time.perf_counter()
            wait_time = next_publish - now_perf
            if wait_time > 0:
                time.sleep(min(wait_time, 0.0015))
                continue

            wall_now = time.time()
            with self._pose_clock_lock:
                latest_pose_source_ts = self._latest_pose_source_ts
                latest_pose_perf_ts = self._latest_pose_perf_ts

            if latest_pose_source_ts is None or latest_pose_perf_ts is None:
                query_ts = wall_now - self.interpolate_lag_s
            else:
                source_now_est = latest_pose_source_ts + (
                    now_perf - latest_pose_perf_ts
                )
                query_ts = source_now_est - self.interpolate_lag_s

            result = self.interpolator.interpolate(query_ts)
            used_fallback = False
            if result is None:
                latest = self.interpolator.get_latest_pose()
                if latest is not None:
                    result = latest
                    used_fallback = True

            if result is not None and self._publish_enabled:
                self.publisher.publish(*result)
                self.stats["publish_count"] += 1
                if used_fallback:
                    self.stats["publish_fallback_count"] += 1
                else:
                    self.stats["publish_interpolated_count"] += 1
                if last_publish is not None:
                    self.stats["publish_intervals"].append(now_perf - last_publish)
                last_publish = now_perf

            next_publish += self.publish_dt
            if next_publish < now_perf - self.publish_dt:
                missed = int((now_perf - next_publish) / self.publish_dt) + 1
                next_publish += missed * self.publish_dt

            if self.video_ended and self.stats["infer_count"] > 0:
                if self.stats["publish_count"] > max(10, int(self.publish_hz * 0.4)):
                    break

            self._maybe_log_live_stats(now_perf)

    def _log_final_stats(self):
        if self._final_stats_logged:
            return

        inf_msg = "Inference n/a"
        if self.stats["inference_times"]:
            inf_mean = np.mean(self.stats["inference_times"]) * 1000
            inf_fps = 1.0 / np.mean(self.stats["inference_times"])
            inf_msg = f"Inference {inf_mean:.1f}ms ({inf_fps:.1f}fps)"
            if self.stats["convert_times"]:
                inf_msg += (
                    f", convert={np.mean(self.stats['convert_times']) * 1000:.1f}ms"
                )

        pub_msg = "Publish n/a"
        if self.stats["publish_intervals"]:
            pub_hz = 1.0 / np.mean(self.stats["publish_intervals"])
            pub_msg = f"Publish {pub_hz:.1f}Hz (target {self.publish_hz:.1f}Hz)"

        logger.info(
            f"Final stats: {inf_msg}, {pub_msg}, "
            f"published={self.stats['publish_count']}, "
            f"interp={self.stats['publish_interpolated_count']}, "
            f"fallback={self.stats['publish_fallback_count']}, "
            f"capture_drop={self.stats['dropped_capture_count']}"
        )
        self._final_stats_logged = True

    def _release_video_source_with_timeout(self, timeout_s=10.0):
        release_errors = []

        def _release():
            try:
                self.video_source.release()
            except Exception as exc:
                release_errors.append(exc)

        t = threading.Thread(target=_release, daemon=True)
        t.start()
        t.join(timeout=timeout_s)

        if t.is_alive():
            logger.warning("Timed out while releasing video source; continue shutdown")
        elif release_errors:
            logger.warning(f"Video source release raised: {release_errors[0]}")

    def start(self):
        logger.info("Starting realtime publisher (Press Ctrl+C to stop)")
        self.running = True

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)

        if self.record:
            self.recording_thread = threading.Thread(
                target=self._recording_loop, daemon=True
            )
            self.recording_thread.start()

        self.capture_thread.start()
        self.inference_thread.start()
        self.publish_thread.start()

        while self.running:
            if self.video_ended and not self.inference_thread.is_alive():
                self.running = False
                break
            if (
                not self.capture_thread.is_alive()
                and not self.inference_thread.is_alive()
            ):
                self.running = False
                break
            time.sleep(0.05)

        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)

        self._log_final_stats()

    def stop(self):
        if self._closed:
            return

        self.running = False
        self._release_video_source_with_timeout(timeout_s=1.0)

        if (
            self.record
            and self.recording_thread is not None
            and self.recording_thread.is_alive()
        ):
            logger.info("Waiting for recording thread to finish writing to disk...")
            self.recording_thread.join(timeout=5.0)

        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.publish_thread is not None and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)

        self._log_final_stats()

        if self.rerun_viz is not None:
            self.rerun_viz.close()
            self.rerun_viz = None

        self.publisher.close()
        self._closed = True


def main():
    parser = argparse.ArgumentParser(
        description="Publish SAM 3D pose from camera/video stream over ZMQ"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="camera",
        choices=["camera", "video", "prediction"],
        help="Video source type",
    )
    parser.add_argument(
        "--video", type=str, help="Path to video file (for --source video)"
    )
    parser.add_argument(
        "--prediction-file",
        type=str,
        help="Path to offline prediction .npz file (for --source prediction)",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        help=(
            "Camera metadata path for --source video. "
            "Supports record_realsense camera.json and GENMO capture.json."
        ),
    )
    parser.add_argument(
        "--no-loop",
        dest="no_loop",
        action="store_true",
        help="Disable loop video playback (for --source video)",
    )
    parser.add_argument(
        "--publish-hz", type=float, default=50.0, help="Publisher output rate in Hz"
    )
    parser.add_argument(
        "--interp-lag-ms",
        type=float,
        default=140.0,
        help="Interpolation lag in ms to make 10Hz inference interpolatable",
    )
    parser.add_argument(
        "--addr", type=str, default="tcp://*:5556", help="ZMQ bind address"
    )
    parser.add_argument(
        "--imu-level-init-frames",
        type=int,
        default=15,
        help=(
            "Use the gravity-aligned RealSense frame to remove startup torso pitch/roll bias. "
            "The publisher waits for this many valid pose frames before publishing. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--body-orient-source",
        type=str,
        default="joint_alignment",
        choices=["joint_alignment", "stage1_global_rot"],
        help=(
            "Root/body orientation source. "
            "joint_alignment estimates orientation by aligning fused canonical joints "
            "to Stage-1 observed joints; stage1_global_rot preserves the original behavior."
        ),
    )
    parser.add_argument(
        "--zmq-protocol-version",
        type=int,
        default=3,
        choices=[2, 3],
        help=(
            "Packed ZMQ protocol version. "
            "Use 3 for official SONIC release configs; "
            "use 2 only for custom SMPL-only subscribers."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        choices=[256, 384, 512],
        help="Image size for SAM3D model (must match TensorRT engine)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=YOLO_MODEL_PATH,
        help="YOLO pose model path (e.g., yolo11m-pose.engine or yolo11n-pose.engine)",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        help="SMPL model pickle file",
    )
    parser.add_argument(
        "--nn-model-dir",
        type=str,
        help="NN fusion model directory (contains best_model.pth and sample_idx.npy)",
    )
    parser.add_argument(
        "--mhr2smpl-mapping-path",
        type=str,
        help="Path to mhr2smpl_mapping.npz (mhr_vert_ids or triangle_ids format)",
    )
    parser.add_argument(
        "--mhr-mesh-path",
        type=str,
        default=None,
        help="Path to MHR mesh PLY (required when mapping uses triangle_ids format)",
    )
    parser.add_argument(
        "--mhr-model-path",
        type=str,
        default=MHR_MODEL_PATH,
        help=(
            "Path to MHR TorchScript model. Used to recover mesh faces automatically "
            "when the mapping file uses triangle_ids."
        ),
    )
    parser.add_argument(
        "--smoother-dir",
        type=str,
        default=None,
        help="Smoother checkpoint directory (contains smoother_best.pth and smoother_config.json)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable recording raw video and SMPL data to disk",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="output/records",
        help="Directory to save recordings (default: output/records)",
    )
    parser.add_argument(
        "--min-person-confidence",
        type=float,
        default=0.75,
        help=(
            "Minimum YOLO detection confidence to count as a real person "
            "(default: 0.75). Detections below this threshold (e.g. humanoid robots) "
            "are filtered out before the single-person safety check."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Enable Rerun visualization during publishing.",
    )
    parser.add_argument(
        "--rerun-session-name",
        type=str,
        default="fast_sam_3d_body_publisher",
        help="Rerun session name.",
    )
    parser.add_argument(
        "--rerun-spawn",
        dest="rerun_spawn",
        action="store_true",
        help="Spawn a local Rerun viewer.",
    )
    parser.add_argument(
        "--no-rerun-spawn",
        dest="rerun_spawn",
        action="store_false",
        help="Do not auto-spawn a local Rerun viewer.",
    )
    parser.add_argument(
        "--rerun-connect",
        type=str,
        default=None,
        help="Connect to an existing Rerun viewer, e.g. 127.0.0.1:9876.",
    )
    parser.add_argument(
        "--rrd-output",
        type=str,
        default="",
        help="Optional path to save the Rerun recording as a .rrd file.",
    )
    parser.add_argument(
        "--rerun-log-stride",
        type=int,
        default=2,
        help="Log one out of every N inference frames to Rerun.",
    )
    parser.add_argument(
        "--rerun-max-image-side",
        type=int,
        default=720,
        help="Downscale Rerun images so the largest side is at most this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--rerun-mesh-overlay-stride",
        type=int,
        default=1,
        help="Render the 2D mesh overlay every N logged Rerun frames. Use values > 1 only to trade alignment fidelity for speed.",
    )
    parser.add_argument(
        "--rerun-mesh-overlay",
        dest="rerun_mesh_overlay",
        action="store_true",
        help="Enable the expensive 2D mesh overlay panel in Rerun.",
    )
    parser.add_argument(
        "--no-rerun-mesh-overlay",
        dest="rerun_mesh_overlay",
        action="store_false",
        help="Disable the expensive 2D mesh overlay panel in Rerun for faster runtime.",
    )
    parser.set_defaults(rerun_spawn=True)
    parser.set_defaults(rerun_mesh_overlay=True)
    args = parser.parse_args()

    if args.publish_hz <= 0:
        parser.error("--publish-hz must be > 0")
    if args.interp_lag_ms < 0:
        parser.error("--interp-lag-ms must be >= 0")
    if args.imu_level_init_frames < 0:
        parser.error("--imu-level-init-frames must be >= 0")

    if args.source in ("camera", "video"):
        missing_model_args = [
            name
            for name, value in (
                ("--smpl-model-path", args.smpl_model_path),
                ("--nn-model-dir", args.nn_model_dir),
                ("--mhr2smpl-mapping-path", args.mhr2smpl_mapping_path),
            )
            if not value
        ]
        if missing_model_args:
            parser.error(
                f"{', '.join(missing_model_args)} required when --source is camera or video"
            )

    if args.source == "camera":
        video_source = create_video_source("camera", width=848, height=480, fps=30)
        publisher = RealtimePublisher(
            video_source=video_source,
            publish_hz=args.publish_hz,
            interpolate_lag_ms=args.interp_lag_ms,
            smpl_model_path=args.smpl_model_path,
            multiview_model_dir=args.nn_model_dir,
            mhr2smpl_mapping_path=args.mhr2smpl_mapping_path,
            mhr_mesh_path=args.mhr_mesh_path,
            mhr_model_path=args.mhr_model_path,
            smoother_dir=args.smoother_dir,
            addr=args.addr,
            image_size=args.image_size,
            yolo_model_path=args.yolo_model,
            record=args.record,
            record_dir=args.record_dir,
            min_person_confidence=args.min_person_confidence,
            rerun=args.rerun,
            rerun_session_name=args.rerun_session_name,
            rerun_spawn=args.rerun_spawn,
            rerun_connect=args.rerun_connect,
            rrd_output=args.rrd_output,
            rerun_log_stride=args.rerun_log_stride,
            rerun_max_image_side=args.rerun_max_image_side,
            rerun_mesh_overlay_stride=args.rerun_mesh_overlay_stride,
            rerun_mesh_overlay=args.rerun_mesh_overlay,
            zmq_protocol_version=args.zmq_protocol_version,
            imu_level_init_frames=args.imu_level_init_frames,
            body_orient_source=args.body_orient_source,
        )
    elif args.source == "video":
        if not args.video:
            parser.error("--video required when --source video")
        if not args.intrinsics:
            parser.error("--intrinsics required when --source video")
        video_source = create_video_source(
            "video",
            video_path=args.video,
            intrinsics_path=args.intrinsics,
            loop=not args.no_loop,
        )
        publisher = RealtimePublisher(
            video_source=video_source,
            publish_hz=args.publish_hz,
            interpolate_lag_ms=args.interp_lag_ms,
            smpl_model_path=args.smpl_model_path,
            multiview_model_dir=args.nn_model_dir,
            mhr2smpl_mapping_path=args.mhr2smpl_mapping_path,
            mhr_mesh_path=args.mhr_mesh_path,
            mhr_model_path=args.mhr_model_path,
            smoother_dir=args.smoother_dir,
            addr=args.addr,
            image_size=args.image_size,
            yolo_model_path=args.yolo_model,
            record=args.record,
            record_dir=args.record_dir,
            min_person_confidence=args.min_person_confidence,
            rerun=args.rerun,
            rerun_session_name=args.rerun_session_name,
            rerun_spawn=args.rerun_spawn,
            rerun_connect=args.rerun_connect,
            rrd_output=args.rrd_output,
            rerun_log_stride=args.rerun_log_stride,
            rerun_max_image_side=args.rerun_max_image_side,
            rerun_mesh_overlay_stride=args.rerun_mesh_overlay_stride,
            rerun_mesh_overlay=args.rerun_mesh_overlay,
            zmq_protocol_version=args.zmq_protocol_version,
            imu_level_init_frames=args.imu_level_init_frames,
            body_orient_source=args.body_orient_source,
        )
    else:
        if not args.prediction_file:
            parser.error("--prediction-file required when --source prediction")
        if args.record:
            parser.error("--record is not supported when --source prediction")
        publisher = OfflinePredictionPublisher(
            prediction_file=args.prediction_file,
            publish_hz=args.publish_hz,
            interpolate_lag_ms=args.interp_lag_ms,
            addr=args.addr,
            zmq_protocol_version=args.zmq_protocol_version,
            rerun=args.rerun,
            rerun_session_name=args.rerun_session_name,
            rerun_spawn=args.rerun_spawn,
            rerun_connect=args.rerun_connect,
            rrd_output=args.rrd_output,
            rerun_log_stride=args.rerun_log_stride,
            rerun_max_image_side=args.rerun_max_image_side,
            rerun_mesh_overlay_stride=args.rerun_mesh_overlay_stride,
            rerun_mesh_overlay=args.rerun_mesh_overlay,
            smpl_model_path=args.smpl_model_path,
        )

    try:
        publisher.start()
    except KeyboardInterrupt:
        logger.info("Stopping...")
        # Ignore further Ctrl+C signals while we are shutting down and flushing to disk
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        publisher.stop()
        logger.success("Stopped.")


if __name__ == "__main__":
    main()
