from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from mocap.realtime.publisher import SONIC_NUM_JOINTS, ZMQPublisher
from mocap.utils.offline_prediction_io import (
    build_prediction_summary,
    load_prediction_archive,
    status_code_to_name,
)
from mocap.utils.rerun_visualizer import RerunVisualizer
from mocap.utils.smpl_render_utils import (
    load_smpl_model,
    smpl_vertices_joints_from_pose,
)


def _scalar_str(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _default_replay_rrd_path(prediction_file: str | Path) -> Path:
    prediction_path = Path(prediction_file)
    stem = prediction_path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return prediction_path.with_name(f"{stem}_replay_publish.rrd")


class OfflinePredictionPublisher:
    def __init__(
        self,
        prediction_file,
        publish_hz,
        interpolate_lag_ms,
        addr="tcp://*:5556",
        zmq_protocol_version=3,
        rerun=False,
        rerun_session_name="fast_sam_3d_body_prediction_replay",
        rerun_spawn=True,
        rerun_connect=None,
        rrd_output="",
        rerun_log_stride=1,
        rerun_max_image_side=720,
        rerun_mesh_overlay_stride=1,
        rerun_mesh_overlay=True,
        smpl_model_path=None,
    ):
        logger.info(f"Loading offline prediction archive: {prediction_file}")
        self.prediction_data = load_prediction_archive(prediction_file)
        self.summary = build_prediction_summary(self.prediction_data)

        self.timestamps = np.asarray(
            self.prediction_data["timestamps"], dtype=np.float64
        ).reshape(-1)
        self.relative_timestamps = self.timestamps - self.timestamps[0]
        self.frame_indices = np.asarray(
            self.prediction_data["frame_indices"], dtype=np.int64
        ).reshape(-1)
        self.valid_mask = np.asarray(
            self.prediction_data["valid_mask"], dtype=bool
        ).reshape(-1)
        self.status_codes = np.asarray(
            self.prediction_data["status_codes"], dtype=np.int32
        ).reshape(-1)
        self.num_persons = np.asarray(
            self.prediction_data["num_persons"], dtype=np.int32
        ).reshape(-1)
        self.body_quats = np.asarray(
            self.prediction_data["body_quats"], dtype=np.float64
        )
        self.smpl_joints = np.asarray(
            self.prediction_data["smpl_joints"], dtype=np.float64
        )
        self.smpl_poses = np.asarray(
            self.prediction_data["smpl_poses"], dtype=np.float64
        )

        self.camera_body_quats_xyzw = None
        if "camera_body_quats_xyzw" in self.prediction_data:
            self.camera_body_quats_xyzw = np.asarray(
                self.prediction_data["camera_body_quats_xyzw"], dtype=np.float64
            )

        self.pred_cam_ts = None
        if "pred_cam_ts" in self.prediction_data:
            self.pred_cam_ts = np.asarray(
                self.prediction_data["pred_cam_ts"], dtype=np.float64
            )

        self.source_fps = float(self.summary.get("fps", 0.0))
        if self.source_fps <= 0.0 and self.timestamps.size > 1:
            dt = float(np.median(np.diff(self.timestamps)))
            if dt > 1e-6:
                self.source_fps = 1.0 / dt

        requested_publish_hz = float(publish_hz)
        if self.source_fps > 0.0:
            self.publish_hz = self.source_fps
            if abs(requested_publish_hz - self.source_fps) > 1e-6:
                logger.info(
                    "Offline prediction replay follows source video timing "
                    f"({self.source_fps:.2f}fps); ignoring --publish-hz={requested_publish_hz:.2f}"
                )
        else:
            self.publish_hz = requested_publish_hz
            logger.warning(
                "Prediction archive does not contain a valid FPS; falling back to --publish-hz"
            )

        if float(interpolate_lag_ms) != 0.0:
            logger.info(
                "Offline prediction replay publishes one message per original frame; "
                f"ignoring --interp-lag-ms={float(interpolate_lag_ms):.1f}"
            )

        self.zmq_protocol_version = int(zmq_protocol_version)
        self.publisher = ZMQPublisher(addr, protocol_version=self.zmq_protocol_version)

        self.running = False
        self.finished_input = False
        self._closed = False
        self._publish_enabled = True
        self._last_pause_code = None
        self._playback_start_perf = None
        self._last_publish_perf = None

        self.input_thread = None

        self.rerun_enabled = bool(rerun)
        self.rerun_viz = None
        self.rerun_log_stride = max(1, int(rerun_log_stride))
        self._rerun_frame_count = 0
        self._video_cap = None
        self._video_next_frame_idx = 0
        self._frame_width = int(
            np.asarray(self.prediction_data.get("frame_width", 0)).item()
        )
        self._frame_height = int(
            np.asarray(self.prediction_data.get("frame_height", 0)).item()
        )
        self._camera_matrix = None
        if "camera_matrix" in self.prediction_data:
            self._camera_matrix = np.asarray(
                self.prediction_data["camera_matrix"], dtype=np.float32
            ).reshape(3, 3)

        self._smpl_model = None
        self._smpl_faces = None
        self._smpl_device = None
        self._smpl_num_betas = None
        self._zero_joint_state = np.zeros((SONIC_NUM_JOINTS,), dtype=np.float64)

        if self.rerun_enabled:
            if not rrd_output:
                rrd_output = str(_default_replay_rrd_path(prediction_file))
            if rrd_output:
                Path(rrd_output).parent.mkdir(parents=True, exist_ok=True)
            self.rerun_viz = RerunVisualizer(
                width=self._frame_width,
                height=self._frame_height,
                session_name=rerun_session_name,
                cam_intrinsics=self._camera_matrix,
                spawn=rerun_spawn,
                connect=rerun_connect,
                rrd_output=rrd_output,
                max_image_side=rerun_max_image_side,
                mesh_overlay_stride=rerun_mesh_overlay_stride,
                enable_mesh_overlay=rerun_mesh_overlay,
            )
            logger.info(
                "Rerun visualization enabled for offline replay "
                f"(log_stride={self.rerun_log_stride}, "
                f"max_side={rerun_max_image_side}, "
                f"mesh_overlay_stride={rerun_mesh_overlay_stride}, "
                f"mesh_overlay={'on' if rerun_mesh_overlay else 'off'})"
            )
            logger.info(f"Rerun replay recording will be saved to: {rrd_output}")
            self._init_video_reader()
            self._init_smpl_reconstruction(smpl_model_path)

        self.stats = {
            "input_count": 0,
            "valid_pose_count": 0,
            "publish_count": 0,
            "publish_intervals": deque(maxlen=500),
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
        }
        self._live_log_interval_s = 2.0
        self._live_last_log_perf = time.perf_counter()
        self._live_prev_stats = {
            "input_count": 0,
            "valid_pose_count": 0,
            "publish_count": 0,
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
        }
        self._final_stats_logged = False

        logger.info(
            "Offline prediction summary: "
            f"frames={self.summary['num_frames']}, "
            f"valid={self.summary['num_valid_frames']}, "
            f"duration={self.summary['duration_s']:.2f}s, "
            f"source_fps={self.publish_hz:.2f}"
        )
        logger.info(
            f"ZMQ publisher ready at {addr} using protocol v{self.zmq_protocol_version}"
        )

    def _init_video_reader(self):
        source_video = self.prediction_data.get("source_video")
        if source_video is None:
            raise RuntimeError(
                "Prediction archive does not contain source_video; cannot enable Rerun replay."
            )
        video_path = Path(_scalar_str(source_video))
        if not video_path.is_file():
            raise RuntimeError(f"Source video for Rerun replay not found: {video_path}")

        self._video_cap = cv2.VideoCapture(str(video_path))
        if not self._video_cap.isOpened():
            raise RuntimeError(f"Failed to open source video for Rerun replay: {video_path}")
        self._video_next_frame_idx = 0

    def _init_smpl_reconstruction(self, smpl_model_path):
        model_path = smpl_model_path
        if not model_path and "smpl_model_path" in self.prediction_data:
            model_path = _scalar_str(self.prediction_data["smpl_model_path"])

        if not model_path:
            logger.warning(
                "No SMPL model path available for offline replay Rerun; "
                "Rerun will show raw frames without body mesh."
            )
            return

        if self.camera_body_quats_xyzw is None or self.pred_cam_ts is None:
            logger.warning(
                "Prediction archive is missing camera_body_quats_xyzw or pred_cam_ts; "
                "Rerun will show raw frames without body mesh."
            )
            return

        logger.info(f"Loading SMPL model for offline replay Rerun: {model_path}")
        (
            self._smpl_model,
            self._smpl_faces,
            self._smpl_device,
            self._smpl_num_betas,
        ) = load_smpl_model(model_path)

    def _sleep_until(self, target_perf: float) -> None:
        while self.running:
            now_perf = time.perf_counter()
            wait_time = target_perf - now_perf
            if wait_time <= 0:
                return
            time.sleep(min(wait_time, 0.0015))

    def _read_video_frame(self, frame_idx: int):
        if self._video_cap is None:
            return None

        if frame_idx != self._video_next_frame_idx:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            self._video_next_frame_idx = int(frame_idx)

        ok, frame = self._video_cap.read()
        if not ok:
            logger.warning(f"Failed to read replay video frame {frame_idx}")
            return None

        self._video_next_frame_idx += 1
        return frame

    def _build_rerun_outputs(self, idx: int):
        if not bool(self.valid_mask[idx]):
            return []
        if (
            self._smpl_model is None
            or self._smpl_faces is None
            or self.camera_body_quats_xyzw is None
            or self.pred_cam_ts is None
        ):
            return []

        verts, joints = smpl_vertices_joints_from_pose(
            self.smpl_poses[idx],
            smpl_model=self._smpl_model,
            device=self._smpl_device,
            num_betas=self._smpl_num_betas,
            body_quat=self.camera_body_quats_xyzw[idx],
        )
        return [
            {
                "pred_vertices": verts,
                "pred_cam_t": self.pred_cam_ts[idx].astype(np.float32, copy=False),
                "pred_keypoints_3d": joints[:24].astype(np.float32, copy=False),
            }
        ]

    def _maybe_log_rerun_frame(self, idx: int, frame_idx: int):
        if self.rerun_viz is None:
            return

        frame_bgr = self._read_video_frame(frame_idx)
        if frame_bgr is None:
            return

        if self._rerun_frame_count % self.rerun_log_stride != 0:
            self._rerun_frame_count += 1
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        outputs = self._build_rerun_outputs(idx)
        try:
            self.rerun_viz.log_frame(
                frame_idx=frame_idx,
                frame_bgr=frame_bgr,
                frame_rgb=frame_rgb,
                outputs=outputs,
                faces=self._smpl_faces if self._smpl_faces is not None else np.zeros((0, 3), dtype=np.int32),
            )
        except Exception as exc:
            logger.warning(f"Disabling Rerun visualization after logging failure: {exc}")
            self.rerun_viz.close()
            self.rerun_viz = None
        finally:
            self._rerun_frame_count += 1

    def _maybe_log_live_stats(self, now_perf: float) -> None:
        elapsed = now_perf - self._live_last_log_perf
        if elapsed < self._live_log_interval_s:
            return

        curr = {
            "input_count": self.stats["input_count"],
            "valid_pose_count": self.stats["valid_pose_count"],
            "publish_count": self.stats["publish_count"],
            "publish_interpolated_count": self.stats["publish_interpolated_count"],
            "publish_fallback_count": self.stats["publish_fallback_count"],
        }
        prev = self._live_prev_stats

        d_input = curr["input_count"] - prev["input_count"]
        d_valid = curr["valid_pose_count"] - prev["valid_pose_count"]
        d_publish = curr["publish_count"] - prev["publish_count"]

        logger.info(
            "Offline replay: "
            f"input={d_input/elapsed:.1f}fps, "
            f"valid={d_valid/elapsed:.1f}fps, "
            f"publish={d_publish/elapsed:.1f}Hz"
        )

        self._live_prev_stats = curr
        self._live_last_log_perf = now_perf

    def _input_loop(self):
        self._playback_start_perf = time.perf_counter()

        for idx in range(self.relative_timestamps.shape[0]):
            if not self.running:
                break

            target_perf = self._playback_start_perf + float(self.relative_timestamps[idx])
            self._sleep_until(target_perf)
            if not self.running:
                break

            now_perf = time.perf_counter()
            frame_idx = int(self.frame_indices[idx])
            self.stats["input_count"] += 1

            self._maybe_log_rerun_frame(idx, frame_idx)
            if self.rerun_viz is not None:
                try:
                    is_valid = bool(self.valid_mask[idx])
                    self.rerun_viz.log_publish_state(
                        frame_idx=frame_idx,
                        timestamp=float(self.relative_timestamps[idx]),
                        status_name=status_code_to_name(int(self.status_codes[idx])),
                        status_code=int(self.status_codes[idx]),
                        publish_enabled=is_valid,
                        num_persons=int(self.num_persons[idx]),
                        body_quat=self.body_quats[idx] if is_valid else None,
                        smpl_joints=self.smpl_joints[idx] if is_valid else None,
                        smpl_pose=self.smpl_poses[idx] if is_valid else None,
                        joint_pos=self._zero_joint_state if is_valid else None,
                        joint_vel=self._zero_joint_state if is_valid else None,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Disabling Rerun publish-state logging after failure: {exc}"
                    )
                    self.rerun_viz.close()
                    self.rerun_viz = None

            if not bool(self.valid_mask[idx]):
                status_code = int(self.status_codes[idx])
                if self._publish_enabled or status_code != self._last_pause_code:
                    logger.info(
                        "Pausing pose publishing for offline playback: "
                        f"frame={frame_idx}, "
                        f"status={status_code_to_name(status_code)}, "
                        f"num_persons={int(self.num_persons[idx])}"
                    )
                self._publish_enabled = False
                self._last_pause_code = status_code
                self._maybe_log_live_stats(now_perf)
                continue

            if not self._publish_enabled:
                logger.info(
                    f"Resuming pose publishing from offline predictions at frame={frame_idx}"
                )
            self._publish_enabled = True
            self._last_pause_code = None

            self.publisher.publish(
                self.body_quats[idx],
                self.smpl_joints[idx],
                self.smpl_poses[idx],
            )
            self.stats["valid_pose_count"] += 1
            self.stats["publish_count"] += 1
            if self._last_publish_perf is not None:
                self.stats["publish_intervals"].append(now_perf - self._last_publish_perf)
            self._last_publish_perf = now_perf

            self._maybe_log_live_stats(now_perf)

        self.finished_input = True

    def _log_final_stats(self):
        if self._final_stats_logged:
            return

        pub_msg = "Publish n/a"
        if self.stats["publish_intervals"]:
            pub_hz = 1.0 / np.mean(self.stats["publish_intervals"])
            pub_msg = f"Publish {pub_hz:.1f}Hz (source {self.publish_hz:.1f}Hz)"

        logger.info(
            f"Final stats: {pub_msg}, "
            f"input={self.stats['input_count']}, "
            f"valid={self.stats['valid_pose_count']}, "
            f"published={self.stats['publish_count']}"
        )
        self._final_stats_logged = True

    def start(self):
        logger.info("Starting offline prediction publisher (Press Ctrl+C to stop)")
        self.running = True

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        while self.running:
            if self.finished_input and (
                self.input_thread is None or not self.input_thread.is_alive()
            ):
                self.running = False
                break
            time.sleep(0.05)

        if self.input_thread is not None and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)

        self._log_final_stats()

    def stop(self):
        if self._closed:
            return

        self.running = False
        if self.input_thread is not None and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        self._log_final_stats()
        if self.rerun_viz is not None:
            self.rerun_viz.close()
            self.rerun_viz = None
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        self.publisher.close()
        self._closed = True
