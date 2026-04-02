#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")

import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation

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

from mocap.core.gravity_alignment import build_camera_to_world_rotation
from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner
from mocap.core.setup_estimator import build_default_estimator
from mocap.realtime.publisher import SONIC_NUM_JOINTS
from mocap.utils.offline_prediction_io import (
    STATUS_CONVERTER_ERROR,
    STATUS_ESTIMATOR_ERROR,
    STATUS_MISSING_KEYS,
    STATUS_MULTI_PERSON,
    STATUS_NO_PERSON,
    STATUS_UPRIGHT_CALIBRATING,
    STATUS_VALID,
    save_prediction_archive,
    save_prediction_summary,
    status_code_to_name,
)
from mocap.utils.pose_protocol import prepare_publish_pose
from mocap.utils.rerun_visualizer import RerunVisualizer
from mocap.utils.upright_leveler import InitialUprightLeveler
from mocap.utils.video_source import create_video_source


YOLO_MODEL_PATH = "checkpoints/yolo/yolo11m-pose.engine"
MHR_MODEL_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
FOV_MODEL_SIZE = "s"
FOV_RESOLUTION_LEVEL = 0
FOV_FIXED_SIZE = 512
FOV_FAST_MODE = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-view offline prediction on every video frame and save publish-ready poses."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path.")
    parser.add_argument(
        "--intrinsics",
        type=str,
        required=True,
        help="Camera metadata path for the input video.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output .npz path. Default: output/offline_predictions/<video_stem>_predictions.npz",
    )
    parser.add_argument(
        "--hand-box-source",
        type=str,
        default="yolo_pose",
        choices=["body_decoder", "yolo_pose"],
        help="Hand box source passed into SAM 3D Body.",
    )
    parser.add_argument(
        "--use-compile",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set USE_COMPILE for this run.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode when --use-compile=1.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        choices=[256, 384, 512],
        help="Image size for SAM3D model.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=YOLO_MODEL_PATH,
        help="YOLO pose model path.",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        required=True,
        help="SMPL model pickle file.",
    )
    parser.add_argument(
        "--nn-model-dir",
        type=str,
        required=True,
        help="NN fusion model directory.",
    )
    parser.add_argument(
        "--mhr2smpl-mapping-path",
        type=str,
        required=True,
        help="Path to mhr2smpl_mapping.npz.",
    )
    parser.add_argument(
        "--mhr-mesh-path",
        type=str,
        default=None,
        help="Path to MHR mesh PLY.",
    )
    parser.add_argument(
        "--mhr-model-path",
        type=str,
        default=MHR_MODEL_PATH,
        help="Path to MHR TorchScript model.",
    )
    parser.add_argument(
        "--smoother-dir",
        type=str,
        default=None,
        help="Smoother checkpoint directory.",
    )
    parser.add_argument(
        "--imu-level-init-frames",
        type=int,
        default=15,
        help="Number of valid pose frames used for initial upright leveling.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Log progress every N frames.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Enable Rerun visualization during offline prediction.",
    )
    parser.add_argument(
        "--rerun-session-name",
        type=str,
        default="fast_sam_3d_body_offline_predict",
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

    if args.imu_level_init_frames < 0:
        parser.error("--imu-level-init-frames must be >= 0")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0")
    if args.rerun_log_stride <= 0:
        parser.error("--rerun-log-stride must be > 0")

    return args


def resolve_output_path(video_path: str, output_path: str) -> Path:
    if output_path:
        return Path(output_path)
    video_stem = Path(video_path).stem
    return Path("output/offline_predictions") / f"{video_stem}_predictions.npz"


def resolve_rerun_output_path(prediction_output_path: Path, requested_rrd_output: str) -> Path:
    if requested_rrd_output:
        return Path(requested_rrd_output)
    stem = prediction_output_path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return prediction_output_path.with_name(f"{stem}_replay.rrd")


def warmup_estimator(estimator, width: int, height: int):
    dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    warmup_bbox = np.array(
        [[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32
    )
    for _ in range(2):
        _ = estimator.process_one_image(
            dummy_img,
            bboxes=warmup_bbox,
            hand_box_source="body_decoder",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def compute_body_quat(global_rot):
    global_rot = np.asarray(global_rot, dtype=np.float64).reshape(3)
    rot = Rotation.from_euler("ZYX", global_rot)
    x180 = Rotation.from_euler("x", 180.0, degrees=True)
    return (x180 * rot).as_quat().astype(np.float64)


def build_publish_rotation(gravity_direction):
    r_world_cam = build_camera_to_world_rotation(gravity_direction)
    r_zup_adjustment = np.array(
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64
    )
    return r_zup_adjustment @ r_world_cam


def main():
    args = parse_args()
    output_path = resolve_output_path(args.video, args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rrd_output_path = None
    if args.rerun:
        rrd_output_path = resolve_rerun_output_path(output_path, args.rrd_output)
        rrd_output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["USE_COMPILE"] = str(args.use_compile)
    os.environ["COMPILE_MODE"] = args.compile_mode

    logger.info(f"Loading video source: {args.video}")
    video_source = create_video_source(
        "video",
        video_path=args.video,
        intrinsics_path=args.intrinsics,
        loop=False,
    )

    frame_width, frame_height = video_source.get_frame_size()
    fps = float(video_source.fps)
    cam_intrinsics_np = video_source.get_camera_intrinsics()
    cam_intrinsics = None
    if cam_intrinsics_np is not None:
        cam_intrinsics = torch.from_numpy(np.asarray(cam_intrinsics_np, dtype=np.float32))

    gravity_direction = np.asarray(
        video_source.get_gravity_direction(), dtype=np.float64
    ).reshape(3)
    r_world_cam = build_publish_rotation(gravity_direction)

    logger.info("Loading SAM 3D estimator...")
    estimator = build_default_estimator(
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
        fov_model_size=FOV_MODEL_SIZE,
        fov_resolution_level=FOV_RESOLUTION_LEVEL,
        fov_fixed_size=FOV_FIXED_SIZE,
        fov_fast_mode=FOV_FAST_MODE,
    )
    logger.info("Warming up estimator...")
    warmup_estimator(estimator, frame_width, frame_height)

    rerun_viz = None
    rerun_frame_idx = 0
    rerun_log_stride = max(1, int(args.rerun_log_stride))
    if args.rerun:
        logger.info("Initializing Rerun visualization...")
        rerun_viz = RerunVisualizer(
            width=frame_width,
            height=frame_height,
            session_name=args.rerun_session_name,
            cam_intrinsics=cam_intrinsics_np[0] if cam_intrinsics_np is not None else None,
            spawn=args.rerun_spawn,
            connect=args.rerun_connect,
            rrd_output=str(rrd_output_path) if rrd_output_path is not None else "",
            max_image_side=args.rerun_max_image_side,
            mesh_overlay_stride=args.rerun_mesh_overlay_stride,
            enable_mesh_overlay=args.rerun_mesh_overlay,
        )
        logger.info(
            "Rerun visualization enabled "
            f"(log_stride={rerun_log_stride}, "
            f"max_side={args.rerun_max_image_side}, "
            f"mesh_overlay_stride={args.rerun_mesh_overlay_stride}, "
            f"mesh_overlay={'on' if args.rerun_mesh_overlay else 'off'})"
        )
        logger.info(f"Rerun recording will be saved to: {rrd_output_path}")

    logger.info("Loading MHR-to-SMPL fusion runner...")
    fusion_runner = MultiViewFusionRunner(
        smpl_model_path=args.smpl_model_path,
        model_dir=args.nn_model_dir,
        mapping_path=args.mhr2smpl_mapping_path,
        mhr_mesh_path=args.mhr_mesh_path,
        mhr_model_path=args.mhr_model_path,
        smoother_dir=args.smoother_dir,
    )

    upright_leveler = InitialUprightLeveler(args.imu_level_init_frames)
    last_level_log_count = 0

    timestamps = []
    frame_indices = []
    valid_mask = []
    status_codes = []
    num_persons_list = []
    body_quats = []
    camera_body_quats_xyzw = []
    pred_cam_ts = []
    smpl_joints = []
    smpl_poses = []

    zero_body_quat = np.zeros((4,), dtype=np.float64)
    zero_camera_body_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    zero_pred_cam_t = np.zeros((3,), dtype=np.float64)
    zero_smpl_joints = np.zeros((24, 3), dtype=np.float64)
    zero_smpl_pose = np.zeros((21, 3), dtype=np.float64)
    zero_joint_state = np.zeros((SONIC_NUM_JOINTS,), dtype=np.float64)

    processed_frames = 0
    first_source_ts = None
    start_perf = time.perf_counter()

    try:
        while True:
            frame, frame_timestamp = video_source.get_frame()
            if frame is None:
                break
            if frame_timestamp is None:
                continue

            if first_source_ts is None:
                first_source_ts = float(frame_timestamp)
            rel_timestamp = float(frame_timestamp - first_source_ts)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            infer_ms = float("nan")
            convert_ms = float("nan")
            outputs = []
            num_persons = -1
            status_code = STATUS_VALID
            is_valid = False
            body_quat = zero_body_quat
            camera_body_quat_xyzw = zero_camera_body_quat_xyzw
            pred_cam_t = zero_pred_cam_t
            smpl_joint = zero_smpl_joints
            smpl_pose = zero_smpl_pose

            try:
                outputs = estimator.process_one_image(
                    frame_rgb,
                    cam_int=cam_intrinsics,
                    hand_box_source=args.hand_box_source,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                num_persons = len(outputs)
            except Exception as exc:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                logger.warning(
                    f"Estimator failed on frame {processed_frames}: {type(exc).__name__}: {exc}"
                )
                status_code = STATUS_ESTIMATOR_ERROR
            else:
                if rerun_viz is not None:
                    current_rerun_frame_idx = rerun_frame_idx
                    if current_rerun_frame_idx % rerun_log_stride == 0:
                        try:
                            rerun_viz.log_frame(
                                frame_idx=current_rerun_frame_idx,
                                frame_bgr=frame,
                                frame_rgb=frame_rgb,
                                outputs=outputs,
                                faces=estimator.faces,
                            )
                        except Exception as exc:
                            logger.warning(
                                f"Disabling Rerun visualization after logging failure: {exc}"
                            )
                            rerun_viz.close()
                            rerun_viz = None
                    rerun_frame_idx += 1

                if num_persons == 0:
                    status_code = STATUS_NO_PERSON
                elif num_persons != 1:
                    status_code = STATUS_MULTI_PERSON
                else:
                    out = outputs[0]
                    required = ("pred_vertices", "pred_cam_t", "global_rot")
                    missing = [key for key in required if key not in out]
                    if missing:
                        logger.warning(
                            f"Frame {processed_frames} missing model outputs: {missing}"
                        )
                        status_code = STATUS_MISSING_KEYS
                    else:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        convert_start = time.perf_counter()
                        try:
                            pred_vertices = np.asarray(
                                out["pred_vertices"], dtype=np.float32
                            )
                            pred_cam_t = np.asarray(out["pred_cam_t"], dtype=np.float32)
                            body_quat_xyzw = compute_body_quat(out["global_rot"])
                            camera_body_quat_xyzw = np.asarray(
                                body_quat_xyzw, dtype=np.float64
                            )
                            smpl_pose_pred, canonical_joints, _betas, _weights = (
                                fusion_runner.infer([(pred_vertices, pred_cam_t)])
                            )
                            body_quat, smpl_joint, smpl_pose = prepare_publish_pose(
                                body_quat_xyzw,
                                canonical_joints,
                                smpl_pose_pred,
                                r_world_cam,
                            )

                            if upright_leveler.enabled and not upright_leveler.ready:
                                upright_leveler.update(body_quat)
                                collected = upright_leveler.num_collected
                                if collected != last_level_log_count and (
                                    collected == 1
                                    or collected == upright_leveler.calibration_frames
                                    or collected % 5 == 0
                                ):
                                    logger.info(
                                        "Collecting initial upright calibration "
                                        f"({collected}/{upright_leveler.calibration_frames})"
                                    )
                                    last_level_log_count = collected

                                if not upright_leveler.ready:
                                    status_code = STATUS_UPRIGHT_CALIBRATING
                                else:
                                    logger.info(
                                        "Initial upright calibration locked "
                                        f"(removed ~{upright_leveler.estimated_tilt_deg:.2f} deg startup tilt)"
                                    )

                            if status_code == STATUS_VALID and upright_leveler.enabled:
                                body_quat = upright_leveler.apply(body_quat)

                            if status_code == STATUS_VALID:
                                is_valid = True
                        except Exception as exc:
                            logger.warning(
                                f"Pose conversion failed on frame {processed_frames}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            status_code = STATUS_CONVERTER_ERROR
                            body_quat = zero_body_quat
                            smpl_joint = zero_smpl_joints
                            smpl_pose = zero_smpl_pose
                        finally:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            convert_ms = (time.perf_counter() - convert_start) * 1000.0

            timestamps.append(rel_timestamp)
            frame_indices.append(processed_frames)
            valid_mask.append(is_valid)
            status_codes.append(status_code)
            num_persons_list.append(num_persons)
            body_quats.append(np.asarray(body_quat, dtype=np.float64))
            camera_body_quats_xyzw.append(
                np.asarray(camera_body_quat_xyzw, dtype=np.float64)
            )
            pred_cam_ts.append(np.asarray(pred_cam_t, dtype=np.float64))
            smpl_joints.append(np.asarray(smpl_joint, dtype=np.float64))
            smpl_poses.append(np.asarray(smpl_pose, dtype=np.float64))

            if rerun_viz is not None:
                try:
                    rerun_viz.log_publish_state(
                        frame_idx=processed_frames,
                        timestamp=rel_timestamp,
                        status_name=status_code_to_name(status_code),
                        status_code=status_code,
                        publish_enabled=is_valid,
                        num_persons=num_persons,
                        body_quat=body_quat if is_valid else None,
                        smpl_joints=smpl_joint if is_valid else None,
                        smpl_pose=smpl_pose if is_valid else None,
                        joint_pos=zero_joint_state if is_valid else None,
                        joint_vel=zero_joint_state if is_valid else None,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Disabling Rerun publish-state logging after failure: {exc}"
                    )
                    rerun_viz.close()
                    rerun_viz = None

            if (
                processed_frames == 0
                or (processed_frames + 1) % args.log_every == 0
                or status_code not in (STATUS_VALID, STATUS_UPRIGHT_CALIBRATING)
            ):
                infer_ms_str = (
                    f"{infer_ms:.1f}" if np.isfinite(infer_ms) else "n/a"
                )
                convert_ms_str = (
                    f"{convert_ms:.1f}" if np.isfinite(convert_ms) else "n/a"
                )
                logger.info(
                    f"[frame {processed_frames}] "
                    f"t={rel_timestamp:.3f}s "
                    f"persons={num_persons} "
                    f"status={status_code_to_name(status_code)} "
                    f"infer={infer_ms_str}ms "
                    f"convert={convert_ms_str}ms"
                )

            processed_frames += 1

    finally:
        if rerun_viz is not None:
            rerun_viz.close()
        video_source.release()

    prediction_data = {
        "prediction_version": 1,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "status_codes": np.asarray(status_codes, dtype=np.int32),
        "num_persons": np.asarray(num_persons_list, dtype=np.int32),
        "body_quats": np.asarray(body_quats, dtype=np.float64),
        "camera_body_quats_xyzw": np.asarray(
            camera_body_quats_xyzw, dtype=np.float64
        ),
        "pred_cam_ts": np.asarray(pred_cam_ts, dtype=np.float64),
        "smpl_joints": np.asarray(smpl_joints, dtype=np.float64),
        "smpl_poses": np.asarray(smpl_poses, dtype=np.float64),
        "fps": fps,
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "camera_matrix": (
            np.asarray(cam_intrinsics_np[0], dtype=np.float32)
            if cam_intrinsics_np is not None
            else np.eye(3, dtype=np.float32)
        ),
        "gravity": gravity_direction,
        "imu_level_init_frames": int(args.imu_level_init_frames),
        "source_video": str(Path(args.video).resolve()),
        "source_intrinsics": str(Path(args.intrinsics).resolve()),
        "smpl_model_path": str(Path(args.smpl_model_path).resolve()),
    }

    archive_path = save_prediction_archive(output_path, prediction_data)
    summary_path = save_prediction_summary(summary_path, prediction_data)

    elapsed = time.perf_counter() - start_perf
    valid_count = int(np.asarray(valid_mask, dtype=bool).sum())
    logger.info(
        f"Saved offline predictions to {archive_path} "
        f"(frames={processed_frames}, valid={valid_count}, elapsed={elapsed:.2f}s)"
    )
    logger.info(f"Saved summary to {summary_path}")
    if rrd_output_path is not None:
        logger.info(f"Saved Rerun recording to {rrd_output_path}")


if __name__ == "__main__":
    main()
