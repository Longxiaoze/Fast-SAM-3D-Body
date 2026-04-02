#!/usr/bin/env python3
"""
Video demo for Fast SAM 3D Body with Rerun visualization.

Recommended Rerun install (same pin used in GENMO):
    pip install rerun-sdk==0.19.1

Example:
    python demo_video_rerun.py \
        --video_path /path/to/video.mp4 \
        --detector yolo_pose \
        --detector_model ./checkpoints/yolo/yolo11m-pose.engine \
        --hand_box_source yolo_pose \
        --num-frames 300 \
        --rrd-output output/video_demo.rrd
"""

import argparse
import os
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from notebook.utils import setup_sam_3d_body, setup_visualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
PERSON_COLORS = np.array(
    [
        [255, 80, 80],
        [80, 220, 255],
        [120, 255, 120],
        [255, 200, 80],
        [220, 120, 255],
        [255, 120, 180],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fast SAM 3D Body on a video and visualize results in Rerun."
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video path.")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam-3d-body-dinov3",
        choices=["facebook/sam-3d-body-dinov3", "facebook/sam-3d-body-vith"],
        help="Model selection.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo_pose",
        choices=["vitdet", "yolo", "yolo_pose"],
        help="Person detector.",
    )
    parser.add_argument(
        "--hand_box_source",
        type=str,
        default="yolo_pose",
        choices=["body_decoder", "yolo_pose"],
        help="Hand box source.",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="./checkpoints/yolo/yolo11m-pose.engine",
        help="Detector model path.",
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default="./checkpoints/sam-3d-body-dinov3",
        help="Local checkpoint directory.",
    )
    parser.add_argument(
        "--local_mhr_path",
        type=str,
        default="",
        help="Optional explicit MHR model path.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start processing from this frame index.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every N-th frame.",
    )
    parser.add_argument(
        "--max_frames",
        "--num-frames",
        type=int,
        dest="max_frames",
        default=0,
        help="Maximum number of processed frames. Use 0 for the full video.",
    )
    parser.add_argument(
        "--rrd-output",
        type=str,
        default="",
        help="Optional path to save the Rerun recording as a .rrd file.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Override the model image size via IMG_SIZE.",
    )
    parser.add_argument(
        "--use-compile",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set USE_COMPILE for this run. 0 is recommended for faster startup when debugging videos.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode when --use-compile=1.",
    )
    parser.add_argument(
        "--session_name",
        type=str,
        default="fast_sam_3d_body_video",
        help="Rerun session name.",
    )
    parser.add_argument(
        "--rerun",
        dest="rerun",
        action="store_true",
        help="Enable Rerun logging.",
    )
    parser.add_argument(
        "--no_rerun",
        dest="rerun",
        action="store_false",
        help="Disable Rerun logging.",
    )
    parser.add_argument(
        "--rerun_spawn",
        dest="rerun_spawn",
        action="store_true",
        help="Spawn a local Rerun viewer.",
    )
    parser.add_argument(
        "--no_rerun_spawn",
        dest="rerun_spawn",
        action="store_false",
        help="Do not auto-spawn a Rerun viewer.",
    )
    parser.add_argument(
        "--rerun_connect",
        type=str,
        default=None,
        help="Connect to an existing Rerun viewer, e.g. 127.0.0.1:9876.",
    )
    parser.set_defaults(rerun=True, rerun_spawn=True)
    return parser.parse_args()


def build_skeleton_links() -> List[Tuple[int, int]]:
    keypoint_name_to_id = {
        info["name"]: idx for idx, info in mhr70_pose_info["keypoint_info"].items()
    }
    links = []
    for _, info in mhr70_pose_info["skeleton_info"].items():
        start_name, end_name = info["link"]
        if start_name in keypoint_name_to_id and end_name in keypoint_name_to_id:
            links.append((keypoint_name_to_id[start_name], keypoint_name_to_id[end_name]))
    return links


SKELETON_LINKS = build_skeleton_links()


def init_rerun_if_needed(args, width: int, height: int):
    if not args.rerun:
        return None

    try:
        import rerun as rr  # type: ignore
        import rerun.blueprint as rrb  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "rerun is not installed in the current environment. "
            "Install it with `pip install rerun-sdk==0.19.1`."
        ) from e

    spawn_viewer = bool(args.rerun_spawn and args.rerun_connect is None)
    rr.init(args.session_name, spawn=spawn_viewer)
    if args.rerun_connect is not None:
        rr.connect(addr=args.rerun_connect)
    if args.rrd_output:
        rr.save(args.rrd_output)

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log("world/camera", rr.ViewCoordinates.RDF, static=True)
    rr.log("world/camera", rr.Transform3D(translation=[0.0, 0.0, 0.0]), static=True)
    rr.log("world/body", rr.ViewCoordinates.RDF, static=True)

    portrait_layout = bool(height > width)
    if portrait_layout:
        layout = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="world/camera",
                    contents=["world/camera/image_raw"],
                    name="1 Raw",
                ),
                rrb.Spatial2DView(
                    origin="world/camera",
                    contents=["world/camera/image_2d_overlay"],
                    name="2 2D Overlay",
                ),
                rrb.Spatial2DView(
                    origin="world/camera",
                    contents=["world/camera/image_mesh_overlay"],
                    name="3 Mesh Overlay",
                ),
            ),
            rrb.Spatial3DView(
                origin="world/body",
                contents=[
                    "world/body/mesh",
                    "world/body/joints3d",
                    "world/body/joints3d_skeleton",
                ],
                name="4 Body3D",
            ),
        )
    else:
        layout = rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin="world/camera",
                    contents=["world/camera/image_raw"],
                    name="1 Raw",
                ),
                rrb.Spatial2DView(
                    origin="world/camera",
                    contents=["world/camera/image_2d_overlay"],
                    name="2 2D Overlay",
                ),
                rrb.Spatial2DView(
                    origin="world/camera",
                    contents=["world/camera/image_mesh_overlay"],
                    name="3 Mesh Overlay",
                ),
            ),
            rrb.Spatial3DView(
                origin="world/body",
                contents=[
                    "world/body/mesh",
                    "world/body/joints3d",
                    "world/body/joints3d_skeleton",
                ],
                name="4 Body3D",
            ),
        )

    rr.send_blueprint(rrb.Blueprint(layout, collapse_panels=True))
    return rr


def draw_2d_overlay_image(frame_bgr: np.ndarray, outputs: Sequence[dict], visualizer) -> np.ndarray:
    overlay = frame_bgr.copy()
    for person_id, person in enumerate(outputs):
        keypoints_2d = person["pred_keypoints_2d"]
        if keypoints_2d is not None and len(keypoints_2d) > 0:
            keypoints_2d_vis = np.concatenate(
                [
                    keypoints_2d.astype(np.float32, copy=False),
                    np.ones((keypoints_2d.shape[0], 1), dtype=np.float32),
                ],
                axis=-1,
            )
            overlay = visualizer.draw_skeleton(overlay, keypoints_2d_vis)

        bbox = person.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"P{person_id}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        for hand_key, color, prefix in (
            ("lhand_bbox", (255, 0, 0), "L"),
            ("rhand_bbox", (0, 0, 255), "R"),
        ):
            hand_bbox = person.get(hand_key)
            if hand_bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in hand_bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"{prefix}{person_id}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return overlay


def render_mesh_overlay_image(frame_bgr: np.ndarray, outputs: Sequence[dict], faces: np.ndarray) -> np.ndarray:
    if not outputs:
        return frame_bgr.copy()

    mesh_overlay = frame_bgr.copy()
    all_depths = np.stack([person["pred_cam_t"] for person in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    for person in outputs_sorted:
        vertices = person.get("pred_vertices")
        cam_t = person.get("pred_cam_t")
        if vertices is None or cam_t is None:
            continue
        if len(vertices) == 0 or np.any(np.isnan(vertices)) or np.any(np.isnan(cam_t)):
            continue

        renderer = Renderer(focal_length=person["focal_length"], faces=faces)
        mesh_overlay = (
            renderer(
                vertices,
                cam_t,
                mesh_overlay,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

    return mesh_overlay


def combine_meshes(outputs: Sequence[dict], faces: np.ndarray):
    vertices_all = []
    faces_all = []
    vertex_offset = 0

    for person in outputs:
        vertices = person.get("pred_vertices")
        cam_t = person.get("pred_cam_t")
        if vertices is None or cam_t is None:
            continue
        if len(vertices) == 0 or np.any(np.isnan(vertices)) or np.any(np.isnan(cam_t)):
            continue

        vertices_world = vertices.astype(np.float32, copy=False) + cam_t.astype(
            np.float32, copy=False
        )[None, :]
        vertices_all.append(vertices_world)
        faces_all.append(faces.astype(np.int32, copy=False) + vertex_offset)
        vertex_offset += vertices_world.shape[0]

    if not vertices_all:
        return None, None

    return np.concatenate(vertices_all, axis=0), np.concatenate(faces_all, axis=0)


def combine_joints_and_skeleton(outputs: Sequence[dict]):
    points_all = []
    point_colors_all = []
    strips_all = []
    strip_colors_all = []

    for person_id, person in enumerate(outputs):
        joints = person.get("pred_keypoints_3d")
        cam_t = person.get("pred_cam_t")
        if joints is None or cam_t is None:
            continue
        if len(joints) == 0 or np.any(np.isnan(joints)) or np.any(np.isnan(cam_t)):
            continue

        joints_world = joints.astype(np.float32, copy=False) + cam_t.astype(
            np.float32, copy=False
        )[None, :]
        color = PERSON_COLORS[person_id % len(PERSON_COLORS)]

        points_all.append(joints_world)
        point_colors_all.append(np.tile(color[None, :], (joints_world.shape[0], 1)))

        for start_idx, end_idx in SKELETON_LINKS:
            if start_idx >= joints_world.shape[0] or end_idx >= joints_world.shape[0]:
                continue
            segment = joints_world[[start_idx, end_idx]]
            if np.any(np.isnan(segment)):
                continue
            strips_all.append(segment)
            strip_colors_all.append(color)

    if not points_all:
        return None, None, [], None

    line_colors = (
        np.asarray(strip_colors_all, dtype=np.uint8)
        if strip_colors_all
        else np.zeros((0, 3), dtype=np.uint8)
    )
    return (
        np.concatenate(points_all, axis=0),
        np.concatenate(point_colors_all, axis=0),
        strips_all,
        line_colors,
    )


def log_rerun_frame(
    rr,
    frame_idx: int,
    frame_rgb: np.ndarray,
    overlay_bgr: np.ndarray,
    mesh_overlay_bgr: np.ndarray,
    outputs: Sequence[dict],
    faces: np.ndarray,
    width: int,
    height: int,
):
    rr.set_time_sequence("frame", frame_idx)

    focal_length = float(max(width, height))
    valid_focals = [
        float(person["focal_length"])
        for person in outputs
        if person.get("focal_length") is not None and not np.isnan(person["focal_length"])
    ]
    if valid_focals:
        focal_length = float(np.median(valid_focals))

    K = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rr.log(
        "world/camera",
        rr.Pinhole(image_from_camera=K, resolution=[int(width), int(height)]),
    )

    rr.log("world/camera/image_raw", rr.Image(frame_rgb))
    rr.log("world/camera/image_2d_overlay", rr.Image(overlay_bgr[..., ::-1]))
    rr.log("world/camera/image_mesh_overlay", rr.Image(mesh_overlay_bgr[..., ::-1]))

    vertices, triangle_indices = combine_meshes(outputs, faces)
    if vertices is None:
        rr.log(
            "world/body/mesh",
            rr.Mesh3D(
                vertex_positions=np.zeros((0, 3), dtype=np.float32),
                triangle_indices=np.zeros((0, 3), dtype=np.int32),
            ),
        )
    else:
        rr.log(
            "world/body/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangle_indices,
                albedo_factor=[200, 200, 200, 255],
            ),
        )

    joints, joint_colors, strips3d, strip_colors = combine_joints_and_skeleton(outputs)
    if joints is None:
        rr.log(
            "world/body/joints3d",
            rr.Points3D(np.zeros((0, 3), dtype=np.float32)),
        )
        rr.log("world/body/joints3d_skeleton", rr.LineStrips3D([]))
    else:
        rr.log(
            "world/body/joints3d",
            rr.Points3D(
                joints,
                colors=joint_colors,
                radii=np.full(len(joints), 0.01, dtype=np.float32),
            ),
        )
        rr.log(
            "world/body/joints3d_skeleton",
            rr.LineStrips3D(
                strips3d,
                colors=strip_colors,
                radii=np.full(len(strips3d), 0.004, dtype=np.float32),
            ),
        )


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print("=" * 60)
    print("Fast SAM 3D Body Video Demo (Rerun)")
    print("=" * 60)
    print(f"Video: {args.video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}" if fps > 0 else "FPS: unknown")
    print(f"Frames: {total_frames}" if total_frames > 0 else "Frames: unknown")
    print(f"Detector: {args.detector} ({args.detector_model})")
    print(f"Hand Box Source: {args.hand_box_source}")
    print(f"Local Checkpoint: {args.local_checkpoint or 'HuggingFace'}")
    print(f"Rerun Recording: {args.rrd_output}" if args.rrd_output else "Rerun Recording: disabled")

    os.environ["USE_COMPILE"] = str(args.use_compile)
    os.environ["COMPILE_MODE"] = args.compile_mode
    if args.img_size > 0:
        os.environ["IMG_SIZE"] = str(args.img_size)

    rr = init_rerun_if_needed(args, width=width, height=height)

    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
        local_mhr_path=args.local_mhr_path,
        device=None,
    )
    visualizer = setup_visualizer()

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    processed_frames = 0
    input_frame_idx = args.start_frame
    demo_start = time.perf_counter()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            current_input_idx = input_frame_idx
            input_frame_idx += 1

            if current_input_idx < args.start_frame:
                continue
            if args.frame_stride > 1 and (
                (current_input_idx - args.start_frame) % args.frame_stride != 0
            ):
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = estimator.process_one_image(
                frame_rgb,
                hand_box_source=args.hand_box_source,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time_s = time.perf_counter() - t0

            overlay_bgr = draw_2d_overlay_image(frame_bgr, outputs, visualizer)
            mesh_overlay_bgr = render_mesh_overlay_image(
                frame_bgr, outputs, estimator.faces
            )

            if rr is not None:
                log_rerun_frame(
                    rr=rr,
                    frame_idx=processed_frames,
                    frame_rgb=frame_rgb,
                    overlay_bgr=overlay_bgr,
                    mesh_overlay_bgr=mesh_overlay_bgr,
                    outputs=outputs,
                    faces=estimator.faces,
                    width=width,
                    height=height,
                )

            print(
                f"[frame {current_input_idx}] "
                f"persons={len(outputs)} "
                f"infer_ms={infer_time_s * 1000.0:.2f}"
            )

            processed_frames += 1
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

    finally:
        cap.release()

    elapsed = time.perf_counter() - demo_start
    print("=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Processed frames: {processed_frames}")
    print(f"Elapsed: {elapsed:.2f}s")
    if processed_frames > 0:
        print(f"Average processed FPS: {processed_frames / max(elapsed, 1e-6):.2f}")


if __name__ == "__main__":
    main()
