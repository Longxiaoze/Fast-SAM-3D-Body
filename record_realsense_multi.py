#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def _open_video_writer(base_path: Path, fps: int, width: int, height: int):
    candidates = [
        ("avc1", ".mp4"),
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
    ]

    for codec, suffix in candidates:
        video_path = base_path.with_suffix(suffix)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, video_path, codec
        writer.release()
        video_path.unlink(missing_ok=True)

    tried = ", ".join(f"{codec}{suffix}" for codec, suffix in candidates)
    raise RuntimeError(
        "Failed to open video writer. "
        f"Tried: {tried}. "
        "Install a compatible ffmpeg/gstreamer codec or use a build of OpenCV with mp4v/MJPG support."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Record RGB video from multiple RealSense D435i/D455 cameras"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/records/<timestamp>)",
    )
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--imu-samples",
        type=int,
        default=100,
        help="Number of IMU samples for gravity calibration (default: 100)",
    )
    args = parser.parse_args()

    if args.imu_samples < 10:
        parser.error("--imu-samples must be at least 10")

    import time as _time
    output_dir = Path(args.output_dir) if args.output_dir else Path("output/records") / _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense devices found.")

    print(f"Found {len(devices)} RealSense devices.")

    pipelines = {}
    writers = {}
    intrinsics_data = {}
    video_paths = {}

    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"[{i}] {name} (S/N: {serial})")

        pipe = rs.pipeline(ctx)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

        try:
            profile = pipe.start(config)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to start pipeline for {serial}: {e}") from e

        try:
            writer, video_path, video_codec = _open_video_writer(
                output_dir / serial, args.fps, args.width, args.height
            )
        except RuntimeError:
            pipe.stop()
            raise

        pipelines[serial] = pipe
        writers[serial] = writer
        video_paths[serial] = video_path

        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        intrinsics_data[serial] = {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "width": args.width,
            "height": args.height,
            "camera_matrix": [
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1],
            ],
        }
        print(f"  Initialized {serial}")
        print(f"    Video: {video_path}")
        print(f"    Video codec: {video_codec}")

    print(f"\nCalibrating gravity ({args.imu_samples} samples per camera)...")
    accel_samples = {sn: [] for sn in pipelines}

    while any(len(s) < args.imu_samples for s in accel_samples.values()):
        for sn, pipe in pipelines.items():
            if len(accel_samples[sn]) >= args.imu_samples:
                continue
            frames = pipe.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                d = accel_frame.as_motion_frame().get_motion_data()
                accel_samples[sn].append([d.x, d.y, d.z])

    for sn in pipelines:
        accel_array = np.array(accel_samples[sn], dtype=np.float64)
        gravity_avg = -np.mean(accel_array, axis=0)
        gravity_norm = np.linalg.norm(gravity_avg)
        if gravity_norm < 1e-6:
            print(f"  Warning: failed to calibrate gravity for {sn}, using default")
            gravity_unit = [0.0, -1.0, 0.0]
        else:
            gravity_unit = (gravity_avg / gravity_norm).tolist()
        print(f"  {sn} gravity: [{gravity_unit[0]:+.3f}, {gravity_unit[1]:+.3f}, {gravity_unit[2]:+.3f}]")

        intrinsics_data[sn]["gravity"] = gravity_unit
        with open(output_dir / f"{sn}.json", "w") as f:
            json.dump(intrinsics_data[sn], f, indent=2)

    print("\nRecording... (Ctrl+C to stop)")
    frame_count = 0

    try:
        while True:
            color_images = {}
            for sn, pipe in pipelines.items():
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_images[sn] = np.asanyarray(color_frame.get_data())

            if len(color_images) == len(pipelines):
                for sn, img in color_images.items():
                    writers[sn].write(img)
                frame_count += 1
                if frame_count % args.fps == 0:
                    print(f"\r{frame_count} frames ({frame_count / args.fps:.1f}s)", end="", flush=True)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        for pipe in pipelines.values():
            pipe.stop()
        for writer in writers.values():
            writer.release()
        print(f"\nSaved {frame_count} frames ({frame_count / args.fps:.1f}s) to {output_dir}")


if __name__ == "__main__":
    main()
