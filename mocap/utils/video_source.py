import cv2
from abc import ABC, abstractmethod
import json
import time
from pathlib import Path

import numpy as np


def _rotate_intrinsics_90deg(camera_matrix, width, height, angle_deg_ccw):
    angle_norm = int(round(float(angle_deg_ccw))) % 360
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    if angle_norm == 0:
        out_w, out_h = int(width), int(height)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    elif angle_norm == 90:
        out_w, out_h = int(height), int(width)
        K = np.array(
            [[fy, 0.0, cy], [0.0, fx, width - 1.0 - cx], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    elif angle_norm == 180:
        out_w, out_h = int(width), int(height)
        K = np.array(
            [[fx, 0.0, width - 1.0 - cx], [0.0, fy, height - 1.0 - cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    elif angle_norm == 270:
        out_w, out_h = int(height), int(width)
        K = np.array(
            [[fy, 0.0, height - 1.0 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"Unsupported snap90 rotation angle: {angle_deg_ccw}. Expected multiples of 90 degrees."
        )

    return K, out_w, out_h


def _load_intrinsics_metadata(intrinsics_path):
    intrinsics_path = Path(intrinsics_path)
    if not intrinsics_path.exists():
        raise RuntimeError(f"Intrinsics JSON not found: {intrinsics_path}")

    with open(intrinsics_path, "r") as f:
        intrinsics_data = json.load(f)

    if "camera_matrix" in intrinsics_data and "gravity" in intrinsics_data:
        cam_matrix = np.array(intrinsics_data["camera_matrix"], dtype=np.float32)
        gravity = np.array(intrinsics_data["gravity"], dtype=np.float64)
        width = intrinsics_data.get("width")
        height = intrinsics_data.get("height")
        source_format = "camera_json"
        return cam_matrix, gravity, width, height, source_format

    if "color_intrinsics" in intrinsics_data and "imu_gravity_color" in intrinsics_data:
        color_intrinsics = intrinsics_data["color_intrinsics"]
        base_width = int(color_intrinsics["width"])
        base_height = int(color_intrinsics["height"])
        base_camera_matrix = np.array(
            [
                [float(color_intrinsics["fx"]), 0.0, float(color_intrinsics["ppx"])],
                [0.0, float(color_intrinsics["fy"]), float(color_intrinsics["ppy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        rotation_meta = intrinsics_data.get("image_rotation_locked")
        if rotation_meta is not None:
            mode = rotation_meta.get("mode", "snap90")
            if mode != "snap90":
                raise RuntimeError(
                    "Only snap90 capture.json rotation is supported for offline video playback."
                )
            cam_matrix, width, height = _rotate_intrinsics_90deg(
                base_camera_matrix,
                base_width,
                base_height,
                rotation_meta.get("angle_deg_ccw", 0.0),
            )
        else:
            cam_matrix = base_camera_matrix
            width = int(intrinsics_data.get("output_width", base_width))
            height = int(intrinsics_data.get("output_height", base_height))

        gravity = np.array(intrinsics_data["imu_gravity_color"], dtype=np.float64)
        source_format = "genmo_capture_json"
        return cam_matrix, gravity, width, height, source_format

    raise RuntimeError(
        f"Unsupported intrinsics file format: {intrinsics_path}. "
        "Expected either record_realsense camera.json or GENMO capture.json."
    )


class VideoSource(ABC):
    @abstractmethod
    def get_frame(self):
        """Return (frame_bgr, timestamp) or (None, None) if end"""
        ...

    @abstractmethod
    def release(self): ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @abstractmethod
    def get_camera_intrinsics(self) -> np.ndarray: ...

    @abstractmethod
    def get_frame_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_gravity_direction(self) -> np.ndarray: ...


class RealSenseSource(VideoSource):

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 15,
        imu_samples: int = 100,
        startup_timeout_s: float = 60.0,
        frame_timeout_ms: int = 1000,
    ):
        import pyrealsense2 as rs

        self.rs = rs
        self.width = int(width)
        self.height = int(height)
        self._fps = float(fps)
        self.imu_samples = int(imu_samples)
        self.startup_timeout_s = float(startup_timeout_s)
        self.frame_timeout_ms = int(frame_timeout_ms)

        if self.imu_samples <= 0:
            raise ValueError("imu_samples must be > 0")
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be > 0")
        if self.frame_timeout_ms <= 0:
            raise ValueError("frame_timeout_ms must be > 0")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, int(self._fps)
        )

        # Enable IMU for gravity calibration
        config.enable_stream(rs.stream.accel)

        try:
            profile = self.pipeline.start(config)

            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            self.cam_intrinsics = np.array(
                [
                    [intrinsics.fx, 0.0, intrinsics.ppx],
                    [0.0, intrinsics.fy, intrinsics.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )[None, ...]

            print("RealSense camera intrinsics:")
            print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
            print(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            print(f"  cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

            self._calibrate_gravity()
        except Exception:
            self._stop_pipeline_quietly()
            raise

    @staticmethod
    def _is_frame_timeout_error(exc: RuntimeError) -> bool:
        return "frame didn't arrive" in str(exc).lower()

    def _stop_pipeline_quietly(self):
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _wait_for_frames_until(self, purpose: str, deadline: float):
        last_error = None
        next_notice = time.monotonic() + 5.0

        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                detail = f" Last RealSense error: {last_error}" if last_error else ""
                raise RuntimeError(
                    f"Timed out waiting for RealSense frames while {purpose}. "
                    f"Requested {self.width}x{self.height}@{int(self._fps)} "
                    f"with accel stream.{detail} "
                    "Check that the camera is connected to a USB3 port, is not open "
                    "in another process, and supports this color/IMU mode."
                ) from last_error

            timeout_ms = max(1, min(self.frame_timeout_ms, int(remaining_s * 1000)))
            try:
                return self.pipeline.wait_for_frames(timeout_ms)
            except RuntimeError as exc:
                if not self._is_frame_timeout_error(exc):
                    raise

                last_error = exc
                now = time.monotonic()
                if now >= next_notice:
                    print(f"  Still waiting for RealSense frames while {purpose}...")
                    next_notice = now + 5.0

    def _calibrate_gravity(self):
        rs = self.rs

        # Calibrate gravity direction (once, assuming camera is static)
        print(
            f"Calibrating gravity direction ({self.imu_samples} IMU samples, keep camera steady)..."
        )
        accel_samples = []
        deadline = time.monotonic() + self.startup_timeout_s
        next_notice = time.monotonic() + 5.0

        while len(accel_samples) < self.imu_samples:
            frames = self._wait_for_frames_until("calibrating IMU gravity", deadline)
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                accel_samples.append([accel_data.x, accel_data.y, accel_data.z])
                continue

            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(
                    "Timed out collecting RealSense IMU samples: "
                    f"{len(accel_samples)}/{self.imu_samples}. "
                    "Check that this camera has an accelerometer and that the accel stream "
                    "is not blocked by another RealSense process."
                )
            if now >= next_notice:
                print(
                    "  Waiting for RealSense accel samples "
                    f"({len(accel_samples)}/{self.imu_samples})..."
                )
                next_notice = now + 5.0

        accel_array = np.array(accel_samples, dtype=np.float64)
        gravity_avg = -np.mean(accel_array, axis=0)
        gravity_norm = np.linalg.norm(gravity_avg)

        if gravity_norm < 1e-6:
            raise RuntimeError("Gravity magnitude is near zero - invalid IMU data")

        self.gravity_direction = gravity_avg / gravity_norm
        print(
            f"  Gravity direction: [{self.gravity_direction[0]:.3f}, {self.gravity_direction[1]:.3f}, {self.gravity_direction[2]:.3f}]"
        )
        print(f"  Magnitude: {gravity_norm:.2f} m/s^2")

    def get_frame(self) -> tuple[np.ndarray, float]:
        deadline = time.monotonic() + max(10.0, self.frame_timeout_ms / 1000.0)

        while True:
            frames = self._wait_for_frames_until("capturing color frame", deadline)
            color_frame = frames.get_color_frame()
            if color_frame:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for RealSense color frames")

        frame = np.asanyarray(color_frame.get_data())
        return frame, time.time()

    def release(self):
        self._stop_pipeline_quietly()
        self.pipeline = None

    @property
    def fps(self) -> float:
        return self._fps

    def get_camera_intrinsics(self) -> np.ndarray:
        return self.cam_intrinsics

    def get_frame_size(self) -> tuple[int, int]:
        return self.width, self.height

    def get_gravity_direction(self) -> np.ndarray:
        return self.gravity_direction


class VideoFileSource(VideoSource):
    def __init__(self, video_path, intrinsics_path, loop=False):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.loop = bool(loop)
        self.video_path = video_path
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_time = 1.0 / self._fps
        self.start_time = None
        self.frame_count = 0

        print(f"Video: {video_path}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self._fps:.1f}")
        print(f"  Total frames: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        cam_matrix, gravity, intr_width, intr_height, source_format = _load_intrinsics_metadata(
            intrinsics_path
        )
        self.cam_intrinsics = cam_matrix[None, ...]
        print(f"  Loaded camera intrinsics from: {intrinsics_path} ({source_format})")
        print(f"    fx={cam_matrix[0, 0]:.2f}, fy={cam_matrix[1, 1]:.2f}")
        print(f"    cx={cam_matrix[0, 2]:.2f}, cy={cam_matrix[1, 2]:.2f}")
        if intr_width is not None and intr_height is not None:
            print(f"    intrinsics image size: {int(intr_width)}x{int(intr_height)}")

        gravity_norm = np.linalg.norm(gravity)
        if gravity_norm < 1e-6:
            raise RuntimeError("Gravity magnitude is near zero - invalid data")

        self.gravity_direction = gravity / gravity_norm
        print(
            f"  Gravity direction: [{self.gravity_direction[0]:.3f}, {self.gravity_direction[1]:.3f}, {self.gravity_direction[2]:.3f}]"
        )

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None, None
                self.start_time = None
                self.frame_count = 0
            else:
                return None, None

        if self.start_time is None:
            self.start_time = time.time()

        timestamp = self.start_time + self.frame_count * self.frame_time
        self.frame_count += 1
        return frame, timestamp

    def release(self):
        self.cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    def get_camera_intrinsics(self):
        return self.cam_intrinsics

    def get_frame_size(self):
        return self.width, self.height

    def get_gravity_direction(self):
        return self.gravity_direction


def create_video_source(source_type, **kwargs):
    if source_type == "camera":
        return RealSenseSource(**kwargs)
    if source_type == "video":
        return VideoFileSource(**kwargs)
    raise ValueError(f"Unknown source type: {source_type}")
