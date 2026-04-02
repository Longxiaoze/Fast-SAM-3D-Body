from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


PROTOCOL_UP_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


class InitialUprightLeveler:
    """Estimate and remove startup pitch/roll bias in a gravity-aligned frame.

    The RealSense IMU is already used to align the published world frame with
    gravity. This helper uses the first few valid body quaternions to estimate
    any residual torso tilt and removes only that tilt, while leaving yaw
    untouched.
    """

    def __init__(self, calibration_frames: int = 15):
        self.calibration_frames = max(0, int(calibration_frames))
        self.up_axis = PROTOCOL_UP_AXIS.copy()
        self._up_samples: list[np.ndarray] = []
        self._correction: Rotation | None = None
        self._estimated_tilt_deg = 0.0

    @property
    def enabled(self) -> bool:
        return self.calibration_frames > 0

    @property
    def ready(self) -> bool:
        return (not self.enabled) or (self._correction is not None)

    @property
    def num_collected(self) -> int:
        return len(self._up_samples)

    @property
    def estimated_tilt_deg(self) -> float:
        return float(self._estimated_tilt_deg)

    def update(self, body_quat_wxyz: np.ndarray) -> bool:
        if self.ready:
            return True

        rot = Rotation.from_quat(_quat_wxyz_to_xyzw(body_quat_wxyz))
        up_vec = rot.apply(self.up_axis)
        up_norm = np.linalg.norm(up_vec)
        if up_norm < 1e-8:
            return False

        self._up_samples.append(up_vec / up_norm)

        if len(self._up_samples) >= self.calibration_frames:
            self._finalize()

        return self.ready

    def _finalize(self) -> None:
        avg_up = np.mean(np.stack(self._up_samples, axis=0), axis=0)
        avg_up_norm = np.linalg.norm(avg_up)
        if avg_up_norm < 1e-8:
            self._correction = Rotation.identity()
            self._estimated_tilt_deg = 0.0
            return

        avg_up /= avg_up_norm
        self._correction, _ = Rotation.align_vectors(
            self.up_axis[None, :], avg_up[None, :]
        )
        cos_tilt = float(np.clip(np.dot(avg_up, self.up_axis), -1.0, 1.0))
        self._estimated_tilt_deg = float(np.degrees(np.arccos(cos_tilt)))

    def apply(self, body_quat_wxyz: np.ndarray) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("InitialUprightLeveler is not ready yet")
        if self._correction is None:
            return np.asarray(body_quat_wxyz, dtype=np.float64).reshape(4)

        corrected = self._correction * Rotation.from_quat(
            _quat_wxyz_to_xyzw(body_quat_wxyz)
        )
        return _quat_xyzw_to_wxyz(corrected.as_quat()).astype(np.float64)
