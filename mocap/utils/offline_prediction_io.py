from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OFFLINE_PREDICTION_VERSION = 1

STATUS_VALID = 0
STATUS_NO_PERSON = 1
STATUS_MULTI_PERSON = 2
STATUS_MISSING_KEYS = 3
STATUS_UPRIGHT_CALIBRATING = 4
STATUS_ESTIMATOR_ERROR = 5
STATUS_CONVERTER_ERROR = 6

STATUS_LABELS = {
    STATUS_VALID: "valid",
    STATUS_NO_PERSON: "no_person",
    STATUS_MULTI_PERSON: "multi_person",
    STATUS_MISSING_KEYS: "missing_keys",
    STATUS_UPRIGHT_CALIBRATING: "upright_calibrating",
    STATUS_ESTIMATOR_ERROR: "estimator_error",
    STATUS_CONVERTER_ERROR: "converter_error",
}


def status_code_to_name(code: int) -> str:
    return STATUS_LABELS.get(int(code), f"unknown_{int(code)}")


def _as_scalar_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def normalize_prediction_data(data: dict) -> dict:
    required = ("timestamps", "body_quats", "smpl_joints", "smpl_poses")
    missing = [name for name in required if name not in data]
    if missing:
        raise RuntimeError(
            f"Prediction archive is missing required fields: {', '.join(missing)}"
        )

    timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    num_frames = int(timestamps.shape[0])
    if num_frames == 0:
        raise RuntimeError("Prediction archive contains zero frames")

    body_quats = np.asarray(data["body_quats"], dtype=np.float64)
    smpl_joints = np.asarray(data["smpl_joints"], dtype=np.float64)
    smpl_poses = np.asarray(data["smpl_poses"], dtype=np.float64)

    if body_quats.shape != (num_frames, 4):
        raise RuntimeError(
            f"Expected body_quats shape {(num_frames, 4)}, got {body_quats.shape}"
        )
    if smpl_joints.shape != (num_frames, 24, 3):
        raise RuntimeError(
            f"Expected smpl_joints shape {(num_frames, 24, 3)}, got {smpl_joints.shape}"
        )
    if smpl_poses.shape != (num_frames, 21, 3):
        raise RuntimeError(
            f"Expected smpl_poses shape {(num_frames, 21, 3)}, got {smpl_poses.shape}"
        )

    valid_mask = np.asarray(
        data.get("valid_mask", np.ones(num_frames, dtype=bool)), dtype=bool
    ).reshape(-1)
    if valid_mask.shape != (num_frames,):
        raise RuntimeError(
            f"Expected valid_mask shape {(num_frames,)}, got {valid_mask.shape}"
        )

    status_codes = np.asarray(
        data.get("status_codes", np.full(num_frames, STATUS_VALID, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if status_codes.shape != (num_frames,):
        raise RuntimeError(
            f"Expected status_codes shape {(num_frames,)}, got {status_codes.shape}"
        )

    frame_indices = np.asarray(
        data.get("frame_indices", np.arange(num_frames, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    if frame_indices.shape != (num_frames,):
        raise RuntimeError(
            f"Expected frame_indices shape {(num_frames,)}, got {frame_indices.shape}"
        )

    num_persons = np.asarray(
        data.get("num_persons", np.full(num_frames, -1, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if num_persons.shape != (num_frames,):
        raise RuntimeError(
            f"Expected num_persons shape {(num_frames,)}, got {num_persons.shape}"
        )

    diffs = np.diff(timestamps)
    if np.any(diffs < -1e-9):
        raise RuntimeError("Prediction timestamps must be non-decreasing")

    normalized = {
        "prediction_version": np.asarray(
            int(data.get("prediction_version", OFFLINE_PREDICTION_VERSION)),
            dtype=np.int32,
        ),
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "valid_mask": valid_mask,
        "status_codes": status_codes,
        "num_persons": num_persons,
        "body_quats": body_quats,
        "smpl_joints": smpl_joints,
        "smpl_poses": smpl_poses,
    }

    if "fps" in data:
        normalized["fps"] = np.asarray(float(data["fps"]), dtype=np.float64)
    if "frame_width" in data:
        normalized["frame_width"] = np.asarray(int(data["frame_width"]), dtype=np.int32)
    if "frame_height" in data:
        normalized["frame_height"] = np.asarray(
            int(data["frame_height"]), dtype=np.int32
        )
    if "camera_matrix" in data:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float32).reshape(3, 3)
        normalized["camera_matrix"] = camera_matrix
    if "gravity" in data:
        gravity = np.asarray(data["gravity"], dtype=np.float64).reshape(3)
        normalized["gravity"] = gravity
    if "imu_level_init_frames" in data:
        normalized["imu_level_init_frames"] = np.asarray(
            int(data["imu_level_init_frames"]), dtype=np.int32
        )
    if "camera_body_quats_xyzw" in data:
        camera_body_quats_xyzw = np.asarray(
            data["camera_body_quats_xyzw"], dtype=np.float64
        )
        if camera_body_quats_xyzw.shape != (num_frames, 4):
            raise RuntimeError(
                "Expected camera_body_quats_xyzw shape "
                f"{(num_frames, 4)}, got {camera_body_quats_xyzw.shape}"
            )
        normalized["camera_body_quats_xyzw"] = camera_body_quats_xyzw
    if "pred_cam_ts" in data:
        pred_cam_ts = np.asarray(data["pred_cam_ts"], dtype=np.float64)
        if pred_cam_ts.shape != (num_frames, 3):
            raise RuntimeError(
                f"Expected pred_cam_ts shape {(num_frames, 3)}, got {pred_cam_ts.shape}"
            )
        normalized["pred_cam_ts"] = pred_cam_ts
    if "source_video" in data:
        normalized["source_video"] = np.asarray(_as_scalar_str(data["source_video"]))
    if "source_intrinsics" in data:
        normalized["source_intrinsics"] = np.asarray(
            _as_scalar_str(data["source_intrinsics"])
        )
    if "smpl_model_path" in data:
        normalized["smpl_model_path"] = np.asarray(
            _as_scalar_str(data["smpl_model_path"])
        )

    return normalized


def save_prediction_archive(path: str | Path, data: dict) -> Path:
    archive_path = Path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_prediction_data(data)
    np.savez_compressed(archive_path, **normalized)
    return archive_path


def load_prediction_archive(path: str | Path) -> dict:
    archive_path = Path(path)
    if not archive_path.exists():
        raise RuntimeError(f"Prediction archive not found: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    normalized = normalize_prediction_data(raw)
    normalized["archive_path"] = str(archive_path)
    return normalized


def build_prediction_summary(data: dict) -> dict:
    timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    valid_mask = np.asarray(data["valid_mask"], dtype=bool).reshape(-1)
    status_codes = np.asarray(data["status_codes"], dtype=np.int32).reshape(-1)

    by_status = {}
    for code, count in zip(*np.unique(status_codes, return_counts=True)):
        by_status[status_code_to_name(int(code))] = int(count)

    duration_s = 0.0
    if timestamps.size > 1:
        duration_s = float(timestamps[-1] - timestamps[0])

    summary = {
        "prediction_version": int(np.asarray(data["prediction_version"]).item()),
        "num_frames": int(timestamps.size),
        "num_valid_frames": int(valid_mask.sum()),
        "num_invalid_frames": int((~valid_mask).sum()),
        "duration_s": duration_s,
        "status_counts": by_status,
    }

    if "fps" in data:
        summary["fps"] = float(np.asarray(data["fps"]).item())
    if "frame_width" in data and "frame_height" in data:
        summary["frame_size"] = [
            int(np.asarray(data["frame_width"]).item()),
            int(np.asarray(data["frame_height"]).item()),
        ]
    if "source_video" in data:
        summary["source_video"] = _as_scalar_str(data["source_video"])
    if "source_intrinsics" in data:
        summary["source_intrinsics"] = _as_scalar_str(data["source_intrinsics"])
    if "smpl_model_path" in data:
        summary["smpl_model_path"] = _as_scalar_str(data["smpl_model_path"])

    return summary


def save_prediction_summary(path: str | Path, data: dict) -> Path:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_prediction_summary(data)
    summary["status_labels"] = STATUS_LABELS
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path
