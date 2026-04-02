import json

import numpy as np
import zmq

ZMQ_HEADER_SIZE = 1280
SONIC_NUM_JOINTS = 29


class ZMQPublisher:
    def __init__(
        self,
        addr,
        topic="pose",
        header_size=ZMQ_HEADER_SIZE,
        protocol_version=2,
        joint_count=SONIC_NUM_JOINTS,
        default_joint_pos=None,
        default_joint_vel=None,
    ):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(addr)

        self.topic = topic
        self.topic_bytes = topic.encode("utf-8")
        self.header_size = header_size
        self.frame_idx = 0
        if protocol_version not in (2, 3):
            raise ValueError(
                f"Unsupported protocol_version={protocol_version}; expected 2 or 3"
            )
        self.protocol_version = int(protocol_version)
        self.joint_count = int(joint_count)
        self.default_joint_pos = self._prepare_joint_array(
            default_joint_pos, name="default_joint_pos"
        )
        self.default_joint_vel = self._prepare_joint_array(
            default_joint_vel, name="default_joint_vel"
        )

        fields = [
            {"name": "frame_index", "dtype": "i64", "shape": [1]},
            {"name": "body_quat", "dtype": "f64", "shape": [1, 4]},  # WXYZ format
        ]
        if self.protocol_version == 3:
            fields.extend(
                [
                    {"name": "joint_pos", "dtype": "f64", "shape": [1, self.joint_count]},
                    {"name": "joint_vel", "dtype": "f64", "shape": [1, self.joint_count]},
                ]
            )
        fields.extend(
            [
                {"name": "smpl_joints", "dtype": "f64", "shape": [1, 24, 3]},
                {"name": "smpl_pose", "dtype": "f64", "shape": [1, 21, 3]},
            ]
        )
        self.header_bytes = self._build_header(fields, version=self.protocol_version)

    def _build_header(self, fields, version=2, count=1):
        header = {"v": version, "endian": "le", "count": count, "fields": fields}
        return json.dumps(header, separators=(",", ":")).encode("utf-8").ljust(self.header_size, b"\0")

    def _prepare_joint_array(self, joint_values, *, name):
        if joint_values is None:
            return np.zeros((1, self.joint_count), dtype=np.float64)

        arr = np.asarray(joint_values, dtype=np.float64)
        expected_size = self.joint_count
        if arr.size != expected_size:
            raise ValueError(
                f"{name} must contain exactly {expected_size} values, got shape {arr.shape}"
            )
        return arr.reshape(1, self.joint_count)

    def publish(self, body_quat, smpl_joints, smpl_pose, joint_pos=None, joint_vel=None):
        chunks = [
            self.topic_bytes,
            self.header_bytes,
            np.array([self.frame_idx], dtype=np.int64).tobytes(),
            np.asarray(body_quat, dtype=np.float64).reshape(1, 4).tobytes(),
        ]
        if self.protocol_version == 3:
            joint_pos_arr = (
                self.default_joint_pos
                if joint_pos is None
                else self._prepare_joint_array(joint_pos, name="joint_pos")
            )
            joint_vel_arr = (
                self.default_joint_vel
                if joint_vel is None
                else self._prepare_joint_array(joint_vel, name="joint_vel")
            )
            chunks.extend([joint_pos_arr.tobytes(), joint_vel_arr.tobytes()])
        chunks.extend(
            [
                np.asarray(smpl_joints, dtype=np.float64).reshape(1, 24, 3).tobytes(),
                np.asarray(smpl_pose, dtype=np.float64).reshape(1, 21, 3).tobytes(),
            ]
        )
        data = b"".join(chunks)

        self.sock.send(data)
        self.frame_idx += 1

    def close(self):
        self.sock.close(0)
        self.ctx.term()
