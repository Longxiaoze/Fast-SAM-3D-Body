import json
import re
import socket

import numpy as np
import zmq

ZMQ_HEADER_SIZE = 1280
SONIC_NUM_JOINTS = 29
TCP_ADDR_RE = re.compile(r"^tcp://([^:]+):(\d+)$")


def _collect_local_ipv4_candidates():
    candidates = {"127.0.0.1"}

    for host in (socket.gethostname(), socket.getfqdn(), "localhost"):
        if not host:
            continue
        try:
            _, _, addrs = socket.gethostbyname_ex(host)
        except OSError:
            continue
        candidates.update(addr for addr in addrs if "." in addr)

    for probe_host in ("8.8.8.8", "1.1.1.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe_host, 80))
                candidates.add(sock.getsockname()[0])
            break
        except OSError:
            continue

    return sorted(candidates)


def _format_bind_error_hint(addr, exc):
    if "Cannot assign requested address" not in str(exc):
        return None

    match = TCP_ADDR_RE.match(addr)
    if match is None:
        return None

    host, port = match.groups()
    if host in {"*", "0.0.0.0", "localhost", "127.0.0.1"}:
        return None

    local_ipv4s = _collect_local_ipv4_candidates()
    explicit_bind = None
    for ip in local_ipv4s:
        if not ip.startswith("127."):
            explicit_bind = f"tcp://{ip}:{port}"
            break

    suggested_bind = f"tcp://*:{port}"
    local_ip_text = ", ".join(local_ipv4s) if local_ipv4s else "unavailable"

    message_lines = [
        f"Failed to bind ZMQ PUB socket to {addr}: {exc}",
        "`--addr` is the publisher bind address on the machine running this process, not the remote robot/subscriber address.",
        f"Use `{suggested_bind}` here",
    ]
    if explicit_bind is not None:
        message_lines[-1] += f" (or an explicit local interface like `{explicit_bind}`)"
    message_lines[-1] += "."
    message_lines.append(
        "On the Unitree/SONIC side, set `--zmq-host` to this publisher machine's LAN IP instead."
    )
    message_lines.append(f"Detected local IPv4 candidates: {local_ip_text}")
    return "\n".join(message_lines)


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
        try:
            self.sock.bind(addr)
        except zmq.ZMQError as exc:
            hint = _format_bind_error_hint(addr, exc)
            if hint is not None:
                self.sock.close(0)
                self.ctx.term()
                raise RuntimeError(hint) from exc
            raise

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
