import cv2
import numpy as np
from uuid import uuid4

from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

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


def _build_skeleton_links():
    keypoint_name_to_id = {
        info["name"]: idx for idx, info in mhr70_pose_info["keypoint_info"].items()
    }
    links = []
    for _, info in mhr70_pose_info["skeleton_info"].items():
        start_name, end_name = info["link"]
        if start_name in keypoint_name_to_id and end_name in keypoint_name_to_id:
            links.append((keypoint_name_to_id[start_name], keypoint_name_to_id[end_name]))
    return links


SKELETON_LINKS = _build_skeleton_links()


def create_skeleton_visualizer():
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


class RerunVisualizer:
    def __init__(
        self,
        *,
        width,
        height,
        session_name,
        cam_intrinsics=None,
        spawn=True,
        connect=None,
        rrd_output="",
        max_image_side=720,
        mesh_overlay_stride=1,
        enable_mesh_overlay=True,
    ):
        try:
            import rerun as rr  # type: ignore
            import rerun.blueprint as rrb  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "rerun is not installed in the current environment. "
                "Install it with `pip install rerun-sdk==0.19.1`."
            ) from e

        self.rr = rr
        self.width = int(width)
        self.height = int(height)
        self.cam_intrinsics = (
            np.asarray(cam_intrinsics, dtype=np.float32).reshape(3, 3)
            if cam_intrinsics is not None
            else None
        )
        self.visualizer = create_skeleton_visualizer()
        self.data_recordings = []
        self.max_image_side = int(max_image_side) if max_image_side is not None else 0
        self.mesh_overlay_stride = max(1, int(mesh_overlay_stride))
        self.enable_mesh_overlay = bool(enable_mesh_overlay)
        self._logged_frame_count = 0

        live_recording = None
        if spawn or connect is not None:
            live_recording = rr.new_recording(
                session_name,
                recording_id=uuid4(),
                make_default=False,
                make_thread_default=False,
                spawn=False,
            )
            if connect is not None:
                rr.connect(addr=connect, recording=live_recording)
            elif spawn:
                rr.spawn(recording=live_recording, connect=True)
            self.data_recordings.append(live_recording)

        if rrd_output:
            file_recording = rr.new_recording(
                session_name,
                recording_id=uuid4(),
                make_default=False,
                make_thread_default=False,
                spawn=False,
            )
            rr.save(rrd_output, recording=file_recording)
            self.data_recordings.append(file_recording)

        if not self.data_recordings:
            buffered_recording = rr.new_recording(
                session_name,
                recording_id=uuid4(),
                make_default=False,
                make_thread_default=False,
                spawn=False,
            )
            self.data_recordings.append(buffered_recording)

        self._log_all("world", rr.ViewCoordinates.RDF, static=True)
        self._log_all("world/camera", rr.ViewCoordinates.RDF, static=True)
        self._log_all(
            "world/camera",
            rr.Transform3D(translation=[0.0, 0.0, 0.0]),
            static=True,
        )
        self._log_all("world/body", rr.ViewCoordinates.RDF, static=True)

        portrait_layout = bool(self.height > self.width)
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

        self._send_blueprint_all(rrb.Blueprint(layout, collapse_panels=True))

    def close(self):
        for recording in self.data_recordings:
            try:
                self.rr.disconnect(recording=recording)
            except Exception:
                pass
        self.data_recordings = []

    def _log_all(self, entity_path, *entities, **kwargs):
        for recording in self.data_recordings:
            self.rr.log(entity_path, *entities, recording=recording, **kwargs)

    def _set_time_sequence_all(self, timeline, sequence):
        for recording in self.data_recordings:
            self.rr.set_time_sequence(timeline, sequence, recording=recording)

    def _send_blueprint_all(self, blueprint):
        for recording in self.data_recordings:
            self.rr.send_blueprint(blueprint, recording=recording)

    def _compute_image_scale(self, width, height):
        if self.max_image_side <= 0:
            return 1.0
        max_side = max(int(width), int(height))
        if max_side <= self.max_image_side:
            return 1.0
        return float(self.max_image_side) / float(max_side)

    def _resize_image(self, image, scale):
        if abs(scale - 1.0) < 1e-6:
            return image
        out_w = max(1, int(round(image.shape[1] * scale)))
        out_h = max(1, int(round(image.shape[0] * scale)))
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def _scaled_camera_matrix(self, outputs, width, height, scale):
        if self.cam_intrinsics is not None:
            camera_matrix = self.cam_intrinsics.copy()
        else:
            focal_length = float(max(width, height))
            valid_focals = [
                float(person["focal_length"])
                for person in outputs
                if person.get("focal_length") is not None
                and not np.isnan(person["focal_length"])
            ]
            if valid_focals:
                focal_length = float(np.median(valid_focals))
            camera_matrix = np.array(
                [
                    [focal_length, 0.0, width / 2.0],
                    [0.0, focal_length, height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        if abs(scale - 1.0) >= 1e-6:
            camera_matrix = camera_matrix.copy()
            camera_matrix[0, 0] *= scale
            camera_matrix[1, 1] *= scale
            camera_matrix[0, 2] *= scale
            camera_matrix[1, 2] *= scale
        return camera_matrix

    def _project_keypoints_2d(self, person, camera_matrix):
        keypoints_3d = person.get("pred_keypoints_3d")
        cam_t = person.get("pred_cam_t")
        if keypoints_3d is None or cam_t is None:
            return None

        keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)
        cam_t = np.asarray(cam_t, dtype=np.float32)
        if len(keypoints_3d) == 0 or np.any(np.isnan(keypoints_3d)) or np.any(np.isnan(cam_t)):
            return None

        keypoints_cam = keypoints_3d + cam_t[None, :]
        valid_depth = keypoints_cam[:, 2] > 1e-6
        if not np.any(valid_depth):
            return None

        keypoints_2d = np.zeros((keypoints_cam.shape[0], 2), dtype=np.float32)
        keypoints_2d[valid_depth] = keypoints_cam[valid_depth, :2] / keypoints_cam[valid_depth, 2:3]
        keypoints_2d[:, 0] = keypoints_2d[:, 0] * camera_matrix[0, 0] + camera_matrix[0, 2]
        keypoints_2d[:, 1] = keypoints_2d[:, 1] * camera_matrix[1, 1] + camera_matrix[1, 2]
        return keypoints_2d

    def log_frame(self, *, frame_idx, frame_bgr, frame_rgb, outputs, faces):
        scale = self._compute_image_scale(frame_bgr.shape[1], frame_bgr.shape[0])
        frame_bgr_log = self._resize_image(frame_bgr, scale)
        frame_rgb_log = self._resize_image(frame_rgb, scale)
        camera_matrix = self._scaled_camera_matrix(
            outputs,
            frame_bgr.shape[1],
            frame_bgr.shape[0],
            scale,
        )

        overlay_bgr = self._draw_2d_overlay_image(
            frame_bgr_log,
            outputs,
            camera_matrix=camera_matrix,
            scale=scale,
        )

        if self.enable_mesh_overlay and (
            self._logged_frame_count % self.mesh_overlay_stride == 0
        ):
            mesh_overlay_bgr = self._render_mesh_overlay_image(
                frame_bgr_log, outputs, faces, camera_matrix
            )
        else:
            mesh_overlay_bgr = frame_bgr_log.copy()

        self._log_rerun_frame(
            frame_idx=frame_idx,
            frame_rgb=frame_rgb_log,
            overlay_bgr=overlay_bgr,
            mesh_overlay_bgr=mesh_overlay_bgr,
            outputs=outputs,
            faces=faces,
            width=frame_bgr_log.shape[1],
            height=frame_bgr_log.shape[0],
            camera_matrix=camera_matrix,
        )
        self._logged_frame_count += 1

    def _draw_2d_overlay_image(self, frame_bgr, outputs, *, camera_matrix, scale=1.0):
        overlay = frame_bgr.copy()
        for person_id, person in enumerate(outputs):
            keypoints_2d = self._project_keypoints_2d(person, camera_matrix)
            if keypoints_2d is None:
                keypoints_2d = person["pred_keypoints_2d"]
                if keypoints_2d is not None and len(keypoints_2d) > 0:
                    keypoints_2d = keypoints_2d.astype(np.float32, copy=True)
                    keypoints_2d *= scale
            if keypoints_2d is not None and len(keypoints_2d) > 0:
                keypoints_2d_vis = np.concatenate(
                    [
                        keypoints_2d,
                        np.ones((keypoints_2d.shape[0], 1), dtype=np.float32),
                    ],
                    axis=-1,
                )
                overlay = self.visualizer.draw_skeleton(overlay, keypoints_2d_vis)

            bbox = person.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = [int(v * scale) for v in bbox]
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
                x1, y1, x2, y2 = [int(v * scale) for v in hand_bbox]
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

    def _render_mesh_overlay_image(self, frame_bgr, outputs, faces, camera_matrix):
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

            renderer = Renderer(
                focal_length=(camera_matrix[0, 0], camera_matrix[1, 1]),
                faces=faces,
            )
            mesh_overlay = (
                renderer(
                    vertices,
                    cam_t,
                    mesh_overlay,
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    camera_center=(camera_matrix[0, 2], camera_matrix[1, 2]),
                )
                * 255
            ).astype(np.uint8)

        return mesh_overlay

    def _combine_meshes(self, outputs, faces):
        vertices_all = []
        faces_all = []
        vertex_offset = 0

        for person in outputs:
            vertices = person.get("pred_vertices")
            if vertices is None:
                continue
            if len(vertices) == 0 or np.any(np.isnan(vertices)):
                continue

            vertices_body = vertices.astype(np.float32, copy=False)
            vertices_all.append(vertices_body)
            faces_all.append(faces.astype(np.int32, copy=False) + vertex_offset)
            vertex_offset += vertices_body.shape[0]

        if not vertices_all:
            return None, None

        return np.concatenate(vertices_all, axis=0), np.concatenate(faces_all, axis=0)

    def _combine_joints_and_skeleton(self, outputs):
        points_all = []
        point_colors_all = []
        strips_all = []
        strip_colors_all = []

        for person_id, person in enumerate(outputs):
            joints = person.get("pred_keypoints_3d")
            if joints is None:
                continue
            if len(joints) == 0 or np.any(np.isnan(joints)):
                continue

            joints_world = joints.astype(np.float32, copy=False)
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

    def _log_rerun_frame(
        self,
        *,
        frame_idx,
        frame_rgb,
        overlay_bgr,
        mesh_overlay_bgr,
        outputs,
        faces,
        width,
        height,
        camera_matrix,
    ):
        rr = self.rr

        self._set_time_sequence_all("frame", int(frame_idx))
        self._log_all(
            "world/camera",
            rr.Pinhole(image_from_camera=camera_matrix, resolution=[int(width), int(height)]),
        )

        self._log_all("world/camera/image_raw", rr.Image(frame_rgb))
        self._log_all("world/camera/image_2d_overlay", rr.Image(overlay_bgr[..., ::-1]))
        self._log_all("world/camera/image_mesh_overlay", rr.Image(mesh_overlay_bgr[..., ::-1]))

        vertices, triangle_indices = self._combine_meshes(outputs, faces)
        mesh_extent = None
        if vertices is None:
            self._log_all(
                "world/body/mesh",
                rr.Mesh3D(
                    vertex_positions=np.zeros((0, 3), dtype=np.float32),
                    triangle_indices=np.zeros((0, 3), dtype=np.int32),
                ),
            )
        else:
            mesh_extent = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
            self._log_all(
                "world/body/mesh",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=triangle_indices,
                    albedo_factor=[200, 200, 200, 255],
                ),
            )

        joints, joint_colors, strips3d, strip_colors = self._combine_joints_and_skeleton(
            outputs
        )
        point_radius = 0.006
        line_radius = 0.002
        if mesh_extent is not None and mesh_extent > 1e-6:
            point_radius = max(0.002, 0.004 * mesh_extent)
            line_radius = max(0.001, 0.0015 * mesh_extent)
        if joints is None:
            self._log_all(
                "world/body/joints3d",
                rr.Points3D(np.zeros((0, 3), dtype=np.float32)),
            )
            self._log_all("world/body/joints3d_skeleton", rr.LineStrips3D([]))
        else:
            self._log_all(
                "world/body/joints3d",
                rr.Points3D(
                    joints,
                    colors=joint_colors,
                    radii=np.full(len(joints), point_radius, dtype=np.float32),
                ),
            )
            self._log_all(
                "world/body/joints3d_skeleton",
                rr.LineStrips3D(
                    strips3d,
                    colors=strip_colors,
                    radii=np.full(len(strips3d), line_radius, dtype=np.float32),
                ),
            )
