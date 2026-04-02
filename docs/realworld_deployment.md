# Real-World Deployment

**Note:** Pose estimation may occasionally fail and produce jerky motions. Always validate in simulation and confirm with the debug scripts before deploying to real hardware.

## Prerequisites

Download the following assets and place them somewhere accessible:

- **SMPL model** from [SMPLify](https://smplify.is.tue.mpg.de/)
- **MHR conversion assets** from [MHR tools](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion/assets)

This deployment uses [SONIC](https://nvlabs.github.io/GR00T-WholeBodyControl/) for robot control. Please follow the [SONIC installation guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html) to set up the environment.

## Running the Publisher

Start the pose publisher, which reads from a camera (or recorded video), runs Fast SAM 3D Body inference, and broadcasts SMPL pose over ZMQ.

### Single-View (single camera)

```bash
python run_publisher.py \
    --source camera \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 \
    --mhr2smpl-mapping-path path/to/mhr2smpl_mapping.npz \
    --mhr-mesh-path path/to/mhr_face_mask.ply \
    --smoother-dir mhr2smpl/experiments/smoother_w5 \
    --record
```

### Single-View From Recorded Video

`run_publisher.py` also supports offline playback from a recorded mp4 while keeping the same SONIC ZMQ publish format as the live camera path.

For videos recorded by this repo, pass the `record_realsense.py` JSON.
For portrait D455 videos recorded by GENMO, you can pass the session `capture.json` directly.

```bash
python run_publisher.py \
    --source video \
    --video /path/to/session.mp4 \
    --intrinsics /path/to/camera.json \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 \
    --mhr2smpl-mapping-path path/to/mhr2smpl_mapping.npz \
    --smoother-dir mhr2smpl/experiments/smoother_w5 \
    --no-loop
```

To visualize the same offline publisher stream in Rerun while it publishes to SONIC:

```bash
python run_publisher.py \
    --source video \
    --video /path/to/session.mp4 \
    --intrinsics /path/to/camera.json \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 \
    --mhr2smpl-mapping-path path/to/mhr2smpl_mapping.npz \
    --smoother-dir mhr2smpl/experiments/smoother_w5 \
    --rerun \
    --rerun-log-stride 2 \
    --rerun-max-image-side 720 \
    --rerun-mesh-overlay-stride 4 \
    --rrd-output output/publisher_video.rrd \
    --no-loop
```

For accurate SMPL-to-image alignment in Rerun, make sure the intrinsics file contains the real
`fx`, `fy`, `cx`, and `cy` for the video stream. `run_publisher.py` will now use those exact
intrinsics for the mesh overlay instead of a guessed square focal length.

If runtime is still too slow on your machine, the fastest option is to disable the expensive 2D
mesh overlay panel:

```bash
python run_publisher.py \
    ... \
    --rerun \
    --no-rerun-mesh-overlay
```

Install Rerun once with:

```bash
pip install rerun-sdk==0.19.1
```

### Multi-View (multiple RealSense cameras)

```bash
python run_multiview_publisher.py \
    --source camera \
    --serials <serial_0>,<serial_1> \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 \
    --mhr2smpl-mapping-path path/to/mhr2smpl_mapping.npz \
    --mhr-mesh-path path/to/mhr_face_mask.ply \
    --smoother-dir mhr2smpl/experiments/smoother_w5 \
    --record
```

`--serials` is required. Pass RealSense serial numbers comma-separated; the first serial is treated as the main camera.

## Running SONIC

Fast SAM 3D Body can stream SONIC-compatible motion over ZMQ.

- For the official SONIC release observation config, use [Protocol v3](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html#protocol-v3-joint-smpl-combined-encode-mode-2). The release encoder mode `smpl` requires `smpl_joints`, `smpl_pose`, and wrist joint observations.
- [Protocol v2](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html#protocol-v2-smpl-based-encode-mode-2) is only for custom SMPL-only applications. The official SONIC docs explicitly note that v2 is not used by SONIC's built-in pipelines.
- `run_publisher.py` and `run_multiview_publisher.py` now default to `--zmq-protocol-version 3` and publish neutral 29-DoF `joint_pos` / `joint_vel` arrays alongside the SMPL fields, which is enough for the stock SONIC release config to start. Use `--zmq-protocol-version 2` only if your receiver is a custom SMPL-only subscriber.

```bash
# Simulation
bash deploy.sh sim --input-type zmq --obs-config path/to/obs_config.yaml

# Real robot
bash deploy.sh real --input-type zmq --obs-config path/to/obs_config.yaml
```

<details>
<summary><strong>Observation config</strong></summary>

```yaml
observations:

  - name: "token_state"
    enabled: true

  - name: "his_base_angular_velocity_10frame_step1"
    enabled: true

  - name: "his_body_joint_positions_10frame_step1"
    enabled: true

  - name: "his_body_joint_velocities_10frame_step1"
    enabled: true

  - name: "his_last_actions_10frame_step1"
    enabled: true

  - name: "his_gravity_dir_10frame_step1"
    enabled: true

encoder:
  dimension: 64
  use_fp16: false
  encoder_observations:
    - name: "encoder_mode_4"
      enabled: true
    - name: "motion_joint_positions_10frame_step5"
      enabled: true
    - name: "motion_joint_velocities_10frame_step5"
      enabled: true
    - name: "motion_root_z_position_10frame_step5"
      enabled: true
    - name: "motion_root_z_position"
      enabled: true
    - name: "motion_anchor_orientation"
      enabled: true
    - name: "motion_anchor_orientation_10frame_step5"
      enabled: true
    - name: "motion_joint_positions_lowerbody_10frame_step5"
      enabled: true
    - name: "motion_joint_velocities_lowerbody_10frame_step5"
      enabled: true
    - name: "vr_3point_local_target"
      enabled: true
    - name: "vr_3point_local_orn_target"
      enabled: true
    - name: "smpl_joints_10frame_step1"
      enabled: true
    - name: "smpl_anchor_orientation_10frame_step1"
      enabled: true
    - name: "motion_joint_positions_wrists_10frame_step1"
      enabled: true
  encoder_modes:
    - name: "g1"
      mode_id: 0
      required_observations:
        - encoder_mode_4
        - motion_joint_positions_10frame_step5
        - motion_joint_velocities_10frame_step5
        - motion_anchor_orientation_10frame_step5
    - name: "teleop"
      mode_id: 1
      required_observations:
        - encoder_mode_4
        - motion_joint_positions_lowerbody_10frame_step5
        - motion_joint_velocities_lowerbody_10frame_step5
        - vr_3point_local_target
        - vr_3point_local_orn_target
        - motion_anchor_orientation
    - name: "smpl"
      mode_id: 2
      required_observations:
        - encoder_mode_4
        - smpl_joints_10frame_step1
        - smpl_anchor_orientation_10frame_step1
        - motion_joint_positions_wrists_10frame_step1
```

</details>

## Recording and Debugging

To capture video for offline replay, use `record_realsense.py` or `record_realsense_multi.py`.

To verify the pose stream visually, `debug_smpl_stream.py` subscribes to the publisher and renders a front/side-view SMPL mesh video.

```bash
python debug_smpl_stream.py \
    --smpl-model-path path/to/SMPL_NEUTRAL.pkl \
    --num-frames 300 \
    --render-output output/smpl_debug.mp4 \
    --show-joints
```

To inspect an offline RGB recording with Fast SAM 3D Body in Rerun, use `demo_video_rerun.py`:

```bash
python demo_video_rerun.py \
    --video_path /path/to/video.mp4 \
    --detector yolo_pose \
    --detector_model ./checkpoints/yolo/yolo11m-pose.engine \
    --hand_box_source yolo_pose \
    --num-frames 300 \
    --rrd-output output/video_demo.rrd

rerun output/video_demo.rrd
```

`--use-compile 0` is the default for this script so the first debug run reaches the first frame faster.
