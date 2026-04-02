# Run Guide

This file summarizes the current single-camera online and offline workflows for this repo.

## Environment

All commands below assume:

```bash
cd /home/techshare/liu_projects/Fast-SAM-3D-Body
source /home/techshare/anaconda3/etc/profile.d/conda.sh
conda activate fast_sam_3d_body
```

## Paths Used Below

These examples use the paths that already exist on this machine:

```bash
SMPL_MODEL=/home/techshare/liu_projects/unitree_projects/GENMO/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
NN_MODEL_DIR=/home/techshare/liu_projects/Fast-SAM-3D-Body/mhr2smpl/experiments/multiview_n30000_e500
MAPPING_PATH=/home/techshare/liu_projects/Fast-SAM-3D-Body/mhr2smpl/data/mhr2smpl_mapping.npz
MHR_MESH_PATH=/home/techshare/liu_projects/Fast-SAM-3D-Body/checkpoints/mhr_face_mask.ply
SMOOTHER_DIR=/home/techshare/liu_projects/Fast-SAM-3D-Body/mhr2smpl/experiments/smoother_w5
```

## Online: RealSense D455 -> SONIC

Live single-camera publishing:

```bash
USE_COMPILE=0 python /home/techshare/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source camera \
    --smpl-model-path "$SMPL_MODEL" \
    --nn-model-dir "$NN_MODEL_DIR" \
    --mhr2smpl-mapping-path "$MAPPING_PATH" \
    --mhr-mesh-path "$MHR_MESH_PATH" \
    --smoother-dir "$SMOOTHER_DIR" \
    --zmq-protocol-version 3 \
    --imu-level-init-frames 20 \
    --rerun \
    --rerun-log-stride 2 \
    --rerun-mesh-overlay-stride 1 \
    --addr tcp://*:5556
```

If you do not want live Rerun:

```bash
USE_COMPILE=0 python /home/techshare/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source camera \
    --smpl-model-path "$SMPL_MODEL" \
    --nn-model-dir "$NN_MODEL_DIR" \
    --mhr2smpl-mapping-path "$MAPPING_PATH" \
    --mhr-mesh-path "$MHR_MESH_PATH" \
    --smoother-dir "$SMOOTHER_DIR" \
    --zmq-protocol-version 3 \
    --imu-level-init-frames 20 \
    --addr tcp://*:5556
```

## Offline Workflow

### Step 1: Record RealSense Video

This saves RGB video plus camera intrinsics / gravity metadata.

```bash
python /home/techshare/liu_projects/Fast-SAM-3D-Body/record_realsense.py \
    --output-dir /home/techshare/liu_projects/Fast-SAM-3D-Body/output/records/test_session \
    --width 848 \
    --height 480 \
    --fps 30 \
    --imu-samples 100
```

Outputs:

```text
/home/techshare/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.mp4
/home/techshare/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.json
```

### Step 2: Offline Predict All Frames And Save Full Rerun Record

This runs the full single-view pipeline on every recorded frame.

It saves:

- `test_session_predictions.npz`
- `test_session_predictions_summary.json`
- `test_session_replay.rrd`

The `test_session_replay.rrd` contains:

- raw video
- 2D overlay
- mesh overlay
- 3D body
- SONIC publish state
- published `body_quat`
- published `smpl_joints`
- published `smpl_pose`
- default `joint_pos` / `joint_vel`

```bash
python /home/techshare/liu_projects/Fast-SAM-3D-Body/offline_predict_video.py \
    --video /home/techshare/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.mp4 \
    --intrinsics /home/techshare/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.json \
    --output /home/techshare/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_predictions.npz \
    --use-compile 0 \
    --hand-box-source yolo_pose \
    --smpl-model-path "$SMPL_MODEL" \
    --nn-model-dir "$NN_MODEL_DIR" \
    --mhr2smpl-mapping-path "$MAPPING_PATH" \
    --mhr-mesh-path "$MHR_MESH_PATH" \
    --smoother-dir "$SMOOTHER_DIR" \
    --imu-level-init-frames 20 \
    --rerun \
    --rerun-log-stride 1 \
    --rerun-mesh-overlay-stride 1 \
    --rrd-output /home/techshare/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_replay.rrd
```

### Step 3: Publish Offline Prediction To SONIC

This replays the saved prediction result and publishes to SONIC.

This mode now follows the original video timing automatically. You do not need to tune `--publish-hz`.

Recommended command: no Rerun here, because Step 2 already saved the complete `.rrd`.

```bash
USE_COMPILE=0 python /home/techshare/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source prediction \
    --prediction-file /home/techshare/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_predictions.npz \
    --zmq-protocol-version 3 \
    --addr tcp://*:5556
```

### Optional: Offline Prediction Replay With Rerun

If you still want the prediction replay stage itself to also save another `.rrd`, use:

```bash
USE_COMPILE=0 python /home/techshare/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source prediction \
    --prediction-file /home/techshare/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_predictions.npz \
    --smpl-model-path "$SMPL_MODEL" \
    --zmq-protocol-version 3 \
    --rerun \
    --no-rerun-spawn \
    --rerun-log-stride 1 \
    --rerun-mesh-overlay-stride 1 \
    --rrd-output /home/techshare/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_replay_publish.rrd \
    --addr tcp://*:5556
```

Note:

- This mode is heavier than Step 3 without Rerun.
- If your priority is stable real-time publishing to SONIC, use Step 3 without Rerun.

## SONIC Receiver Side

After starting either the online publisher or the offline prediction publisher in this repo, you can run the SONIC side with the following commands.

### Run SONIC Sim Loop

From:

```text
/home/techshare/liu_projects/unitree_projects/GR00T-WholeBodyControl
```

run:

```bash
python gear_sonic/scripts/run_sim_loop.py
```

Example shell prompt:

```bash
(gear_sonic_sim) (base) techshare@techshare:~/liu_projects/unitree_projects/GR00T-WholeBodyControl$ python gear_sonic/scripts/run_sim_loop.py
```

### Run SONIC Deploy With ZMQ Input

From:

```text
/home/techshare/liu_projects/unitree_projects/GR00T-WholeBodyControl/gear_sonic_deploy
```

run:

```bash
bash deploy.sh \
  --input-type zmq \
  --obs-config /home/techshare/liu_projects/Fast-SAM-3D-Body/configs/sonic/observation_config_smpl_anchor_only.yaml \
  sim
```

Example shell prompt:

```bash
techshare@techshare:~/liu_projects/unitree_projects/GR00T-WholeBodyControl/gear_sonic_deploy$ bash deploy.sh \
  --input-type zmq \
  --obs-config /home/techshare/liu_projects/Fast-SAM-3D-Body/configs/sonic/observation_config_smpl_anchor_only.yaml \
  sim
```

## Result Files Summary

### Recording Stage

```text
output/records/test_session/recording.mp4
output/records/test_session/recording.json
```

### Offline Prediction Stage

```text
output/offline_predictions/test_session_predictions.npz
output/offline_predictions/test_session_predictions_summary.json
output/offline_predictions/test_session_replay.rrd
```

### Optional Replay-Publish Stage

```text
output/offline_predictions/test_session_replay_publish.rrd
```

## Recommended Daily Usage

For most work, use this sequence:

1. `record_realsense.py`
2. `offline_predict_video.py --rerun --rrd-output .../test_session_replay.rrd`
3. `run_publisher.py --source prediction` without `--rerun`

This gives you:

- one complete offline `.rrd` saved during prediction
- one stable SONIC publishing pass at original video timing
