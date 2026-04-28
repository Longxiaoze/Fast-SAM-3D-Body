# Run Guide

This file summarizes the current single-camera online and offline workflows for this repo.

## Environment

All commands below assume:

```bash
cd /home/longxiaoze/liu_projects/Fast-SAM-3D-Body
source /home/longxiaoze/anaconda3/etc/profile.d/conda.sh
conda activate fast_sam_3d_body
```

## Paths Used Below

These examples use the paths that already exist on this machine:

```bash
SMPL_MODEL=/home/longxiaoze/liu_projects/smpl_inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
NN_MODEL_DIR=/home/longxiaoze/liu_projects/Fast-SAM-3D-Body/mhr2smpl/experiments/multiview_n30000_e500
MAPPING_PATH=/home/longxiaoze/liu_projects/Fast-SAM-3D-Body/mhr2smpl/data/mhr2smpl_mapping.npz
MHR_MESH_PATH=/home/longxiaoze/liu_projects/Fast-SAM-3D-Body/checkpoints/mhr_face_mask.ply
SMOOTHER_DIR=/home/longxiaoze/liu_projects/Fast-SAM-3D-Body/mhr2smpl/experiments/smoother_w5
```

## Online: RealSense D455 -> SONIC

Live single-camera publishing:

```bash
conda activate fast_sam_3d_body
cd /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/
USE_COMPILE=0 python run_publisher.py \
    --source camera \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
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
conda activate fast_sam_3d_body
cd /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/
USE_COMPILE=0 python /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source camera \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 15 \
    --smpl-model-path "$SMPL_MODEL" \
    --nn-model-dir "$NN_MODEL_DIR" \
    --mhr2smpl-mapping-path "$MAPPING_PATH" \
    --mhr-mesh-path "$MHR_MESH_PATH" \
    --smoother-dir "$SMOOTHER_DIR" \
    --zmq-protocol-version 3 \
    --imu-level-init-frames 20 \
    --addr tcp://*:5556
```

Important:

- `--addr` is the ZMQ publisher bind address on the machine running `run_publisher.py`.
- If SONIC deploy runs on another machine or the real robot, do not set `--addr` to the robot IP.
- Keep `--addr tcp://*:5556` here, and set `--zmq-host <publisher_machine_ip>` on the SONIC side instead.

## Offline Workflow

### Step 1: Record RealSense Video

This saves RGB video plus camera intrinsics / gravity metadata.

```bash
python /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/record_realsense.py \
    --output-dir /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/records/test_session \
    --width 640 \
    --height 480 \
    --fps 15 \
    --imu-samples 100
```

Outputs:

```text
/home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.mp4
/home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.json
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
python /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/offline_predict_video.py \
    --video /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.mp4 \
    --intrinsics /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/records/test_session/recording.json \
    --output /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_predictions.npz \
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
    --rrd-output /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_replay.rrd
```

### Step 3: Publish Offline Prediction To SONIC

This replays the saved prediction result and publishes to SONIC.

This mode now follows the original video timing automatically. You do not need to tune `--publish-hz`.

Recommended command: no Rerun here, because Step 2 already saved the complete `.rrd`.

```bash
USE_COMPILE=0 python /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source prediction \
    --prediction-file /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_predictions.npz \
    --zmq-protocol-version 3 \
    --addr tcp://*:5556
```

### Optional: Offline Prediction Replay With Rerun

If you still want the prediction replay stage itself to also save another `.rrd`, use:

```bash
USE_COMPILE=0 python /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source prediction \
    --prediction-file /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_predictions.npz \
    --smpl-model-path "$SMPL_MODEL" \
    --zmq-protocol-version 3 \
    --rerun \
    --no-rerun-spawn \
    --rerun-log-stride 1 \
    --rerun-mesh-overlay-stride 1 \
    --rrd-output /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/output/offline_predictions/test_session_replay_publish.rrd \
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
cd /home/longxiaoze/liu_projects/unitree_projects/GR00T-WholeBodyControl
```

run:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Example shell prompt:

```bash
(gear_sonic_sim) (base) longxiaoze@longxiaoze:~/liu_projects/unitree_projects/GR00T-WholeBodyControl$ python gear_sonic/scripts/run_sim_loop.py
```

### Run SONIC Deploy With ZMQ Input

From:

```text
cd /home/longxiaoze/liu_projects/unitree_projects/GR00T-WholeBodyControl/gear_sonic_deploy
```

run:

```bash
bash deploy.sh \
  --input-type zmq \
  --obs-config /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/configs/sonic/observation_config_smpl_anchor_only.yaml \
  sim
```

Example shell prompt:

```bash
longxiaoze@longxiaoze:~/liu_projects/unitree_projects/GR00T-WholeBodyControl/gear_sonic_deploy$ bash deploy.sh \
  --input-type zmq \
  --obs-config /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/configs/sonic/observation_config_smpl_anchor_only.yaml \
  sim
```

### Actual Unitree G1 Robot On Another Machine

If the camera / publisher machine is `192.168.0.155` and the Unitree robot machine is `192.168.0.79`:

- Run the publisher on `192.168.0.155` with `--addr tcp://*:5556`
- Run SONIC deploy on the robot with `--zmq-host 192.168.0.155`

Example robot-side command for the real G1:

```bash
bash deploy.sh \
  --input-type zmq \
  --obs-config ~/observation_config_smpl_anchor_only.yaml \
  --zmq-host 192.168.0.155 \
  --zmq-port 5556 \
  --zmq-topic pose \
  real
```

If the robot machine also has this repo checked out at the same path, you can use:

```bash
bash deploy.sh \
  --input-type zmq \
  --obs-config /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/configs/sonic/observation_config_smpl_anchor_only.yaml \
  --zmq-host 192.168.0.155 \
  --zmq-port 5556 \
  --zmq-topic pose \
  real
```

If you want SONIC sim instead of the real robot, keep the same ZMQ arguments and change only the final mode from `real` to `sim`.

Do not use `--addr tcp://192.168.0.79:5556` on the publisher side unless `192.168.0.79` is actually a local IP assigned to the machine running the publisher process.

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

<details>
<summary><strong>2026-04-03 腿部前后摆动异常排查与修复记录（点击展开）</strong></summary>

## 现象

在线和离线路径都能正常运行，`SMPL` 可视化整体也正常，但下肢有一个非常具体的问题：

- 人体右脚向前、左脚向后时，机器人表现正常
- 但人体右脚向后时，机器人更像是“左脚向前”
- 人体左脚向前时，机器人更像是“右脚向后”

这说明问题不是整个人体姿态完全错乱，而是某类“前后 + 左右”组合在进入 SONIC 前被错误解释了。

## 一开始怀疑的方向

最开始主要怀疑两类问题：

1. 腿部映射表或左右腿 joint order 写错
2. 发布给 SONIC 的 `smpl_joints` 局部坐标系轴顺序不对

但是完整梳理代码后发现，这两个都不是最核心的问题。

## 完整排查过程

### 1. 先确认 `mhr2smpl` 是否真的把左右腿映射错了

重点查看：

- `mocap/core/multiview_mhr2smpl.py`
- `mhr2smpl/multi_view/infer_multiview.py`
- `mhr2smpl/multi_view/step2_train.py`

结论：

- `fusion_runner.infer()` 的输出是 `body_pose + canonical_joints + betas`
- `canonical_joints` 是通过 `SMPL(global_orient=0, body_pose=..., betas=...)` 做 FK 得到的 root-relative joints
- 这里没有“交换左右腿”的逻辑
- 训练时 `FK loss` 也是拿 canonical joints 对 canonical joints，不存在把左腿监督到右腿上的过程

也就是说：

- `mhr2smpl_mapping.npz` 负责的是 MHR mesh 到 SMPL 顶点拓扑的重建映射
- 它不是“腿部左右语义映射表”
- 如果这里错了，通常 `SMPL` 可视化本身也会明显出问题

所以：

- “腿部映射表错了”不是这次问题的主要原因

### 2. 再看发布前的 `smpl_joints_local` 是否被错误变换

重点查看：

- `mocap/utils/pose_protocol.py`

发布时的关键过程是：

1. `canonical_joints` 先根据 `body_quat_xyzw` 旋到相机系
2. 再通过 `R_world_cam` 变到 world
3. 再减根节点得到 root-local joints
4. 再通过 `SMPL_BASE_REMOVE_QUAT_XYZW` 和额外旋转得到最终发给协议的 `body_quat`

这里非常关键的一点是：

- `smpl_joints_local` 和 `body_quat` 是配套使用的
- 如果只改 `smpl_joints_local` 的轴，不改根朝向来源，整个人很容易被扭到奇怪的位置

这也是为什么之前直接改 `JOINTS_COORD_TRANSFORM` 会把身体整体扭坏：

- 那种改法相当于“强行改解释坐标系”
- 但真正的问题并不在局部关节坐标轴本身

所以最终决定：

- 不再修改 `mocap/utils/pose_protocol.py`
- 保持原来的 `smpl_joints` 局部坐标发布方式不动

### 3. 真正的问题出在“姿态”和“根朝向”来自两条不同分支

这是这次排查最重要的结论。

在线/离线原始逻辑里：

- `smpl_pose` 和 `canonical_joints` 来自 `fusion_runner.infer(...)`
- 但 `body_quat_xyzw` 却来自 Stage-1 的 `out["global_rot"]`

也就是：

- 下肢姿态来自融合后的 SMPL body pose
- 根朝向来自另一条独立回归分支

当人体动作是视觉上容易混淆的单目姿态时，比如：

- 右后抬腿
- 左前抬腿

这两类动作在单目视角下可能非常像，尤其当躯干朝向、相机位置、髋部旋转比较接近时更明显。

结果就是：

- `fusion_runner` 预测的腿部姿态可能是对的
- 但 Stage-1 的 `global_rot` 把身体朝向解释成了另一种近似姿态
- SONIC 同时读取 `smpl_joints` 和 `smpl_anchor_orientation`
- 于是策略侧会把“同一组腿部姿态”解释成另一条腿的前后方向

这就能解释为什么：

- `SMPL` 可视化整体看起来没问题
- 但机器人执行出来会出现“右后像左前、左前像右后”

### 4. 为什么仓库自带 demo 反而给了正确提示

在 `mhr2smpl/multi_view/step3_demo_RICH.py` 和 `step3_demo_AIST.py` 里，作者并没有直接把 `global_rot` 当成最终根朝向来用。

他们的做法是：

1. 先得到 `canonical_joints`
2. 再从 Stage-1 的观测 mesh 中恢复一组观测到的 SMPL joints
3. 然后用两组 joints 做刚体对齐，求 `R_cam`

这说明原作者自己的 demo 里，其实已经倾向于：

- 用“几何对齐后的朝向”
- 而不是单独相信 Stage-1 的 `global_rot`

这和我们当前问题完全对上了。

## 最终修复方法

修复目标不是改腿部 joint order，而是让：

- `smpl_pose / canonical_joints`
- `body_quat / anchor orientation`

来自同一套几何结果。

### 修复思路

新增一条根朝向估计路径：

1. 先把 Stage-1 的 `pred_vertices + pred_cam_t` 重新映射到 24 个 SMPL joints
2. 再把 fusion 输出的 `canonical_joints` 与这组观测 joints 做刚体对齐
3. 用 SVD 求最佳旋转矩阵
4. 转成 `body_quat_xyzw`
5. 再交给原来的 `prepare_publish_pose()` 走后续协议转换

这样做的好处是：

- 不改 `smpl_joints_local` 的坐标系定义
- 不改 `smpl_pose`
- 只修正“根朝向来源”
- 姿态与朝向来自同一几何结果，语义一致

### 新增的核心代码

新增位置：

- `mocap/core/multiview_mhr2smpl.py`

新增了两个函数：

```python
def observed_smpl_joints_from_mhr(self, pred_vertices, pred_cam_t):
    verts = np.asarray(pred_vertices, dtype=np.float32)
    cam_t = np.asarray(pred_cam_t, dtype=np.float32).reshape(3)
    mhr_cam = verts + cam_t[None, :]
    face_verts = mhr_cam[self._mhr_vert_ids]
    smpl_verts = (face_verts * self._baryc[:, :, None]).sum(axis=1)
    joints = self._smpl_j_reg @ smpl_verts
    joints -= joints[0:1]
    return joints.astype(np.float32)


def estimate_body_quat_xyzw_from_joint_alignment(
    self,
    canonical_joints,
    pred_vertices,
    pred_cam_t,
    *,
    prev_quat_xyzw=None,
):
    canon = np.asarray(canonical_joints, dtype=np.float64).reshape(24, 3)
    observed = self.observed_smpl_joints_from_mhr(pred_vertices, pred_cam_t).astype(
        np.float64
    )

    cov = canon[1:].T @ observed[1:]
    u, _s, vt = np.linalg.svd(cov)
    rot_mat = vt.T @ u.T
    if np.linalg.det(rot_mat) < 0.0:
        vt[-1, :] *= -1.0
        rot_mat = vt.T @ u.T

    quat_xyzw = Rotation.from_matrix(rot_mat).as_quat().astype(np.float64)
    if prev_quat_xyzw is not None and np.dot(quat_xyzw, prev_quat_xyzw) < 0.0:
        quat_xyzw = -quat_xyzw
    return quat_xyzw
```

这里的重点是：

- 先把观测 mesh 转成观测 joints
- 再和 canonical joints 做刚体配准
- 只取旋转，不改 body pose

## 各运行路径里的对应改动

### 1. 单目实时发布

文件：

- `run_publisher.py`

原逻辑：

```python
body_quat_xyzw = self._compute_body_quat(out["global_rot"])
smpl_pose, canonical_joints, _betas, _weights = self.fusion_runner.infer(
    [(pred_vertices, pred_cam_t)]
)
```

新逻辑：

```python
smpl_pose, canonical_joints, _betas, _weights = self.fusion_runner.infer(
    [(pred_vertices, pred_cam_t)]
)
body_quat_xyzw = self._estimate_body_quat_xyzw(out, canonical_joints)
```

并新增：

```python
--body-orient-source {joint_alignment, stage1_global_rot}
```

默认值：

```text
joint_alignment
```

含义：

- `joint_alignment`：使用新方法，通过 joints 几何对齐估计根朝向
- `stage1_global_rot`：保留旧行为，继续使用 `out["global_rot"]`

### 2. 双目/多视角实时发布

文件：

- `run_multiview_publisher.py`

同样新增：

```python
_estimate_main_body_quat(...)
--body-orient-source {joint_alignment, stage1_global_rot}
```

逻辑与单目一致，只是 main camera 用于提供 Stage-1 观测 joints。

### 3. 离线预测

文件：

- `offline_predict_video.py`

新增：

```python
def estimate_body_quat_xyzw(...):
    ...
```

并把原来的：

```python
body_quat_xyzw = compute_body_quat(out["global_rot"])
```

替换为：

```python
body_quat_xyzw = estimate_body_quat_xyzw(
    fusion_runner,
    out,
    canonical_joints,
    body_orient_source=args.body_orient_source,
    prev_quat_xyzw=prev_body_quat_xyzw,
)
```

同时把本次离线预测使用的根朝向来源写入输出：

```python
"body_orient_source": np.asarray(args.body_orient_source)
```

这样后续回放时也知道当时用的是哪种根朝向来源。

## 为什么这次不再改 `pose_protocol.py`

这次专门保留了：

- `mocap/utils/pose_protocol.py`

不做改动，原因是：

- 之前改 `JOINTS_COORD_TRANSFORM` 会导致整体身体被扭曲
- 说明根问题不是局部关节轴定义本身
- 而是“局部关节 + 根朝向”这对组合的来源不一致

因此本次修复遵循的原则是：

- 不改 `smpl_joints_local` 坐标系
- 不改 `smpl_pose`
- 只改 `body_quat` 的来源

## 当前推荐使用方式

### 单目实时发布

```bash
python run_publisher.py \
  ... \
  --body-orient-source joint_alignment
```

### 双目实时发布

```bash
python run_multiview_publisher.py \
  ... \
  --body-orient-source joint_alignment
```

### 离线预测

```bash
python offline_predict_video.py \
  ... \
  --body-orient-source joint_alignment
```

### 如果需要快速回退旧行为

```bash
--body-orient-source stage1_global_rot
```

## 建议验证动作

建议重点验证以下几种下肢动作：

1. 右脚前、左脚后
2. 右脚后、左脚前
3. 左脚前、右脚后
4. 左脚后、右脚前
5. 原地抬右腿向右后
6. 原地抬左腿向左前

如果修复有效，最明显的变化应该是：

- “右后”和“左前”不再互相混淆
- 身体整体不会再因为改局部坐标轴而被扭曲

## 这次修复的边界

这次修复解决的是：

- 发布给 SONIC 时，根朝向和融合后的 SMPL 姿态不一致的问题

这次修复不直接解决的是：

- 单目视觉本身对某些极端姿态的深度歧义
- 机器人策略在训练分布外动作上的泛化问题

如果后续仍存在少量混淆，但已经比之前明显减轻，那么下一步应该优先排查：

1. SONIC 策略侧对 `smpl_anchor_orientation + smpl_joints` 的使用方式
2. 某些动作在单目视角下是否天然存在前后歧义
3. 是否需要双目/多目作为最终部署路径

</details>

## 2026-04-28 加速更新：`process_one_image` 首帧冷启动和实时路径同步开销

这次更新针对现象：

```text
[process_one_image] TOTAL: 0.2186s
```

当时运行命令里是：

```bash
USE_COMPILE=0 python /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/run_publisher.py ...
```

所以这里的主要问题不是 `torch.compile` 持续编译，而是实时发布路径没有被正确 warmup，以及默认调试检查带来的 GPU 同步开销。

### 本次测试结论

在本机 RTX 5090 上，用 `notebook/images/dancing.jpg` 模拟实时路径测试：

- 未覆盖真实 `yolo_pose` merged 路径 warmup 时，第一帧可能达到约 `2.08s`
- 后续稳定帧约 `0.143s`
- 修复 warmup 后，真实 `yolo_pose` 第一帧约 `0.141s`
- 修复后 4 次真实路径平均约 `0.1407s`

因此 `0.2186s` 更像是冷路径或额外同步开销，不是 `USE_COMPILE=0` 下的编译耗时。

### 1. `run_publisher.py`：warmup 必须覆盖真实在线路径

实时推理循环中实际调用的是：

```python
outputs = self.estimator.process_one_image(
    frame_rgb,
    cam_int=self.cam_intrinsics,
    hand_box_source="yolo_pose",
)
```

旧版 `_warmup()` 使用的是：

```python
self.estimator.process_one_image(
    dummy_img,
    bboxes=warmup_bbox,
    hand_box_source="body_decoder",
)
```

这只预热了 `body_decoder` 顺序路径，没有覆盖在线循环中的 `yolo_pose` hand-box 路径，也没有覆盖 body+hand merged batch 路径。因此第一帧真实相机输入到来时，仍然会触发 YOLO TensorRT context、backbone/decoder kernel、merged batch 等冷启动成本。

新版改为先读取相机内参，再 warmup：

```python
cam_intrinsics_np = self.video_source.get_camera_intrinsics()
if cam_intrinsics_np is not None:
    self.cam_intrinsics = torch.from_numpy(
        np.asarray(cam_intrinsics_np, dtype=np.float32)
    )
else:
    self.cam_intrinsics = None

logger.info("Warming up model...")
self._warmup()
```

这样 warmup 阶段也能使用和正式推理一致的 `cam_int`，避免正式第一帧再切换分支。

新版 `_warmup()` 做两层预热：

```python
warmup_keypoints = np.zeros((1, 17, 3), dtype=np.float32)
warmup_keypoints[0, 9] = [width * 0.35, height * 0.48, 1.0]
warmup_keypoints[0, 10] = [width * 0.65, height * 0.48, 1.0]

if self.estimator.detector is not None:
    dummy_bgr = cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR)
    for _ in range(2):
        _ = self.estimator.detector.run_human_detection(
            dummy_bgr,
            bbox_thr=0.01,
            nms_thr=0.3,
            default_to_full_image=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

这部分专门预热 YOLO-Pose TensorRT engine。第一次运行 TensorRT engine 会创建 execution context，所以不能留到正式第一帧。

随后 warmup 真实 merged 推理路径：

```python
for _ in range(2):
    _ = self.estimator.process_one_image(
        dummy_img,
        bboxes=warmup_bbox,
        cam_int=self.cam_intrinsics,
        hand_box_source="yolo_pose",
        yolo_pose_keypoints=warmup_keypoints,
        yolo_pose_body_boxes=warmup_bbox,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
```

关键点是 `hand_box_source="yolo_pose"` 加上合成的 wrist keypoints。这样即使 warmup 不跑真实 YOLO 检测，也能进入和在线循环相同的：

- YOLO wrist hand-box 分支
- body + left hand + right hand merged batch
- shared backbone
- combined body/hand decoder
- postprocess IK

这比旧的 `body_decoder` warmup 更贴近真实运行。

### 2. `sam_3d_body_estimator.py`：允许传入预计算 YOLO-Pose 结果

为了让 warmup 能跳过真实检测、但仍然走 `yolo_pose` hand-box 分支，`process_one_image()` 新增了两个可选参数：

```python
def process_one_image(
    ...,
    hand_box_source: str = "body_decoder",
    yolo_pose_keypoints: Optional[np.ndarray] = None,
    yolo_pose_body_boxes: Optional[np.ndarray] = None,
):
```

当 warmup 传入 `bboxes` 和 `yolo_pose_keypoints` 时：

```python
if bboxes is not None:
    boxes = bboxes.reshape(-1, 4)
    if yolo_pose_body_boxes is None and yolo_pose_keypoints is not None:
        yolo_pose_body_boxes = boxes.copy()
```

这样 warmup 不需要真实图像里有人，也不需要 YOLO 检出人体，就能构造出完整的 body/hand 路径。

### 3. 关闭每帧 `torch.cuda.empty_cache()` 默认行为

旧逻辑每次 `process_one_image()` 都执行：

```python
torch.cuda.empty_cache()
```

这会释放 PyTorch CUDA caching allocator 里的缓存。它对避免显存碎片可能有帮助，但在实时推理里会破坏缓存复用，并可能引入同步和重新分配开销。

现在改为默认关闭，只在显式需要时打开：

```python
if os.environ.get("SAM3D_EMPTY_CACHE_EACH_FRAME", "0") == "1":
    torch.cuda.empty_cache()
```

正常实时运行不建议打开。如果怀疑显存泄漏或显存碎片，可以临时使用：

```bash
SAM3D_EMPTY_CACHE_EACH_FRAME=1 USE_COMPILE=0 python run_publisher.py ...
```

### 4. 关闭默认 NaN debug 扫描，避免隐藏 GPU 同步

旧代码里有几处 NaN 检查默认开启，例如：

```python
torch.isnan(tensor).any().item()
```

这里的 `.item()` 会把 GPU 上的结果同步回 CPU。对调试 NaN 很有用，但放在实时路径里会让每帧都产生额外同步。

现在默认关闭：

```python
_DEBUG_NAN_DECODERS = os.environ.get("DEBUG_NAN", "0") == "1"
_DEBUG_NAN = os.environ.get("DEBUG_NAN", "0") == "1"
```

并且删除了 merged 路径里无条件开启的：

```python
_DEBUG_NAN = True
```

如果后续怀疑模型输出 NaN，再临时打开：

```bash
DEBUG_NAN=1 USE_COMPILE=0 python run_publisher.py ...
```

### 5. `USE_COMPILE=1` 时增加 compile warmup 耗时日志

这次命令使用的是 `USE_COMPILE=0`，不会走 torch compile。不过为了以后排查 `USE_COMPILE=1` 的真实开销，compile warmup 现在会打印每个 batch size 的耗时：

```python
warmup_start = time.perf_counter()
...
batch_warmup_start = time.perf_counter()
...
print(
    f"[SAM3DBodyEstimator] Warmup for batch_size={batch_size} "
    f"completed in {time.perf_counter() - batch_warmup_start:.3f}s"
)
...
print(
    "[SAM3DBodyEstimator] Multi-person compile warmup completed "
    f"in {time.perf_counter() - warmup_start:.3f}s"
)
```

如果以后开启：

```bash
USE_COMPILE=1 COMPILE_WARMUP_BATCH_SIZES=1,2,4 python run_publisher.py ...
```

启动阶段就可以直接看到编译 warmup 具体消耗，而不是把它误判成第一帧耗时。

### 6. 4090/5090 环境安装分支

`setup_env.sh` 现在会自动判断 GPU：

- RTX 4090：保持更新前的 CUDA 12.4 安装命令
- RTX 5090：使用 CUDA 12.8 / `cu128`

4090 保留旧命令：

```bash
conda install -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev ninja -y

pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

5090 使用：

```bash
conda install -c "nvidia/label/${CUDA_CONDA_LABEL}" \
    "cuda-nvcc=${CUDA_TOOLKIT_VERSION}.*" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_VERSION}.*" \
    "cuda-libraries-dev=${CUDA_TOOLKIT_VERSION}.*" \
    ninja -y

pip install "torch==${PYTORCH_VERSION}+${PYTORCH_CUDA}" \
    "torchvision==${TORCHVISION_VERSION}+${PYTORCH_CUDA}" \
    --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
```

当前默认映射：

```text
4090 -> CUDA 12.4, cu124, torch 2.5.1, torchvision 0.20.1
5090 -> CUDA 12.8, cu128, torch 2.7.1, torchvision 0.22.1
```

如需手动覆盖检测结果：

```bash
GPU_PROFILE=4090 bash setup_env.sh
GPU_PROFILE=5090 bash setup_env.sh
```

### 当前推荐实时命令

继续使用 `USE_COMPILE=0` 进行低启动成本调试：

```bash
USE_COMPILE=0 python /home/longxiaoze/liu_projects/Fast-SAM-3D-Body/run_publisher.py \
    --source camera \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
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

如果目标是最低实时耗时，而不是调试可视化，可以先关掉较重的 Rerun mesh overlay：

```bash
--no-rerun-mesh-overlay
```

或者完全不启用：

```bash
# 不加 --rerun
```

### 排查耗时的判断顺序

如果再次看到 `process_one_image TOTAL` 偏大，优先按这个顺序看：

1. 是否是刚启动后的前 1 到 2 帧冷启动
2. `human_detection` 是否明显偏大，可能是 YOLO TensorRT context 或输入尺寸变化
3. `model_run_inference` 是否明显偏大，重点看 `merged_forward`
4. 是否误开了 `DEBUG_NAN=1`
5. 是否误开了 `SAM3D_EMPTY_CACHE_EACH_FRAME=1`
6. 是否开启了 `USE_COMPILE=1` 但 compile warmup 没结束

正常 `USE_COMPILE=0`、真实路径 warmup 后，本机测试中 `process_one_image` 稳态约 `0.14s`。
