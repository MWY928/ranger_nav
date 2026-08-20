# Falcon on Unitree Go2 + RealSense D435

本分支在原始 [Falcon](https://zeying-gong.github.io/projects/falcon/) 社会导航算法的基础上，增加了 **Unitree Go2 实机接口、D435 远程图像输入、AprilTag 目标定位和 ROS1 多机部署**。当前实机链路使用深度图与极坐标目标作为 Falcon 输入，策略输出离散动作，再通过 Unitree SDK2 控制 Go2。

> 当前分支：`falcon/go2-d435-april`
>
> ROS 版本：ROS1 Noetic
>
> 默认相机：Intel RealSense D435
>
> 默认目标：`tag36h11`、ID `0`、边长 `0.08 m`

原始项目：[Project Website](https://zeying-gong.github.io/projects/falcon/) · [Paper](https://arxiv.org/abs/2409.13244) · [Video](https://www.youtube.com/watch?v=elNI7XlRyvU)

## 1. 本分支与原版的主要区别

| 模块 | 原版/通用 ROS 版本 | 当前 Go2 分支 |
| --- | --- | --- |
| 相机 | 算法机本地启动相机或仿真传感器 | Go2 Jetson 启动 `realsense2_ros`，通过 ROS 多机网络发送图像 |
| 目标输入 | 仿真 PointGoal 或通用目标接口 | `apriltag_ros` 检测目标，并转换成 `/tag_polar` |
| 策略输出 | 直接发布 `/cmd_vel` | 发布 `/falcon/action_id`，类型为 `std_msgs/Int32` |
| Go2 控制 | 无 Unitree 专用接口 | `unitree_sdk2py` 的 `SportClient.Move/StopMove` |
| ROS 包 | 主要使用 `ranger_nav` | 新增独立的 `go_nav` 包和 Go2 专用 launch |
| 安全机制 | 单一推理链路 | Falcon 输入 watchdog + Unitree 动作 watchdog |

当前动作定义如下：

| Action ID | 含义 | Unitree SDK2 命令 |
| --- | --- | --- |
| `0` | 停止 | `StopMove()` |
| `1` | 前进 | `Move(forward_speed, 0, 0)` |
| `2` | 左转 | `Move(0, 0, +turn_speed)` |
| `3` | 右转 | `Move(0, 0, -turn_speed)` |

策略本身不再负责生成连续速度。速度由动作映射节点统一设置，默认 `forward_speed=0.6`、`turn_speed=0.6`。

## 2. 系统架构

推荐使用两台计算机：

- **Go2 Jetson**：连接 D435，只运行 `realsense2_ros`，发布彩色图、相机内参和对齐深度图。
- **算法机**：运行 ROS master、AprilTag 检测、目标极坐标转换、Falcon 推理和 Unitree 动作映射。动作映射也可以放到 Jetson，只要该机器同时能访问 ROS master 和 Go2 的 Unitree SDK2 网络。

```text
Go2 Jetson
┌─────────────────────────────────────────────────────────────┐
│ D435 → realsense2_ros                                      │
│   ├─ /camera/color/image_raw                               │
│   ├─ /camera/color/camera_info                             │
│   └─ /camera/aligned_depth_to_color/image_raw              │
└──────────────────────────┬──────────────────────────────────┘
                           │ ROS1 多机网络
                           ▼
算法机
┌─────────────────────────────────────────────────────────────┐
│ color + camera_info → apriltag_ros → /tag_detections       │
│                                      │                      │
│                                      ▼                      │
│                            go_nav polar node                │
│                                      │ /tag_polar           │
│ aligned depth ────────────────────────┤                      │
│                                      ▼                      │
│                         Falcon ROS bridge                   │
│                                      │ /falcon/action_id    │
│                                      ▼                      │
│                      Unitree action mapper                  │
│                                      │ unitree_sdk2py       │
└──────────────────────────────────────┼──────────────────────┘
                                       ▼
                                  Unitree Go2
```

## 3. 运行环境与依赖

### 3.1 硬件与网络

- Unitree Go2，且运行动作映射节点的机器能够访问 Go2 运动控制网络。
- Intel RealSense D435，连接到 Go2 Jetson。
- Jetson 与算法机位于同一局域网，能够互相直接访问对方声明的 `ROS_IP`。
- 算法机建议使用 NVIDIA GPU 进行 Falcon 推理。
- 实机首次测试时应保留遥控器/急停手段，并先使用 dry-run。

ROS1 节点会使用随机 TCP 端口建立连接，因此不能只开放 `11311`。两台机器之间应能够双向直连，防火墙和路由也需要允许 ROS 节点通信。

### 3.2 软件

- Ubuntu 20.04 + ROS Noetic。
- Go2 Jetson：`realsense2_camera`。
- 算法机：`apriltag_ros`、`rospy`、`geometry_msgs`、`sensor_msgs`、`nav_msgs`、`std_msgs`。
- Falcon 环境：Python 3.9、PyTorch、OpenCV、Gym，以及本仓库中的 `habitat-lab` 和 `habitat-baselines`。
- Unitree 环境：`unitree_sdk2py` 和可用的 `rospy`。

Falcon 基础环境可按以下方式安装：

```bash
conda create -n falcon python=3.9 cmake=3.14.0
conda activate falcon

conda install habitat-sim=0.3.1 withbullet headless \
  -c conda-forge -c aihabitat

pip install -e habitat-lab
pip install -e habitat-baselines
pip install -r requirements.txt
```

构建 Go2 ROS 包：

```bash
source /opt/ros/noetic/setup.bash
cd /home/mobile/ranger_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make --pkg go_nav
source devel/setup.bash
```

`go_nav` 的源码位于 `ranger_ws/src/go_nav`。请将它放入算法机实际使用的 catkin workspace 后再构建。

### 3.3 当前脚本的路径假设

仓库中的快捷脚本按照现有部署机编写，默认使用：

```text
/home/mobile/ranger_nav                # 仓库
/home/mobile/ranger_ws                 # catkin workspace
/home/mobile/catkin_ws                 # 其他 ROS 包（若存在）
/home/mobile/miniconda3                # Conda
conda env: falcon / simple_nav / unitree
```

如果用户名、仓库位置、workspace 或 Conda 安装位置不同，请修改以下脚本中的 `source` 和路径：

- `start_detection.sh`
- `start_detection_full.sh`
- `run_bridge.sh`
- `start_falcon_bridge.launch`
- `go2/start_action_mapper.sh`
- `sensor/falcon_ros_bridge.py` 顶部的 Python 搜索路径

注意：`start_falcon_bridge.launch` 实际是 Bash 包装脚本，不是 ROS XML launch 文件，应使用 `bash start_falcon_bridge.launch ...`，不要对它执行 `roslaunch`。

### 3.4 模型文件

`.pth`、`.pt` 和 `.ckpt` 已被 `.gitignore` 排除，仓库不会自动包含实机 checkpoint。默认快捷脚本期望：

```text
weights/falcon_bc_70traj_action_head_lstm0045.pth
```

请将兼容的模型放到该位置，或使用 `start_falcon_bridge.launch --checkpoint /absolute/path/model.pth` 显式指定。默认网络参数为：

```text
resolution=256
hidden_size=512
num_recurrent_layers=2
backbone=resnet50
rnn_type=LSTM
depth_obs_key=articulated_agent_jaw_depth
goal_obs_key=pointgoal_with_gps_compass
```

模型结构或 observation key 不一致时，需要同时向 bridge 传入对应参数。

## 4. 配置 ROS1 多机通信

以下示例让算法机充当 ROS master。示例地址为：

```text
算法机：192.168.123.10
Go2 Jetson：192.168.123.20
```

请替换为实际网卡地址，不要把 `127.0.0.1`、Docker 地址或无法被另一台机器访问的地址设置为 `ROS_IP`。

### 4.1 算法机

在算法机的每个 ROS 终端中设置：

```bash
export ROS_MASTER_URI=http://192.168.123.10:11311
export ROS_IP=192.168.123.10
```

然后启动 ROS master：

```bash
roscore
```

### 4.2 Go2 Jetson

在 Jetson 的相机终端中设置：

```bash
export ROS_MASTER_URI=http://192.168.123.10:11311
export ROS_IP=192.168.123.20
```

确认两台机器能够互相 `ping`，并使用 NTP/chrony 同步时间。Falcon bridge 默认只接受与深度帧时间差不超过 `0.12 s` 的极坐标消息，明显的时钟偏差或网络拥塞会导致策略持续输出停止。

## 5. 实机启动顺序

### 5.1 启动 Jetson 上的 D435

在 Go2 Jetson 上运行：

```bash
source /opt/ros/noetic/setup.bash

roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  enable_color:=true \
  enable_depth:=true
```

算法机应能看到以下三个必要话题：

```bash
rostopic list | grep camera
rostopic hz /camera/color/image_raw
rostopic echo -n 1 /camera/color/camera_info
rostopic hz /camera/aligned_depth_to_color/image_raw
```

如果 RealSense 使用了不同 namespace，请在后续步骤覆盖对应 topic。当前 bridge 的正确默认深度话题是：

```text
/camera/aligned_depth_to_color/image_raw
```

### 5.2 启动 AprilTag 检测与极坐标目标

默认使用简单版本：

```bash
cd /home/mobile/ranger_nav
bash start_detection.sh
```

它在算法机上执行：

```text
/camera/color/image_raw
  → apriltag_ros
  → /tag_detections
  → go_nav/polar_distance.py
  → /tag_polar
```

检查输出：

```bash
rostopic echo -n 1 /tag_detections
rostopic echo /tag_polar
```

`/tag_polar` 为 `geometry_msgs/PointStamped`：

- `point.x`：目标距离 `r`，单位米。
- `point.y`：目标方位角 `theta`，单位弧度；正值表示目标在左侧，负值表示目标在右侧。
- `point.z`：AprilTag ID。

默认距离计算为：

```text
r = max(min_distance, measured_distance - distance_offset)
```

其中 `distance_offset=0.6 m`。可通过环境变量调整：

```bash
TARGET_TAG_ID=0 \
DISTANCE_OFFSET=0.6 \
THETA_OFFSET_RAD=0.0 \
THETA_DEADBAND_RAD=0.0 \
bash start_detection.sh
```

检测不同 ID 或尺寸的 AprilTag 时，还必须同步修改 `ranger_ws/src/go_nav/config/tags.yaml` 中的 `standalone_tags`。仅修改 `TARGET_TAG_ID` 不会自动更新标签物理尺寸。

#### 可选：遮挡期间使用里程计外推

如已有可靠的 `nav_msgs/Odometry`：

```bash
USE_ODOM_FALLBACK=true \
ODOM_TOPIC=/odom \
bash start_detection_full.sh
```

完整版本在标签短暂丢失后，使用 odom 中保存的标签位置继续发布 `/tag_polar`。默认在丢失 `0.30 s` 后开始外推，最多持续 `5.0 s`，发布频率 `20 Hz`。如果 Go2 的 odom 坐标、朝向符号或时间戳尚未验证，应保持默认的 `USE_ODOM_FALLBACK=false`。

### 5.3 启动 Falcon 推理

使用默认 checkpoint：

```bash
cd /home/mobile/ranger_nav
bash run_bridge.sh
```

或显式指定模型与 topic：

```bash
bash start_falcon_bridge.launch \
  --checkpoint /absolute/path/to/model.pth \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --polar_topic /tag_polar \
  --action_topic /falcon/action_id
```

检查离散动作和推理心跳：

```bash
rostopic echo /falcon/action_id
rostopic hz /falcon/obs_heartbeat
```

Falcon bridge 由深度帧驱动推理。没有有效 `/tag_polar`、输入时间戳不匹配、回调异常或成功推理超过 `0.3 s` 未更新时，它会发布 Action `0`。

### 5.4 先以 dry-run 验证 Unitree 映射

运行动作映射节点前，先确认 Unitree Python 环境能够导入 SDK：

```bash
conda activate unitree
python -c "import unitree_sdk2py; print('unitree_sdk2py OK')"
```

然后启用 dry-run。此模式只打印 SDK 命令，不控制机器人：

```bash
cd /home/mobile/ranger_nav

UNITREE_DRY_RUN=true \
UNITREE_NETWORK_INTERFACE=eth0 \
bash go2/start_action_mapper.sh
```

可在 dry-run 下手动验证动作映射：

```bash
rostopic pub -1 /falcon/action_id std_msgs/Int32 "data: 1"
rostopic pub -1 /falcon/action_id std_msgs/Int32 "data: 2"
rostopic pub -1 /falcon/action_id std_msgs/Int32 "data: 3"
rostopic pub -1 /falcon/action_id std_msgs/Int32 "data: 0"
```

### 5.5 启用 Go2 实际运动

确认图像、标签方位、Falcon 动作、左右转方向和遥控停止手段均正常后，再关闭 dry-run：

```bash
UNITREE_NETWORK_INTERFACE=eth0 \
UNITREE_DOMAIN_ID=0 \
FORWARD_SPEED=0.3 \
TURN_SPEED=0.4 \
ACTION_TIMEOUT_SEC=0.3 \
UNITREE_DRY_RUN=false \
bash go2/start_action_mapper.sh
```

请先使用低速值完成空旷场地测试。`UNITREE_NETWORK_INTERFACE` 必须是能够访问 Go2 SDK2 通道的实际网卡名，例如 `eth0` 或 `enp...`；它不一定与 ROS 多机通信使用同一张网卡。

动作映射节点在以下情况调用 `StopMove()`：

- 收到 Action `0`。
- 收到未知 Action ID。
- 超过 `ACTION_TIMEOUT_SEC` 未收到新动作。
- 节点正常退出。

默认不会在启动时自动执行 `BalanceStand()`。如确有需要，可设置：

```bash
UNITREE_BALANCE_STAND_ON_START=true bash go2/start_action_mapper.sh
```

## 6. Topic 一览

| Topic | 类型 | 发布者 | 使用者 |
| --- | --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | Jetson `realsense2_ros` | `apriltag_ros` |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Jetson `realsense2_ros` | `apriltag_ros` |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Jetson `realsense2_ros` | Falcon bridge |
| `/tag_detections` | `apriltag_ros/AprilTagDetectionArray` | `apriltag_ros` | polar node |
| `/tag_polar` | `geometry_msgs/PointStamped` | polar node | Falcon bridge |
| `/odom` | `nav_msgs/Odometry` | Go2/定位节点 | full tracker，可选 |
| `/falcon/action_id` | `std_msgs/Int32` | Falcon bridge | Unitree action mapper |
| `/falcon/obs_heartbeat` | `std_msgs/Header` | Falcon bridge | 调试/监控 |

本分支的实机主链路不再使用 `/cmd_vel`。如果其他节点仍在等待 `/cmd_vel`，不会收到 Falcon 的控制输出。

## 7. 常用参数

### AprilTag/极坐标

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GO2_IMAGE_TOPIC` | `/camera/color/image_raw` | 彩色图 |
| `GO2_CAMERA_INFO_TOPIC` | `/camera/color/camera_info` | 彩色相机内参 |
| `TAG_DETECTIONS_TOPIC` | `/tag_detections` | AprilTag 检测输出 |
| `POLAR_TOPIC` | `/tag_polar` | Falcon 极坐标目标 |
| `TARGET_TAG_ID` | `0` | 目标标签 ID |
| `USE_FIRST_DETECTION` | `false` | 忽略 ID，使用第一个检测 |
| `THETA_OFFSET_RAD` | `0.0` | 相机安装角修正 |
| `THETA_DEADBAND_RAD` | `0.0` | 角度死区 |
| `DISTANCE_OFFSET` | `0.6` | 从测量距离中减去的偏移 |
| `MIN_DISTANCE` | `0.0` | 输出距离下限 |
| `USE_ODOM_FALLBACK` | `false` | 启用标签丢失后的 odom 外推 |
| `ODOM_TOPIC` | `/odom` | 里程计 topic |

### Unitree 动作映射

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ACTION_TOPIC` | `/falcon/action_id` | 离散动作 topic |
| `UNITREE_CONDA_ENV` | `unitree` | Unitree SDK2 Conda 环境 |
| `UNITREE_NETWORK_INTERFACE` | `eth0` | Unitree SDK2 网卡 |
| `UNITREE_DOMAIN_ID` | `0` | SDK2 DDS domain ID |
| `UNITREE_TIMEOUT_SEC` | `10.0` | SportClient 调用超时 |
| `FORWARD_SPEED` | `0.6` | 前进速度 |
| `TURN_SPEED` | `0.6` | 转向角速度 |
| `ACTION_TIMEOUT_SEC` | `0.3` | 动作流 watchdog |
| `UNITREE_DRY_RUN` | `false` | 只记录命令，不控制机器人 |

## 8. 故障排查

### 算法机看不到 Jetson 图像

依次检查：

```bash
echo "$ROS_MASTER_URI"
echo "$ROS_IP"
rosnode list
rostopic list
```

常见原因是两台机器使用了不同的 `ROS_MASTER_URI`、`ROS_IP` 绑定到错误网卡、主机名无法解析，或防火墙阻止了 ROS 节点的随机端口。

### 有彩色图，但没有 `/tag_polar`

- 确认 `/camera/color/camera_info` 存在且分辨率与彩色图一致。
- 用 `rqt_image_view` 检查标签是否清晰、完整并位于视野内。
- 核对 `tag36h11`、标签 ID 和 `tags.yaml` 中的物理尺寸。
- 检查 `/tag_detections`；若该 topic 为空，问题位于 AprilTag 检测之前。
- 默认只选择 ID `0`。临时调试可设置 `USE_FIRST_DETECTION=true`。

### `/tag_polar` 正常，但没有 Falcon 动作

- 确认对齐深度 topic 名称为 `/camera/aligned_depth_to_color/image_raw`。
- 检查深度图与 `/tag_polar` 的时间戳；默认最大允许差值为 `0.12 s`。
- 确认 checkpoint 文件存在，且网络结构与 bridge 参数一致。
- 使用调试参数查看深度预处理和动作概率：

```bash
bash start_falcon_bridge.launch \
  --checkpoint /absolute/path/to/model.pth \
  --debug_mapping \
  --debug_depth
```

### 有 `/falcon/action_id`，但 Go2 不动

- 确认 `UNITREE_DRY_RUN=false`。
- 在 `unitree` 环境中确认能够导入 `unitree_sdk2py`。
- 确认 `UNITREE_NETWORK_INTERFACE` 指向 Go2 控制网络。
- 查看动作映射节点是否打印 `SportClient ready`，以及 SDK 是否返回非零错误码。
- 确认 Go2 当前模式允许 `SportClient.Move()` 控制。

### Go2 一直收到停止动作

- Falcon bridge 的输入 watchdog 默认为 `0.3 s`。
- 动作映射节点的 watchdog 也默认为 `0.3 s`。
- 原始 RGB-D 图像带宽较高；优先使用稳定的有线网络或高质量无线网络，并检查 `rostopic hz` 是否持续。
- 不要在未验证前放宽 watchdog。应先定位相机掉帧、网络拥塞或推理延迟。

### 左右方向相反或目标中心有固定偏差

当前约定为 `theta > 0` 左转、`theta < 0` 右转。先在 dry-run 中移动标签验证符号，再调整 `THETA_OFFSET_RAD`；不要直接交换 Action `2/3` 来掩盖相机坐标或安装方向问题。

## 9. 仓库中的关键文件

```text
ranger_ws/src/go_nav/
├── config/tags.yaml                    # AprilTag family、ID 和尺寸
├── launch/go2_detection_simple.launch  # AprilTag + 直接极坐标
├── launch/go2_detection_full.launch    # AprilTag + 可选 odom 外推
├── launch/go2_action_mapper.launch     # 离散动作到 Unitree SDK2
└── scripts/
    ├── polar_distance.py
    ├── polar_goal_tracker.py
    └── unitree_action_mapper.py

sensor/falcon_ros_bridge.py              # 深度 + 目标 → Falcon → Action ID
start_detection.sh                       # 简单目标检测入口
start_detection_full.sh                  # 完整目标跟踪入口
run_bridge.sh                            # 默认 Falcon 实机推理入口
start_falcon_bridge.launch               # 可传参数的 Bash 推理入口
go2/start_action_mapper.sh               # Unitree 动作映射入口
record_real_world_trajectory.sh          # 实机推理样本记录
```

## 10. 原始 Falcon 仿真、评测与训练

本分支仍保留原始 Habitat 仿真、Social-HM3D/Social-MP3D 评测和训练代码。数据集准备方式见 [DATASETS.md](DATASETS.md) 及原始 [Falcon 项目说明](https://github.com/Zeying-Gong/Falcon)。

评测示例：

```bash
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d.yaml
```

训练示例：

```bash
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_train.yaml
```

多 GPU 训练：

```bash
sh habitat-baselines/habitat_baselines/rl/ddppo/single_node_falcon.sh
```

## Citation

如果本仓库对你的研究有帮助，请引用原始 Falcon 工作：

```bibtex
@article{gong2024cognition,
  title={From Cognition to Precognition: A Future-Aware Framework for Social Navigation},
  author={Gong, Zeying and Hu, Tianshuai and Qiu, Ronghe and Liang, Junwei},
  journal={arXiv preprint arXiv:2409.13244},
  year={2024}
}
```

## Acknowledgments

- [Falcon](https://github.com/Zeying-Gong/Falcon)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [Proximity](https://github.com/EnricoCancelli/ProximitySocialNav)
- Unitree SDK2 / `unitree_sdk2py`
- `realsense2_ros`
- `apriltag_ros`
