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
| `4` | 遮挡搜索左转 | `Move(0, 0, +search_turn_speed)` |
| `5` | 遮挡搜索右转 | `Move(0, 0, -search_turn_speed)` |

Falcon 模型和动作滤波的输出空间仍严格保持 `0--3` 四个动作；`4/5` 仅由 bridge 在确认目标遮挡且允许搜索时产生，不会写入模型的 `prev_action`。映射节点以固定频率执行限加/减速，默认目标速度为 `forward_speed=0.6`、`turn_speed=0.6`、`search_turn_speed=0.25 rad/s`。

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


构建 Go2 ROS 包：

```bash
source /opt/ros/noetic/setup.bash
cd /home/mobile/wys/go2_falcon_main/ranger_nav_go2/ranger_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make --pkg go_nav
source devel/setup.bash
```

`go_nav` 的源码位于 `ranger_ws/src/go_nav`。请将它放入算法机实际使用的 catkin workspace 后再构建。

### 3.3 当前脚本的路径假设

仓库中的快捷脚本按照现有部署机编写，默认使用：

```text
/home/mobile/wys/go2_falcon_main/ranger_nav_go2            # 工作站仓库
/home/mobile/wys/go2_falcon_main/ranger_nav_go2/ranger_ws  # 工作站 catkin workspace
/home/unitree/go2_detection_ws                             # Jetson catkin workspace
/home/mobile/miniconda3                                    # Conda
/home/mobile/zzy/unitree_sdk2py_env                        # Unitree SDK2 venv
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

当前部署让 Go2 Jetson 充当 ROS master：

```text
算法机：192.168.123.100
Go2 Jetson / ROS master：192.168.123.18
```

请替换为实际网卡地址，不要把 `127.0.0.1`、Docker 地址或无法被另一台机器访问的地址设置为 `ROS_IP`。

### 4.1 算法机

在算法机的每个 ROS 终端中设置：

```bash
export ROS_MASTER_URI=http://192.168.123.18:11311
export ROS_IP=192.168.123.100
```

工作站上的 `~/go2_test_scripts/source_ros_pc_to_go2.sh` 应保存上述两个普通 shell 赋值；URL 不要写成 Markdown 的 `[http://...](http://...)` 形式。

### 4.2 Go2 Jetson

在 Jetson 的相机终端中设置：

```bash
export ROS_MASTER_URI=http://192.168.123.18:11311
export ROS_IP=192.168.123.18
```

如果 Jetson 上尚无 ROS master，先在 Jetson 启动 `roscore`。

确认两台机器能够互相 `ping`，并使用 NTP/chrony 同步时间。`run_bridge.sh` 默认只接受与深度帧时间差不超过 `0.50 s` 的极坐标消息，明显的时钟偏差或网络拥塞会导致策略持续输出停止。

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
cd /home/mobile/wys/go2_falcon_main/ranger_nav_go2
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
ODOM_TOPIC=/go2/sport_odom \
bash start_detection_full.sh
```

完整版本在标签短暂丢失后，使用 odom 中保存的标签位置继续发布 `/tag_polar`。默认在丢失 `0.12 s` 后开始外推，最多持续到丢失满 `6.0 s`，发布频率 `15 Hz`，与当前相机/深度频率一致。如果 Go2 的 odom 坐标或朝向符号尚未验证，应先使用简单版本，并单独验证 `/go2/sport_odom`。

预测时 tracker 会继续读取 `apriltag_ros` 以约 `15 Hz` 发布的空 detection array，并使用每帧数组的相机时间戳；因此预测 `/tag_polar` 与后续深度帧保持同一时间轴。检测数组、odom 任一停止更新时，tracker 会发布 NOT_READY 并停止预测/搜索，不会用旧数据盲目运动。

预测窗口结束后可以进入额外的低速原地搜索阶段。该功能默认关闭，需要 tracker 与 Falcon bridge 两端同时显式启用。状态流为 `VISIBLE(1) -> PREDICTING(2) -> SEARCHABLE(3) -> NOT_READY(0)`：先用 6 秒补偿完成绕障，然后以最后目标方位选择搜索方向；重见 Tag 立即 STOP 并退出搜索，状态/深度/odom 过期或搜索超过 12 秒也立即停车。

#### 推荐部署：在 Jetson 上运行 full detection

当 D435 与 AprilTag 检测已经位于 Jetson 时，建议把整个 `go_nav` 包同步到 Jetson，而不是只复制单个 Python 文件。Jetson 本地处理 `image_raw + camera_info`，工作站只向 Jetson 发送轻量的 `/go2/sport_odom`，Jetson 再向工作站发布 `/tag_polar`。Unitree SDK2 和对应 venv 仍留在工作站，Jetson 不需要安装。

从工作站同步包到 Jetson 的现有 Noetic workspace：

```bash
cd /home/mobile/wys/go2_falcon_main/ranger_nav_go2
rsync -av \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  ranger_ws/src/go_nav/ \
  unitree@192.168.123.18:/home/unitree/go2_detection_ws/src/go_nav/
```

`config/tags.yaml` 中的 Tag family、ID 和物理尺寸必须与实物一致。如果 Jetson 上已有经过标定的配置，同步前先保留并比较该文件。

在 Jetson（ROS Noetic）上构建：

```bash
source /opt/ros/noetic/setup.bash
cd /home/unitree/go2_detection_ws
rosdep install --from-paths src/go_nav --ignore-src -r -y
chmod +x src/go_nav/scripts/polar_goal_tracker.py
catkin_make --pkg go_nav -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

工作站继续运行 SportModeState bridge：

```bash
source /home/mobile/zzy/unitree_sdk2py_env/bin/activate
source ~/go2_test_scripts/source_ros_pc_to_go2.sh
cd /home/mobile/wys/go2_falcon_main/ranger_nav_go2
bash go2/start_sport_state_odom.sh
```

Jetson 必须先能收到工作站发布的 odom：

```bash
rostopic hz /go2/sport_odom
```

如果 Jetson 上已有独立运行且频率稳定的 `apriltag_ros`，保留该检测器，仅停止旧的 `polar_distance.py`（节点通常名为 `/tag_to_polar_node`），避免两个 `/tag_polar` 发布者。然后启动只包含 odom tracker 的 launch：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.123.18:11311
export ROS_IP=192.168.123.18
source /home/unitree/go2_detection_ws/devel/setup.bash

roslaunch go_nav go2_polar_tracker.launch \
  detections_topic:=/tag_detections \
  odom_topic:=/go2/sport_odom \
  polar_topic:=/tag_polar \
  tracking_state_topic:=/tag_tracking_state \
  lost_timeout_sec:=0.15 \
  predict_timeout_sec:=6.0 \
  publish_rate_hz:=15 \
  search_enabled:=true \
  search_timeout_sec:=12.0
```

如果原来的 `go2_detection_simple.launch` 把 AprilTag 和旧 polar 节点绑在同一个 roslaunch 中，则停止整个 simple launch，再改用包含 AprilTag 的 `go2_detection_full.launch`。

两台机器必须使用同一个 `ROS_MASTER_URI`，各自的 `ROS_IP` 必须是对方可访问的地址，并通过 NTP/chrony 同步时钟。工作站上不要再运行 `start_detection.sh` 或 `start_detection_full.sh`；Falcon bridge 直接订阅 Jetson 发布的 `/tag_polar`。

### 5.3 启动 Falcon 推理

使用默认 checkpoint，但保持搜索关闭：

```bash
cd /home/mobile/wys/go2_falcon_main/ranger_nav_go2
bash run_bridge.sh
```

若 Jetson tracker 已使用 `search_enabled:=true`，在工作站同时启用 Falcon 搜索仲裁：

```bash
cd /home/mobile/wys/go2_falcon_main/ranger_nav_go2
TAG_SEARCH_ENABLED=true \
TRACKING_STATE_TOPIC=/tag_tracking_state \
TAG_SEARCH_TIMEOUT_SEC=12.0 \
bash run_bridge.sh
```

或显式指定模型与 topic：

```bash
bash start_falcon_bridge.launch \
  --checkpoint /absolute/path/to/model.pth \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --polar_topic /tag_polar \
  --tracking_state_topic /tag_tracking_state \
  --action_topic /falcon/action_id \
  --tag_search_enabled true \
  --tracking_state_timeout_sec 0.5 \
  --tag_search_timeout_sec 12.0 \
  --tag_search_default_direction left \
  --deterministic \
  --action_filter_tau_sec 0.15 \
  --action_switch_margin 0.10 \
  --action_switch_hold_sec 0.12 \
  --stop_switch_hold_sec 0.20
```

检查离散动作和推理心跳：

```bash
rostopic echo /falcon/action_id
rostopic hz /falcon/obs_heartbeat
rostopic echo /tag_tracking_state
```

Falcon bridge 由深度帧驱动推理。`run_bridge.sh` 默认使用确定性 argmax，并对四个动作的概率执行时间型 EMA；新动作持续满足概率优势和确认时间后才会发布。滤波后的实际动作会作为 RNN 下一帧的 `prev_action`。

没有有效 `/tag_polar`、输入时间戳不匹配、回调异常或输入 watchdog 超时时，安全 Action `0` 会绕过 EMA 和确认时间立即发布。启用搜索后，只有新鲜的状态 `3` 才能输出 `4/5`；进入/退出搜索会先输出 `0` 并重置策略历史，重获 Tag 后下一帧再恢复 Falcon 推理。`run_bridge.sh` 的 watchdog 默认值为 `0.90 s`；直接运行 Python bridge 时默认值为 `0.3 s`。

### 5.4 先以 dry-run 验证 Unitree 映射

运行动作映射节点前，先确认 Unitree Python 环境能够导入 SDK：

```bash
source /home/mobile/zzy/unitree_sdk2py_env/bin/activate
source /opt/ros/noetic/setup.bash
unset UNITREE_CONDA_ENV
python3 -c "import sys, rospy, nav_msgs.msg, unitree_sdk2py; print('Unitree venv OK:', sys.executable)"
```

SDK2 可以安装在系统 Python、普通 venv 或 Conda 环境中。启动脚本默认使用当前
`python3`。当前部署使用普通 venv：先激活 venv，并且不要设置 `UNITREE_CONDA_ENV`；
只有明确使用 Conda 时才设置该变量。

然后启用 dry-run。此模式只打印 SDK 命令，不控制机器人：

```bash
cd /home/mobile/wys/go2_falcon_main/ranger_nav_go2

UNITREE_DRY_RUN=true \
UNITREE_NETWORK_INTERFACE=enxec9a0c1bc5be \
bash go2/start_action_mapper.sh
```

可在 dry-run 下手动验证动作映射：

```bash
rostopic pub -r 20 /falcon/action_id std_msgs/Int32 "data: 1"
rostopic pub -r 20 /falcon/action_id std_msgs/Int32 "data: 2"
rostopic pub -r 20 /falcon/action_id std_msgs/Int32 "data: 3"
rostopic pub -r 20 /falcon/action_id std_msgs/Int32 "data: 4"
rostopic pub -r 20 /falcon/action_id std_msgs/Int32 "data: 5"
rostopic pub -1 /falcon/action_id std_msgs/Int32 "data: 0"
```

前五条命令每次只运行一条，观察完成后按 `Ctrl-C`；持续发布可以覆盖 `0.3 s` 的动作 watchdog，并完整观察速度爬升过程。尤其先确认 `4` 确实向左、`5` 确实向右，再允许自动搜索。

### 5.5 启用 Go2 实际运动

确认图像、标签方位、Falcon 动作、左右转方向和遥控停止手段均正常后，再关闭 dry-run：

```bash
UNITREE_NETWORK_INTERFACE=enxec9a0c1bc5be \
UNITREE_DOMAIN_ID=0 \
FORWARD_SPEED=0.3 \
TURN_SPEED=0.4 \
SEARCH_TURN_SPEED=0.20 \
ACTION_TIMEOUT_SEC=0.3 \
UNITREE_DRY_RUN=false \
bash go2/start_action_mapper.sh
```

请先使用低速值完成空旷场地测试。当前算法机通过 USB 转网口 `enxec9a0c1bc5be` 访问 Go2 SDK2 通道；换机器或更换转接器后，可通过 `UNITREE_NETWORK_INTERFACE` 覆盖默认值。它不一定与 ROS 多机通信使用同一张网卡。

动作映射节点在以下情况调用 `StopMove()`：

- 收到 Action `0`。
- 收到未知 Action ID。
- 超过 `ACTION_TIMEOUT_SEC` 未收到新动作。
- 节点正常退出。

Action `0`、未知动作、watchdog 和退出始终立即停车，不经过速度斜坡。正常的前进/左转/右转以及低速原地搜索会在 `WATCHDOG_RATE_HZ` 控制循环中按线速度和角速度加减速限制渐变；bridge 在进入搜索前先发 `0`，避免从前进直接切换成带线速度的弧线。

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
| `/tag_tracking_state` | `std_msgs/UInt8` | polar tracker | Falcon bridge |
| `/go2/sport_odom` | `nav_msgs/Odometry` | SportModeState bridge | full tracker |
| `/falcon/action_id` | `std_msgs/Int32` | Falcon bridge | Unitree action mapper |
| `/falcon/obs_heartbeat` | `std_msgs/Header` | Falcon bridge | 调试/监控 |

`/tag_tracking_state` 的值为：`0=NOT_READY`、`1=VISIBLE`、`2=PREDICTING`、`3=SEARCHABLE`。搜索模式下仍应确认 `/falcon/action_id` 只有 Falcon bridge 一个发布者。

本分支的实机主链路不再使用 `/cmd_vel`。如果其他节点仍在等待 `/cmd_vel`，不会收到 Falcon 的控制输出。

## 7. 常用参数

### AprilTag/极坐标

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GO2_IMAGE_TOPIC` | `/camera/color/image_raw` | 彩色图 |
| `GO2_CAMERA_INFO_TOPIC` | `/camera/color/camera_info` | 彩色相机内参 |
| `TAG_DETECTIONS_TOPIC` | `/tag_detections` | AprilTag 检测输出 |
| `POLAR_TOPIC` | `/tag_polar` | Falcon 极坐标目标 |
| `TRACKING_STATE_TOPIC` | `/tag_tracking_state` | tracker 到 Falcon 的安全状态心跳 |
| `TARGET_TAG_ID` | `0` | 目标标签 ID |
| `USE_FIRST_DETECTION` | `false` | 忽略 ID，使用第一个检测 |
| `THETA_OFFSET_RAD` | `0.0` | 相机安装角修正 |
| `THETA_DEADBAND_RAD` | `0.0` | 角度死区 |
| `DISTANCE_OFFSET` | `0.6` | 从测量距离中减去的偏移 |
| `MIN_DISTANCE` | `0.0` | 输出距离下限 |
| `USE_ODOM_FALLBACK` | full 脚本为 `true` | 启用标签丢失后的 odom 外推 |
| `ODOM_TOPIC` | `/go2/sport_odom` | Go2 里程计 topic |
| `PREDICT_TIMEOUT_SEC` | `6.0` | 从最后一次可见开始计算的最大预测时长 |
| `PREDICT_RATE_HZ` | `15.0` | 预测与状态心跳频率 |
| `REACQUIRE_RESET_SEC` | `1.0` | 遮挡超过该时间后，重捕获时重置目标估计而非继续 EMA |
| `DETECTION_STREAM_TIMEOUT_SEC` | `0.5` | 空 detection array 也停止更新时禁止预测/搜索 |
| `TAG_SEARCH_ENABLED` | `false` | tracker/bridge 均需开启，才允许低速搜索 |
| `TAG_SEARCH_TIMEOUT_SEC` | `12.0` | 6 秒预测结束后的额外搜索时限 |

### Unitree 动作映射

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ACTION_TOPIC` | `/falcon/action_id` | 离散动作 topic |
| `UNITREE_CONDA_ENV` | 未设置 | 可选的 Unitree SDK2 Conda 环境；非 Conda 安装不要设置 |
| `UNITREE_PYTHON_BIN` | `python3` | 能同时导入 `unitree_sdk2py` 和 `rospy` 的 Python 命令或绝对路径 |
| `UNITREE_NETWORK_INTERFACE` | `enxec9a0c1bc5be` | 当前算法机连接 Go2 的 USB 转网口；可按机器覆盖 |
| `UNITREE_DOMAIN_ID` | `0` | SDK2 DDS domain ID |
| `UNITREE_TIMEOUT_SEC` | `10.0` | SportClient 调用超时 |
| `FORWARD_SPEED` | `0.6` | 前进速度 |
| `TURN_SPEED` | `0.6` | 转向角速度 |
| `SEARCH_TURN_SPEED` | `0.25` | Action `4/5` 的原地搜索角速度 |
| `ACTION_TIMEOUT_SEC` | `0.3` | 动作流 watchdog |
| `WATCHDOG_RATE_HZ` | `20.0` | watchdog 与速度斜坡控制频率 |
| `VELOCITY_SMOOTHING_ENABLED` | `true` | 启用目标速度限加/减速 |
| `LINEAR_ACCEL_LIMIT` | `1.0` | 线加速度上限，单位 `m/s²` |
| `LINEAR_DECEL_LIMIT` | `1.5` | 线减速度上限，单位 `m/s²` |
| `YAW_ACCEL_LIMIT` | `2.0` | 角加速度上限，单位 `rad/s²` |
| `YAW_DECEL_LIMIT` | `3.0` | 角减速度上限，单位 `rad/s²` |
| `UNITREE_DRY_RUN` | `false` | 只记录命令，不控制机器人 |

### Falcon 动作滤波

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ACTION_FILTER_ENABLED` | `true` | 对 categorical 动作概率启用 EMA 与迟滞 |
| `ACTION_FILTER_TAU_SEC` | `0.15` | EMA 时间常数；越大越平滑，但响应越慢 |
| `ACTION_SWITCH_MARGIN` | `0.10` | 新动作相对当前动作所需的概率优势 |
| `ACTION_SWITCH_HOLD_SEC` | `0.12` | 普通动作切换确认时间 |
| `STOP_SWITCH_HOLD_SEC` | `0.20` | 策略 STOP 的确认时间；安全 STOP 不受此参数影响 |
| `TRACKING_STATE_TIMEOUT_SEC` | `0.50` | 搜索状态心跳最大允许间隔 |
| `TAG_SEARCH_DEFAULT_DIRECTION` | `left` | 最后目标角度恰为零/无效时的备用搜索方向 |

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
- 检查深度图与 `/tag_polar` 的时间戳；`run_bridge.sh` 默认最大允许差值为 `0.50 s`。
- 如果旧日志在丢失约 `1.05 s` 后才开始 mismatch，原因通常是旧版 `predict_timeout_sec=1.0` 到期，而不是刚进入预测就错时钟；同步新版 tracker 后默认窗口为 `6.0 s`。
- 新版预测 stamp 来自持续发布的 detection array 相机时间轴。若 `/tag_detections` 本身停止，tracker 会安全转为状态 `0`，不应增大 `MAX_POLAR_AGE_SEC` 掩盖问题。
- 预测满 6 秒后：未启用搜索会按设计停车；已在 tracker 和 bridge 两端启用搜索则应看到 `/tag_tracking_state: 3`，随后 `/falcon/action_id: 4/5`。
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
- 使用启动脚本打印出的 Python 路径确认能够导入 `unitree_sdk2py` 和 `rospy`。
- 确认 `UNITREE_NETWORK_INTERFACE` 指向 Go2 控制网络。
- 查看动作映射节点是否打印 `SportClient ready`，以及 SDK 是否返回非零错误码。
- 确认 Go2 当前模式允许 `SportClient.Move()` 控制。

### Go2 一直收到停止动作

- `run_bridge.sh` 的 Falcon 输入 watchdog 默认为 `0.90 s`；直接运行 bridge 时为 `0.3 s`。
- 动作映射节点的 watchdog 也默认为 `0.3 s`。
- 原始 RGB-D 图像带宽较高；优先使用稳定的有线网络或高质量无线网络，并检查 `rostopic hz` 是否持续。
- 检查 mapper 日志中是否出现 `No action for ... stopping Go2`，并测量 `/falcon/action_id` 的最大或 p99 间隔，不要只看平均频率。
- 不要在未验证前放宽 watchdog。应先定位相机掉帧、网络拥塞或推理延迟；确需调整时，应让 mapper timeout 大于动作间隔的 p99 加安全裕量，同时评估失联后继续运动更久的风险。

### 左右方向相反或目标中心有固定偏差

当前约定为 `theta > 0` 左转、`theta < 0` 右转，但旧的 `follow_tag.py` 使用过相反符号。先在 dry-run 中验证 Action `2/3/4/5` 的实际方向，再允许自动搜索并调整 `THETA_OFFSET_RAD`；不要直接交换动作 ID 来掩盖相机坐标或安装方向问题。

## 9. 仓库中的关键文件

```text
ranger_ws/src/go_nav/
├── config/tags.yaml                    # AprilTag family、ID 和尺寸
├── launch/go2_detection_simple.launch  # AprilTag + 直接极坐标
├── launch/go2_detection_full.launch    # AprilTag + 可选 odom 外推
├── launch/go2_polar_tracker.launch     # 复用已有 AprilTag + odom 外推
├── launch/go2_action_mapper.launch     # 离散动作到 Unitree SDK2
└── scripts/
    ├── polar_distance.py
    ├── polar_goal_tracker.py
    └── unitree_action_mapper.py

sensor/action_filter.py                  # 动作概率 EMA、迟滞与 STOP 确认
sensor/falcon_ros_bridge.py              # 深度 + 目标 → Falcon → 滤波后 Action ID
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
