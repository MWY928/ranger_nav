# Real-World Simulation Diagnostics Summary

Date: 2026-07-17

## Terminology

这里不再用“语义”描述 STOP 行为，避免和 semantic sensor、VLM reasoning 混淆。

- semantic / panoptic sensor: 仿真里的语义或全景传感器，可能触发 `.scn` 语义文件读取。
- action meaning / termination logic: 环境如何解释 policy 输出的动作 id。
- aux heads: 训练辅助任务输出，例如人数、人体位置、未来轨迹预测；它们不是终止判断。

当前没有 VLM 或语言模型 reasoning 参与。policy 主动作是 4 维离散动作分布：

- `0`: `agent_0_discrete_stop`
- `1`: `agent_0_discrete_move_forward`
- `2`: `agent_0_discrete_turn_left`
- `3`: `agent_0_discrete_turn_right`

原版逻辑下，`act_id=0` 会直接设置 `task.is_stop_called=True` 和 `task.should_end=True`，因此 episode 结束。

## What We Checked

1. Config loading
   - 记录了 `pretrained_weights`、`checkpoint_folder`、`load_resume_state_config`、`eval_ckpt_path_dir`、`use_ckpt_config`、`should_load_ckpt`。
   - 最新 eval trace 显示 checkpoint 正在从 `evaluation/falcon/hm3d_real_world_lowmem_jaw_depth_only/checkpoints/latest.pth` 加载。

2. Done-step diagnostics
   - 记录每个 done 的 last action、STOP/pause 标记、human collision、distance to goal、reward。
   - 多轮 eval 中，episode 基本都是远离目标时由 `act_id=0` 结束。
   - `human_collision=0`，说明不是直接因为撞人结束。

3. Step diagnostics
   - 记录每步 `robot_pos_delta` 和 `distance_to_goal_reward`。
   - 已确认机器人实际在动：forward step 的位置增量约为 `0.0167m/step`。
   - geodesic distance 不是固定 100，也不是完全不变；距离会随机器人移动更新。

4. Action probabilities
   - 最新 trace 里记录了 action probabilities。
   - 10 个 done 中 9 个是 STOP 概率本身为 argmax，只有 1 个更像随机采样到了 STOP。
   - 因此 deterministic eval 也大概率会提前 STOP，问题不是单纯采样噪声。

## Current Hypotheses

1. Terminal STOP 代价过低
   - 原版 STOP 表示“我宣布到达目标并结束 episode”。
   - 在 real-world social navigation 中，`STOP`/等待也可能是合理避让动作。
   - 当前两者混在一起，policy 可能学到“早停可以避免后续碰撞、卡住和累计负奖励”。

2. 速度分布变化
   - 原 Falcon 训练里机器人速度较快。
   - real-world train 配置中机器人速度降到 `1.0`，eval 中为 `2.0`。
   - 目前已把 real-world train 中人类 oracle 速度也降到 `1.0`，eval 中降到 `2.0`，先消除“人比机器人快很多”的变量。

3. 机器人本体与碰撞模型
   - real-world task 中 agent radius 为 `0.5`。
   - 本体较大时更容易被窄通道、动态人群或场景碰撞约束影响，保守策略更容易出现。

4. Sensor / observation mismatch
   - localization lab sensor 不能从 task lab sensors 中删除，因为 reward 和 social metrics 依赖它。
   - 但是否放进 `habitat.gym.obs_keys` 决定 policy 是否看到它。
   - 训练和评估 obs keys 的差异需要持续核对。

## Implemented Switches

默认行为保持原版。只有显式打开环境变量时才切换到 real-world pause mode。

### Original Mode

```bash
./train_real_world_lowmem.sh
```

行为：

- `act_id=0` 是 terminal STOP。
- STOP 直接结束 episode。
- 成功需要 STOP 且距离小于 success distance。

### Real-World Pause Mode

```bash
REAL_WORLD_PAUSE_MODE=true ./train_real_world_lowmem.sh
```

行为：

- `act_id=0` 不再直接结束 episode，而是 pause/wait。
- 到达目标后自动成功：`distance_to_goal <= success_distance`。
- 连续 pause 超过 grace steps 后开始加惩罚。
- 连续 pause 超过 max steps 后失败结束。
- 连续 no-progress 超过 window 后失败结束。

可调参数：

```bash
REAL_WORLD_PAUSE_MODE=true \
PAUSE_GRACE_STEPS=60 \
PAUSE_MAX_STEPS=180 \
PAUSE_STEP_PENALTY=-0.002 \
PAUSE_DONE_PENALTY=-0.25 \
NO_PROGRESS_WINDOW=120 \
NO_PROGRESS_MOVE_EPS=0.005 \
NO_PROGRESS_DISTANCE_EPS=0.01 \
NO_PROGRESS_DONE_PENALTY=-0.25 \
./train_real_world_lowmem.sh
```

同样的开关也接入了：

- `debug_train_real_world_lowmem.sh`
- `evaluate_real_world_sim.sh`

## Next Training Plan

1. 先用 `REAL_WORLD_PAUSE_MODE=true` 从现有 checkpoint 微调。
2. 训练后用同样的 `REAL_WORLD_PAUSE_MODE=true` 跑 eval，避免训练/评估终止逻辑不一致。
3. 检查新的 trace：
   - `is_pause_called`
   - done reason 是否从远距离 STOP 变成 success、pause timeout 或 no-progress timeout
   - distance 是否持续下降
   - collision 是否上升
4. 如果 4-action pause mode 有效，再考虑更干净的 5-action 方案：
   - terminal DONE
   - non-terminal PAUSE
   - forward
   - turn left
   - turn right

5-action 方案含义最清晰，但会改变 action head，旧 checkpoint 不能完全无痛复用。
