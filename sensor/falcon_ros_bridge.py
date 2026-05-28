#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Falcon ROS bridge.

Data flow (runtime):
1) Subscribe Depth + goal signal (polar topic PointStamped).
2) Build policy observation in Falcon/PointNav format.
3) Run policy inference (Discrete(4)).
4) Map discrete action to Twist and publish to cmd_vel.

Safety behavior:
- If goal input is missing/too old, publish stop.
- If callback fails, publish stop.
- If input stream stalls (watchdog timeout), publish stop.
"""

import argparse
from typing import Dict, Optional, Tuple
from collections import deque
import json
import os

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PointStamped, Twist
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import sys
sys.path.append("/home/mobile/ranger_nav/habitat-baselines/")
sys.path.append("/home/mobile/ranger_nav/habitat-lab/")
sys.path.append("/home/mobile/ranger_nav")

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs


def _extract_actor_critic_state_dict(ckpt_obj: Dict) -> Dict[str, torch.Tensor]:
    """
    Accept different checkpoint layouts and return state_dict keys
    compatible with PointNavResNetPolicy.
    """
    # 训练脚本保存 checkpoint 的层级可能不同，这里统一整理成 policy 可加载的 key。
    src = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(src, dict):
        raise RuntimeError("Unsupported checkpoint format: state_dict is not a dict.")

    out = {}
    for k, v in src.items():
        if not isinstance(k, str):
            continue
        if not torch.is_tensor(v):
            continue
        if "actor_critic." in k:
            kk = k.split("actor_critic.", 1)[1]
            out[kk] = v
        elif k.startswith("net.") or k.startswith("action_distribution.") or k.startswith("critic."):
            out[k] = v
    if len(out) == 0:
        out = {
            k: v
            for k, v in src.items()
            if isinstance(k, str) and torch.is_tensor(v)
        }
    if len(out) == 0:
        raise RuntimeError("No string-key tensor parameters found in checkpoint.")
    return out


def _select_agent0_checkpoint_state_dict(ckpt_obj):
    """Return the policy state_dict from common single-agent or multi-agent ckpt layouts."""
    if isinstance(ckpt_obj, (list, tuple)) and len(ckpt_obj) > 0 and isinstance(ckpt_obj[0], dict):
        return ckpt_obj[0].get("state_dict", ckpt_obj[0])

    if isinstance(ckpt_obj, dict):
        if 0 in ckpt_obj and isinstance(ckpt_obj[0], dict):
            return ckpt_obj[0].get("state_dict", ckpt_obj[0])
        if "0" in ckpt_obj and isinstance(ckpt_obj["0"], dict):
            return ckpt_obj["0"].get("state_dict", ckpt_obj["0"])
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]

    return ckpt_obj


class FalconRosBridge(object):
    """Bridge ROS sensor streams to Falcon policy and publish cmd_vel."""

    def __init__(self, args):
        # 初始化 ROS 节点和推理设备。
        rospy.init_node("falcon_ros_bridge", anonymous=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo("Falcon device: %s", str(self.device))

        # 保存运行参数：输入分辨率、深度范围、推理模式和调试开关。
        self.resolution = args.resolution
        self.max_depth_m = args.max_depth_m
        self.deterministic = args.deterministic
        self.require_strict_ckpt = args.strict_checkpoint
        self.debug_mapping = args.debug_mapping
        self.debug_depth = args.debug_depth
        self.debug_depth_dump_dir = args.debug_depth_dump_dir
        self._depth_sample_saved = False
        self.replay_dump_enabled = args.replay_dump_enabled
        self.replay_dump_dir = args.replay_dump_dir
        self.replay_dump_limit = args.replay_dump_limit
        self._replay_dump_count = 0
        self._replay_dump_limit_reached = False

        # Policy obs keys should match your social_nav_v2 config.
        # 这些 key 必须和训练 Falcon/PointNav 模型时的 observation space 的key一致。
        self.depth_key = args.depth_obs_key
        self.goal_key = args.goal_obs_key

        # 缓存最近的极坐标目标消息，用于和深度图按时间戳对齐。
        self.latest_polar_msg = None
        self.polar_buffer = deque(maxlen=max(10, args.polar_buffer_size))
        self.last_obs_time = rospy.Time(0)
        self.data_timeout_sec = args.data_timeout_sec
        self.max_polar_age_sec = args.max_polar_age_sec

        # 构建策略网络并加载训练好的 checkpoint。
        self.actor_critic = self._build_policy(
            checkpoint_path=args.checkpoint,
            hidden_size=args.hidden_size,
            num_recurrent_layers=args.num_recurrent_layers,
            backbone=args.backbone,
            rnn_type=args.rnn_type,
        )
        self.actor_critic.eval()

        # RNN/LSTM 策略需要跨帧保存 hidden state、上一帧动作和 episode mask。
        self.hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            args.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        # ROS 发布器：cmd_vel 控制机器人，debug_obs_topic 用作推理心跳。
        self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
        self.debug_obs_pub = rospy.Publisher(args.debug_obs_topic, Header, queue_size=10)

        # ROS 订阅器：极坐标目标和深度图。
        self.polar_sub = rospy.Subscriber(
            args.polar_topic, PointStamped, self._polar_cb, queue_size=20
        )
        self.depth_sub = rospy.Subscriber(args.depth_topic, Image, self._cb_depth, queue_size=10)

        # Falcon 输出 Discrete(4)，这里映射成 ROS Twist 速度指令。
        self.action_to_cmd = {
            0: (0.0, 0.0),  # stop
            1: (args.forward_speed, 0.0),  # forward
            2: (0.0, args.turn_speed),  # turn left
            3: (0.0, -args.turn_speed),  # turn right
        }

        if self.debug_depth:
            os.makedirs(self.debug_depth_dump_dir, exist_ok=True)
            rospy.loginfo("Depth debug dump dir: %s", self.debug_depth_dump_dir)
        if self.replay_dump_enabled:
            os.makedirs(self.replay_dump_dir, exist_ok=True)
            rospy.loginfo("Policy replay dump dir: %s", self.replay_dump_dir)

        rospy.loginfo("Falcon ROS bridge started.")
        rospy.loginfo("Subscribe: %s, %s", args.depth_topic, args.polar_topic)
        rospy.loginfo("Publish:   %s", args.cmd_vel_topic)
        # Watchdog 定时检查输入是否超时，超时就发布停车命令。
        self.watchdog = rospy.Timer(rospy.Duration(0.05), self._watchdog_cb)

    def _build_policy(
        self,
        checkpoint_path: str,
        hidden_size: int,
        num_recurrent_layers: int,
        backbone: str,
        rnn_type: str,
    ):
        # Build observation/action spaces that match the training setup.
        # 这里不连接真实 gym 环境，只构造 policy 初始化所需的空间描述。
        spaces = {
            self.goal_key: Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
            self.depth_key: Box(
                low=0.0,
                high=1.0,
                shape=(self.resolution, self.resolution, 1),
                dtype=np.float32,
            ),
        }

        observation_space = SpaceDict(spaces)
        action_space = Discrete(4)

        policy = PointNavResNetPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            normalize_visual_inputs=False,
        ).to(self.device)

        # 加载 checkpoint，并允许非严格匹配，方便兼容不同训练保存格式。
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt = _select_agent0_checkpoint_state_dict(ckpt)
        policy_sd = _extract_actor_critic_state_dict(ckpt)
        missing, unexpected = policy.load_state_dict(policy_sd, strict=False)

        rospy.logwarn("Checkpoint loaded with strict=False.")
        rospy.logwarn("Missing keys: %d, Unexpected keys: %d", len(missing), len(unexpected))
        if self.require_strict_ckpt and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                "Checkpoint key mismatch: missing={} unexpected={}".format(
                    len(missing), len(unexpected)
                )
            )
        return policy

    @staticmethod
    def _ros_image_to_numpy(msg: Image) -> np.ndarray:
        # Convert common ROS Image encodings to numpy without cv_bridge.
        if msg.encoding == "16UC1":
            row_bytes = msg.width * 2
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            arr = raw[:, :row_bytes].copy().view(np.uint16).reshape(msg.height, msg.width)
        elif msg.encoding == "32FC1":
            row_bytes = msg.width * 4
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            arr = raw[:, :row_bytes].copy().view(np.float32).reshape(msg.height, msg.width)
            print("notice the output depth type is 32FC1")
        # elif msg.encoding in ("rgb8", "bgr8"):
        #     row_bytes = msg.width * 3
        #     raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        #     arr = raw[:, :row_bytes].copy().reshape(msg.height, msg.width, 3)
        #     print("notice the input th color img now")
        else:
            raise ValueError("Unsupported image encoding: {}".format(msg.encoding))

        if msg.is_bigendian:
            arr = arr.byteswap().newbyteorder()
        return arr

    @staticmethod
    def _depth_stats(arr: np.ndarray) -> Dict[str, float]:
        # 为调试日志统计深度数组的有效比例、范围、分位数和 0 值比例。
        arr_f = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr_f)
        valid = arr_f[finite]
        total = float(arr_f.size)
        if valid.size == 0:
            return {
                "valid_ratio": 0.0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "zero_ratio": 0.0,
            }

        return {
            "valid_ratio": float(valid.size) / total,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "p50": float(np.percentile(valid, 50)),
            "p95": float(np.percentile(valid, 95)),
            "zero_ratio": float(np.mean(valid == 0.0)),
        }

    @staticmethod
    def _make_depth_preview(
        arr: np.ndarray, clip_max: Optional[float] = None
    ) -> np.ndarray:
        # 把深度数组归一化并上色，方便保存成 png 直观看输入质量。
        arr_f = arr.astype(np.float32, copy=False)
        if arr_f.ndim == 3 and arr_f.shape[-1] == 1:
            arr_f = arr_f[..., 0]

        finite = np.isfinite(arr_f)
        positive = finite & (arr_f > 0.0)
        if np.any(positive):
            valid = arr_f[positive]
        elif np.any(finite):
            valid = arr_f[finite]
        else:
            valid = np.array([0.0], dtype=np.float32)

        lo = float(np.min(valid))
        hi = float(np.percentile(valid, 99))
        if clip_max is not None:
            hi = min(hi, float(clip_max))
        if hi <= lo:
            hi = lo + 1e-6

        vis = np.clip(arr_f, lo, hi)
        vis = (vis - lo) / (hi - lo)
        vis = np.nan_to_num(vis, nan=0.0, posinf=1.0, neginf=0.0)
        vis_u8 = np.clip(vis * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)

    @staticmethod
    def _center_crop_to_square(arr: np.ndarray) -> np.ndarray:
        # Falcon 训练时使用方形输入，这里从中心裁成正方形再缩放。
        h, w = arr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return arr[y0 : y0 + side, x0 : x0 + side]

    @staticmethod
    def _replace_zero_depth_with_ten(arr: np.ndarray) -> np.ndarray:
        # 某些相机用 0 表示无效深度，这里当作远距离处理，避免被误认为障碍物。
        out = arr.copy()
        out[out == 0.0] = 10.0
        return out

    def _depth_msg_to_norm_depth(
        self, depth_msg: Image
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        # 将 ROS Image 深度消息转换成 Falcon 需要的 [H, W, 1] float32 归一化输入。

        # debug 字典记录每一步的形状、类型和统计信息，供日志和 dump 文件使用。
        debug = {
            "encoding": depth_msg.encoding,
            "raw_shape": None,
            "raw_dtype": None,
            "raw_unit": "m",
            "raw_stats": None,
            "depth_m_stats": None,
            "crop_shape": None,
            "crop_stats": None,
            "norm_shape": None,
            "norm_dtype": None,
            "norm_stats": None,
        }

        # 16UC1 通常是毫米；其他深度输入按当前相机流约定转成米。
        if depth_msg.encoding == "16UC1":
            depth_u16 = self._ros_image_to_numpy(depth_msg)
            raw_depth = depth_u16
            debug["raw_shape"] = tuple(depth_u16.shape)
            debug["raw_dtype"] = str(depth_u16.dtype)
            debug["raw_unit"] = "mm"
            debug["raw_stats"] = self._depth_stats(depth_u16)
            depth_m = depth_u16.astype(np.float32) * 0.001
        else:
            depth_f32 = self._ros_image_to_numpy(depth_msg)
            raw_depth = depth_f32
            debug["raw_shape"] = tuple(depth_f32.shape)
            debug["raw_dtype"] = str(depth_f32.dtype)
            debug["raw_unit"] = "m"
            debug["raw_stats"] = self._depth_stats(depth_f32)
            depth_m = depth_f32.astype(np.float32)*0.001

        # 清理 NaN/Inf，裁剪到最大深度，并把无效 0 深度替换成远距离值。
        depth_m = np.nan_to_num(depth_m, nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        depth_m = self._replace_zero_depth_with_ten(depth_m)
        debug["depth_m_stats"] = self._depth_stats(depth_m)
        # 中心裁剪成方形，缩放到 policy 训练分辨率，再归一化到 [0, 1]。
        depth_m = self._center_crop_to_square(depth_m)
        debug["crop_shape"] = tuple(depth_m.shape)
        debug["crop_stats"] = self._depth_stats(depth_m)
        depth_norm = depth_m / self.max_depth_m
        depth_norm = cv2.resize(depth_norm, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        # 统一成 float32，并给二维深度图增加通道维，得到模型期望的 (H, W, 1)。
        depth_norm = np.expand_dims(depth_norm.astype(np.float32), axis=-1)
        debug["norm_shape"] = tuple(depth_norm.shape)
        debug["norm_dtype"] = str(depth_norm.dtype)
        debug["norm_stats"] = self._depth_stats(depth_norm)

        self._maybe_dump_depth_sample_once(
            raw_depth=raw_depth,
            depth_m=depth_m,
            depth_norm=depth_norm,
            depth_debug=debug,
        )
        return depth_norm, debug

    def _maybe_dump_depth_sample_once(
        self,
        raw_depth: np.ndarray,
        depth_m: np.ndarray,
        depth_norm: np.ndarray,
        depth_debug: Dict[str, object],
    ):
        """dump a frame of depth info to files for debug"""
        # 只保存第一帧，避免调试模式下持续写入大量深度文件。
        if not self.debug_depth or self._depth_sample_saved:
            return

    
        stamp_ns = int(rospy.Time.now().to_nsec())
        prefix = "depth_sample_{}".format(stamp_ns)
        out_dir = self.debug_depth_dump_dir
        try:
            raw_npy = os.path.join(out_dir, prefix + "_raw.npy")
            raw_csv = os.path.join(out_dir, prefix + "_raw.csv")
            meter_npy = os.path.join(out_dir, prefix + "_meter.npy")
            meter_csv = os.path.join(out_dir, prefix + "_meter.csv")
            norm_npy = os.path.join(out_dir, prefix + "_norm.npy")
            norm_csv = os.path.join(out_dir, prefix + "_norm.csv")
            raw_png = os.path.join(out_dir, prefix + "_raw_preview.png")
            meter_png = os.path.join(out_dir, prefix + "_meter_preview.png")
            norm_png = os.path.join(out_dir, prefix + "_norm_preview.png")
            meta_json = os.path.join(out_dir, prefix + "_meta.json")

            # 同时保存 npy/csv 原始数值和 png 预览图，便于离线检查深度处理链路。
            np.save(raw_npy, raw_depth)
            np.savetxt(raw_csv, raw_depth, delimiter=",", fmt="%.6f")
            np.save(meter_npy, depth_m)
            np.savetxt(meter_csv, depth_m, delimiter=",", fmt="%.6f")
            np.save(norm_npy, depth_norm)
            np.savetxt(norm_csv, depth_norm[..., 0], delimiter=",", fmt="%.6f")
            cv2.imwrite(
                raw_png,
                self._make_depth_preview(
                    raw_depth,
                    clip_max=self.max_depth_m * 1000.0
                    if depth_debug["raw_unit"] == "mm"
                    else self.max_depth_m,
                ),
            )
            cv2.imwrite(
                meter_png,
                self._make_depth_preview(depth_m, clip_max=self.max_depth_m),
            )
            cv2.imwrite(
                norm_png,
                self._make_depth_preview(depth_norm, clip_max=1.0),
            )

            
            meta = {
                "prefix": prefix,
                "max_depth_m": float(self.max_depth_m),
                "resolution": int(self.resolution),
                "raw_unit": depth_debug["raw_unit"],
                "raw_shape": depth_debug["raw_shape"],
                "crop_shape": depth_debug["crop_shape"],
                "norm_shape": depth_debug["norm_shape"],
                "raw_dtype": depth_debug["raw_dtype"],
                "norm_dtype": depth_debug["norm_dtype"],
                "raw_stats": depth_debug["raw_stats"],
                "depth_m_stats": depth_debug["depth_m_stats"],
                "crop_stats": depth_debug["crop_stats"],
                "norm_stats": depth_debug["norm_stats"],
                "raw_preview_png": raw_png,
                "meter_preview_png": meter_png,
                "norm_preview_png": norm_png,
            }
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            self._depth_sample_saved = True
            rospy.loginfo(
                "[DBG_DEPTH_DUMP] saved one depth sample to %s (prefix=%s, previews=raw/meter/norm png)",
                out_dir,
                prefix,
            )
        except Exception as e:
            rospy.logerr_throttle(1.0, "Depth sample dump failed: %s", str(e))

    def _build_obs(
        self, depth_msg: Image, polar_msg: PointStamped
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        # Polar convention from sensor/polar_distance.py: x=r, y=theta.
        # 将极坐标目标和深度图打包成 Falcon policy 一次推理需要的 observation。
        r = np.float32(polar_msg.point.x)
        theta = np.float32(float(polar_msg.point.y))
        depth_norm, depth_debug = self._depth_msg_to_norm_depth(depth_msg)

        obs = {
            self.depth_key: depth_norm,
            self.goal_key: np.array([r, theta], dtype=np.float32),
        }
        return obs, depth_debug

    def _infer_action(
        self, obs: Dict[str, np.ndarray]
    ) -> Tuple[int, Optional[np.ndarray], Dict[str, np.ndarray]]:
        # Recurrent policy inference:
        # hidden_states/prev_actions/not_done_masks are carried across timesteps.
        batch = batch_obs([obs], device=self.device)
        with torch.no_grad():
            recurrent_input = {
                "hidden_in": self.hidden_states.detach().cpu().numpy(),
                "prev_action_in": self.prev_actions.detach().cpu().numpy(),
                "not_done_mask_in": self.not_done_masks.detach().cpu().numpy(),
            }
            features, next_hidden_states, _ = self.actor_critic.net(
                batch,
                self.hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )
            distribution = self.actor_critic.action_distribution(features)
            if self.deterministic:
                if self.actor_critic.action_distribution_type == "categorical":
                    actions = distribution.mode()
                elif self.actor_critic.action_distribution_type == "gaussian":
                    actions = distribution.mean
                else:
                    actions = distribution.sample()
            else:
                actions = distribution.sample()

            self.hidden_states = next_hidden_states
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)

            probs = None
            if self.actor_critic.action_distribution_type == "categorical":
                probs = distribution.probs[0].detach().cpu().numpy()

        return int(actions[0][0].item()), probs, recurrent_input

    @staticmethod
    def _fmt_stats(stats: Dict[str, float]) -> str:
        # 将深度统计结果格式化成一行日志文本。
        return (
            "valid={:.1f}% min={:.3f} max={:.3f} mean={:.3f} p50={:.3f} p95={:.3f} zero={:.1f}%".format(
                100.0 * stats["valid_ratio"],
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["p50"],
                stats["p95"],
                100.0 * stats["zero_ratio"],
            )
        )

    def _fmt_action_probs(self, probs: Optional[np.ndarray]) -> str:
        # 将 categorical action 概率转成人类可读的动作名和概率。
        if probs is None:
            return "N/A(non-categorical)"
        labels = ["stop", "forward", "left", "right"]
        items = []
        for i, p in enumerate(probs.tolist()):
            name = labels[i] if i < len(labels) else "a{}".format(i)
            items.append("{}:{:.3f}".format(name, float(p)))
        return " ".join(items)

    def _debug_print_once(
        self,
        obs: Dict[str, np.ndarray],
        act_id: int,
        theta_in: float,
        probs: Optional[np.ndarray],
        depth_debug: Dict[str, object],
    ):
        # 打印一次推理的关键输入、输出和深度处理统计，用于检查映射是否正确。
        g = obs[self.goal_key]
        d = obs[self.depth_key]
        lin, ang = self.action_to_cmd.get(act_id, (0.0, 0.0))

        if self.debug_depth:
            rospy.loginfo(
                "[DBG_DEPTH] ros(enc={}, raw={} {}, unit={}) raw_stats=[{}] depth_m_stats=[{}] "
                "crop(shape={}, stats=[{}]) "
                "falcon_expected(shape=({}, {}, 1), dtype=float32, norm=[0,1]) falcon_input(actual={} {}) norm_stats=[{}]".format(
                    depth_debug["encoding"],
                    depth_debug["raw_shape"],
                    depth_debug["raw_dtype"],
                    depth_debug["raw_unit"],
                    self._fmt_stats(depth_debug["raw_stats"]),
                    self._fmt_stats(depth_debug["depth_m_stats"]),
                    depth_debug["crop_shape"],
                    self._fmt_stats(depth_debug["crop_stats"]),
                    self.resolution,
                    self.resolution,
                    depth_debug["norm_shape"],
                    depth_debug["norm_dtype"],
                    self._fmt_stats(depth_debug["norm_stats"]),
                )
            )

        rospy.loginfo(
            "[DBG_ACT] goal[r,theta]=[{:.3f}, {:.3f}] input_theta={:.3f}(pass-through) depth_shape={} depth[min,max]=[{:.3f},{:.3f}] "
            "act_id={} cmd=({:.3f},{:.3f}) probs=[{}]".format(
                float(g[0]), float(g[1]), float(theta_in),
                tuple(d.shape), float(d.min()), float(d.max()),
                int(act_id), float(lin), float(ang),
                self._fmt_action_probs(probs),
            )
        )

    def _maybe_dump_replay_sample(
        self,
        obs: Dict[str, np.ndarray],
        act_id: int,
        probs: Optional[np.ndarray],
        recurrent_input: Dict[str, np.ndarray],
        depth_debug: Dict[str, object],
        depth_msg: Image,
        polar_msg: PointStamped,
    ) -> bool:
        if not self.replay_dump_enabled:
            return False
        if self.replay_dump_limit >= 0 and self._replay_dump_count >= self.replay_dump_limit:
            self._replay_dump_limit_reached = True
            return True

        stamp_ns = int(rospy.Time.now().to_nsec())
        prefix = "bridge_policy_replay_{}".format(stamp_ns)
        npz_path = os.path.abspath(os.path.join(self.replay_dump_dir, prefix + ".npz"))
        meta_path = os.path.abspath(os.path.join(self.replay_dump_dir, prefix + ".json"))
        lin, ang = self.action_to_cmd.get(act_id, (0.0, 0.0))

        try:
            np.savez_compressed(
                npz_path,
                depth=obs[self.depth_key],
                depth_meter=obs[self.depth_key] * np.float32(self.max_depth_m),
                goal=obs[self.goal_key],
                hidden_in=recurrent_input["hidden_in"],
                prev_action_in=recurrent_input["prev_action_in"],
                not_done_mask_in=recurrent_input["not_done_mask_in"],
                action=np.array([act_id], dtype=np.int64),
                probs=np.array([] if probs is None else probs, dtype=np.float32),
            )
            meta = {
                "prefix": prefix,
                "obs_npz": npz_path,
                "depth_key": self.depth_key,
                "goal_key": self.goal_key,
                "resolution": int(self.resolution),
                "max_depth_m": float(self.max_depth_m),
                "deterministic": bool(self.deterministic),
                "action_id": int(act_id),
                "cmd_linear_x": float(lin),
                "cmd_angular_z": float(ang),
                "action_probs": None if probs is None else probs.tolist(),
                "depth_msg_stamp": depth_msg.header.stamp.to_sec(),
                "polar_msg_stamp": polar_msg.header.stamp.to_sec(),
                "polar_r": float(polar_msg.point.x),
                "polar_theta": float(polar_msg.point.y),
                "depth_debug": depth_debug,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self._replay_dump_count += 1
            rospy.loginfo("[REPLAY_DUMP] saved %s", meta_path)
            if self.replay_dump_limit >= 0 and self._replay_dump_count >= self.replay_dump_limit:
                self._replay_dump_limit_reached = True
                return True
        except Exception as e:
            rospy.logerr_throttle(1.0, "Policy replay dump failed: %s", str(e))
        return False


    def _publish_cmd(self, action_id: int):
        # Action id -> (linear x, angular z).
        lin, ang = self.action_to_cmd.get(action_id, (0.0, 0.0))
        tw = Twist()
        tw.linear.x = lin
        tw.angular.z = ang
        self.cmd_pub.publish(tw)

    def _publish_stop(self):
        # 发布全 0 Twist，作为缺输入、异常和 watchdog 超时的安全停车动作。
        tw = Twist()
        self.cmd_pub.publish(tw)

    def _polar_cb(self, polar_msg: PointStamped):
        # 极坐标目标回调只负责缓存，真正推理由深度图回调驱动。
        self.latest_polar_msg = polar_msg
        self.polar_buffer.append(polar_msg)

    def _pick_polar_for_stamp(self, target_stamp: rospy.Time):
        # Pick the temporally closest polar message to current image timestamp.
        if len(self.polar_buffer) == 0:
            return None

        # If image has no timestamp, fallback to latest.
        if target_stamp == rospy.Time():
            return self.polar_buffer[-1]

        best = None
        best_dt = None
        for msg in self.polar_buffer:
            if msg.header.stamp == rospy.Time():
                continue
            dt = abs((target_stamp - msg.header.stamp).to_sec())
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = msg

        if best is None:
            return self.polar_buffer[-1]
        if best_dt is not None and best_dt > self.max_polar_age_sec:
            return None
        return best

    def _emit_heartbeat(self):
        # Lightweight debug pulse indicating inference loop is alive.
        hdr = Header()
        hdr.stamp = rospy.Time.now()
        hdr.frame_id = "falcon_obs_ok"
        self.debug_obs_pub.publish(hdr)

    def _process_one(self, depth_msg: Image, polar_msg: PointStamped = None):
        # Single end-to-end control step: select goal -> build obs -> infer -> publish cmd.
        if polar_msg is None:
            polar_msg = self._pick_polar_for_stamp(depth_msg.header.stamp)
        if polar_msg is None:
            rospy.logwarn_throttle(2.0, "No polar message received yet on polar topic.")
            self._publish_stop()
            return
        try:
            theta_in = float(polar_msg.point.y)
            obs, depth_debug = self._build_obs(depth_msg=depth_msg, polar_msg=polar_msg)
            act_id, probs, recurrent_input = self._infer_action(obs)
            if self.debug_mapping or self.debug_depth:
                self._debug_print_once(obs, act_id, theta_in, probs, depth_debug)
            replay_limit_reached = self._maybe_dump_replay_sample(
                obs=obs,
                act_id=act_id,
                probs=probs,
                recurrent_input=recurrent_input,
                depth_debug=depth_debug,
                depth_msg=depth_msg,
                polar_msg=polar_msg,
            )
            if replay_limit_reached:
                self._publish_stop()
                self.last_obs_time = rospy.Time.now()
                self._emit_heartbeat()
                rospy.loginfo(
                    "[REPLAY_DUMP] reached limit %d, stopping bridge.",
                    self.replay_dump_limit,
                )
                rospy.signal_shutdown("Replay dump limit reached.")
                return
            self._publish_cmd(act_id)
            self.last_obs_time = rospy.Time.now()
            self._emit_heartbeat()
        except Exception as e:
            self._publish_stop()
            rospy.logerr_throttle(1.0, "Falcon ROS bridge callback failed: %s", str(e))

    def _cb_depth(self, depth_msg: Image):
        # 每收到一帧深度图，就尝试用时间上最近的目标消息进行一次控制推理。
        self._process_one(depth_msg=depth_msg)

    def _watchdog_cb(self, _event):
        # Fail-safe: stop robot if no successful inference for too long.
        if self.last_obs_time == rospy.Time(0):
            return
        dt = (rospy.Time.now() - self.last_obs_time).to_sec()
        if dt > self.data_timeout_sec:
            self._publish_stop()
            rospy.logwarn_throttle(1.0, "Input timeout %.3fs > %.3fs, publish stop.", dt, self.data_timeout_sec)

    def spin(self):
        rospy.spin()


def parse_args():
    # 命令行参数用于适配不同 topic、模型结构、速度标定和调试需求。
    p = argparse.ArgumentParser(description="ROS Depth+Polar -> Falcon -> cmd_vel bridge")
    p.add_argument("--checkpoint", type=str, required=True)
    # Backward-compatibility flags kept for old launch scripts.
    # They are ignored because this bridge is now fixed to depth + polar topic.
    p.add_argument("--input_type", type=str, default="depth", choices=["depth", "rgbd"])
    p.add_argument("--polar_source", type=str, default="topic", choices=["topic", "detections"])

    p.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--polar_topic", type=str, default="/tag_polar")
    p.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    p.add_argument("--debug_obs_topic", type=str, default="/falcon/obs_heartbeat")

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_depth_m", type=float, default=10.0)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_recurrent_layers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--rnn_type", type=str, default="LSTM")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--strict_checkpoint", action="store_true")
    p.add_argument("--debug_mapping", action="store_true")
    p.add_argument("--debug_depth", action="store_true")
    p.add_argument("--debug_depth_dump_dir", type=str, default="./test_modules/test_results/bridge_depth_samples")
    p.add_argument("--replay_dump_enabled", action="store_true")
    p.add_argument("--replay_dump_dir", type=str, default="./test_modules/test_results/bridge_policy_replay")
    p.add_argument("--replay_dump_limit", type=int, default=20)
    p.add_argument("--data_timeout_sec", type=float, default=0.3)
    p.add_argument("--max_polar_age_sec", type=float, default=0.12)
    p.add_argument("--polar_buffer_size", type=int, default=100)

    # Default to non-agent-prefixed keys used by PointNavResNetPolicy sensor handling.
    p.add_argument("--depth_obs_key", type=str, default="articulated_agent_jaw_depth")
    p.add_argument("--goal_obs_key", type=str, default="pointgoal_with_gps_compass")

    p.add_argument("--forward_speed", type=float, default=0.3)
    p.add_argument("--turn_speed", type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    node = FalconRosBridge(args)
    node.spin()
