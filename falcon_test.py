#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import message_filters

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs


def _extract_actor_critic_state_dict(ckpt_obj: Dict) -> Dict[str, torch.Tensor]:
    """
    Accept different checkpoint layouts and return state_dict keys
    compatible with PointNavResNetPolicy.
    """
    src = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(src, dict):
        raise RuntimeError("Unsupported checkpoint format: state_dict is not a dict.")

    out = {}
    for k, v in src.items():
        if not isinstance(k, str):
            continue
        if "actor_critic." in k:
            kk = k.split("actor_critic.", 1)[1]
            out[kk] = v
        elif k.startswith("net.") or k.startswith("action_distribution.") or k.startswith("critic."):
            out[k] = v
    if len(out) == 0:
        # Fallback: assume keys are already policy keys.
        out = src
    return out


class FalconRosBridge(object):
    def __init__(self, args):
        rospy.init_node("falcon_ros_bridge", anonymous=False)
        self.bridge = CvBridge()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo("Falcon device: %s", str(self.device))

        self.resolution = args.resolution
        self.max_depth_m = args.max_depth_m
        self.use_rgb = args.input_type == "rgbd"
        self.deterministic = args.deterministic

        # Policy obs keys should match your social_nav_v2 config.
        self.depth_key = args.depth_obs_key
        self.goal_key = args.goal_obs_key
        self.rgb_key = args.rgb_obs_key

        self.actor_critic = self._build_policy(
            checkpoint_path=args.checkpoint,
            hidden_size=args.hidden_size,
            num_recurrent_layers=args.num_recurrent_layers,
            backbone=args.backbone,
            rnn_type=args.rnn_type,
        )
        self.actor_critic.eval()

        self.hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            args.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
        self.debug_obs_pub = rospy.Publisher(args.debug_obs_topic, Header, queue_size=10)

        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        polar_sub = message_filters.Subscriber(args.polar_topic, PointStamped)
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, polar_sub],
            queue_size=10,
            slop=args.sync_slop_sec,
        )
        sync.registerCallback(self._cb)

        self.action_to_cmd = {
            0: (0.0, 0.0),  # stop
            1: (args.forward_speed, 0.0),  # forward
            2: (0.0, args.turn_speed),  # turn left
            3: (0.0, -args.turn_speed),  # turn right
        }

        rospy.loginfo("Falcon ROS bridge started.")
        rospy.loginfo("Subscribe: %s, %s, %s", args.color_topic, args.depth_topic, args.polar_topic)
        rospy.loginfo("Publish:   %s", args.cmd_vel_topic)

    def _build_policy(
        self,
        checkpoint_path: str,
        hidden_size: int,
        num_recurrent_layers: int,
        backbone: str,
        rnn_type: str,
    ):
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
        if self.use_rgb:
            spaces[self.rgb_key] = Box(
                low=0,
                high=255,
                shape=(self.resolution, self.resolution, 3),
                dtype=np.uint8,
            )

        observation_space = SpaceDict(spaces)
        action_space = Discrete(4)

        policy = PointNavResNetPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            normalize_visual_inputs=self.use_rgb,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        policy_sd = _extract_actor_critic_state_dict(ckpt)
        missing, unexpected = policy.load_state_dict(policy_sd, strict=False)

        rospy.logwarn("Checkpoint loaded with strict=False.")
        rospy.logwarn("Missing keys: %d, Unexpected keys: %d", len(missing), len(unexpected))
        return policy

    def _depth_msg_to_norm_depth(self, depth_msg: Image) -> np.ndarray:
        if depth_msg.encoding == "16UC1":
            depth_u16 = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            depth_m = depth_u16.astype(np.float32) * 0.001
        else:
            depth_f32 = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            depth_m = depth_f32.astype(np.float32)

        depth_m = np.nan_to_num(depth_m, nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        depth_norm = depth_m / self.max_depth_m
        depth_norm = cv2.resize(depth_norm, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        depth_norm = np.expand_dims(depth_norm.astype(np.float32), axis=-1)
        return depth_norm

    def _color_msg_to_rgb(self, color_msg: Image) -> np.ndarray:
        if color_msg.encoding == "rgb8":
            rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        else:
            bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        return rgb.astype(np.uint8)

    def _build_obs(self, color_msg: Image, depth_msg: Image, polar_msg: PointStamped) -> Dict[str, np.ndarray]:
        # polar msg convention from sensor/polar_distance.py:
        # x=r, y=theta
        r = np.float32(polar_msg.point.x)
        theta = np.float32(polar_msg.point.y)

        obs = {
            self.depth_key: self._depth_msg_to_norm_depth(depth_msg),
            self.goal_key: np.array([r, theta], dtype=np.float32),
        }
        if self.use_rgb:
            obs[self.rgb_key] = self._color_msg_to_rgb(color_msg)
        return obs

    def _infer_action(self, obs: Dict[str, np.ndarray]) -> int:
        batch = batch_obs([obs], device=self.device)
        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=self.deterministic,
            )
            self.hidden_states = action_data.rnn_hidden_states
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action_data.actions)
        return int(action_data.env_actions[0][0].item())

    def _publish_cmd(self, action_id: int):
        lin, ang = self.action_to_cmd.get(action_id, (0.0, 0.0))
        tw = Twist()
        tw.linear.x = lin
        tw.angular.z = ang
        self.cmd_pub.publish(tw)

    def _cb(self, color_msg: Image, depth_msg: Image, polar_msg: PointStamped):
        try:
            obs = self._build_obs(color_msg, depth_msg, polar_msg)
            act_id = self._infer_action(obs)
            self._publish_cmd(act_id)

            hdr = Header()
            hdr.stamp = rospy.Time.now()
            hdr.frame_id = "falcon_obs_ok"
            self.debug_obs_pub.publish(hdr)
        except Exception as e:
            rospy.logerr_throttle(1.0, "Falcon ROS bridge callback failed: %s", str(e))

    def spin(self):
        rospy.spin()


def parse_args():
    p = argparse.ArgumentParser(description="ROS RGBD+Polar -> Falcon -> cmd_vel bridge")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input_type", type=str, default="depth", choices=["depth", "rgbd"])

    p.add_argument("--color_topic", type=str, default="/camera/color/image_raw")
    p.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--polar_topic", type=str, default="/tag_polar")
    p.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    p.add_argument("--debug_obs_topic", type=str, default="/falcon/obs_heartbeat")

    p.add_argument("--sync_slop_sec", type=float, default=0.08)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_depth_m", type=float, default=10.0)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_recurrent_layers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--rnn_type", type=str, default="LSTM")
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--depth_obs_key", type=str, default="agent_0_articulated_agent_jaw_depth")
    p.add_argument("--goal_obs_key", type=str, default="agent_0_pointgoal_with_gps_compass")
    p.add_argument("--rgb_obs_key", type=str, default="agent_0_articulated_agent_jaw_rgb")

    p.add_argument("--forward_speed", type=float, default=0.2)
    p.add_argument("--turn_speed", type=float, default=0.8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    node = FalconRosBridge(args)
    node.spin()
