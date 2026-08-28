#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Publish Go2 SportModeState as a ROS1 nav_msgs/Odometry topic."""

import argparse
import math
import threading
import time

import rospy
from nav_msgs.msg import Odometry


def normalize_quaternion_wxyz(values):
    """Return a normalized Unitree [w, x, y, z] quaternion."""
    if len(values) < 4:
        raise ValueError("IMU quaternion has fewer than four elements")

    qw, qx, qy, qz = (float(values[index]) for index in range(4))
    if not all(math.isfinite(value) for value in (qw, qx, qy, qz)):
        raise ValueError("IMU quaternion contains a non-finite value")

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1.0e-6:
        raise ValueError("IMU quaternion norm is zero")
    return qw / norm, qx / norm, qy / norm, qz / norm


def quaternion_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class SportModeStateOdomBridge(object):
    def __init__(self, args, channel_subscriber_cls, sport_mode_state_cls):
        self.args = args
        self._lock = threading.Lock()
        self._last_state_monotonic = None
        self._last_publish_monotonic = None
        self._reported_timeout = False

        self.pub = rospy.Publisher(args.odom_topic, Odometry, queue_size=20)
        self.subscriber = channel_subscriber_cls(args.sdk_topic, sport_mode_state_cls)
        self.subscriber.Init(self.state_cb, args.queue_len)
        self.watchdog = rospy.Timer(
            rospy.Duration(1.0 / max(args.watchdog_rate_hz, 1.0)),
            self.watchdog_cb,
        )
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("Go2 SportModeState -> ROS1 odometry bridge started.")
        rospy.loginfo("SDK topic:       %s", args.sdk_topic)
        rospy.loginfo("ROS odom topic:  %s", args.odom_topic)
        rospy.loginfo("Frames:          %s -> %s", args.odom_frame, args.base_frame)
        rospy.loginfo("Max publish rate: %.1f Hz", args.publish_rate_hz)

    def state_cb(self, state):
        try:
            position = [float(state.position[index]) for index in range(3)]
            velocity_world = [float(state.velocity[index]) for index in range(3)]
            yaw_speed = float(state.yaw_speed)
            if not all(
                math.isfinite(value)
                for value in position + velocity_world + [yaw_speed]
            ):
                raise ValueError("position or velocity contains a non-finite value")

            qw, qx, qy, qz = normalize_quaternion_wxyz(
                state.imu_state.quaternion
            )
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            rospy.logwarn_throttle(1.0, "Invalid SportModeState ignored: %s", str(exc))
            return

        error_code = int(getattr(state, "error_code", 0))
        if error_code != 0:
            rospy.logwarn_throttle(
                1.0,
                "SportModeState error_code=%d; publishing finite pose anyway.",
                error_code,
            )

        state_monotonic = time.monotonic()
        min_publish_period = 1.0 / max(self.args.publish_rate_hz, 1.0)
        with self._lock:
            self._last_state_monotonic = state_monotonic
            recovered = self._reported_timeout
            self._reported_timeout = False
            should_publish = (
                self._last_publish_monotonic is None
                or state_monotonic - self._last_publish_monotonic
                >= min_publish_period
            )
            if should_publish:
                self._last_publish_monotonic = state_monotonic

        if recovered:
            rospy.loginfo("Go2 SportModeState stream recovered.")
        if not should_publish:
            return

        # Unitree documents position and velocity in its odometry frame. ROS
        # Odometry expresses twist in child_frame_id, so rotate planar velocity
        # into base_link while leaving the pose in the odometry frame.
        yaw = quaternion_to_yaw(qw, qx, qy, qz)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_body = cos_yaw * velocity_world[0] + sin_yaw * velocity_world[1]
        vy_body = -sin_yaw * velocity_world[0] + cos_yaw * velocity_world[1]

        out = Odometry()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.args.odom_frame
        out.child_frame_id = self.args.base_frame
        out.pose.pose.position.x = position[0]
        out.pose.pose.position.y = position[1]
        out.pose.pose.position.z = position[2]
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist.twist.linear.x = vx_body
        out.twist.twist.linear.y = vy_body
        out.twist.twist.linear.z = velocity_world[2]
        out.twist.twist.angular.z = yaw_speed
        self.pub.publish(out)

        rospy.logdebug(
            "Go2 odom x=%.3f y=%.3f yaw=%.3f vx=%.3f vy=%.3f wz=%.3f",
            position[0],
            position[1],
            yaw,
            vx_body,
            vy_body,
            yaw_speed,
        )

    def watchdog_cb(self, _event):
        with self._lock:
            last_state = self._last_state_monotonic
            already_reported = self._reported_timeout

        if last_state is None:
            age = None
            timed_out = (
                time.monotonic() - self.args.start_monotonic
                > self.args.state_timeout_sec
            )
        else:
            age = time.monotonic() - last_state
            timed_out = age > self.args.state_timeout_sec

        if timed_out and not already_reported:
            if age is None:
                rospy.logwarn(
                    "No Go2 SportModeState received within %.2f s.",
                    self.args.state_timeout_sec,
                )
            else:
                rospy.logwarn(
                    "Go2 SportModeState is stale (%.2f s); "
                    "no stale odometry is published.",
                    age,
                )
            with self._lock:
                self._reported_timeout = True

    def shutdown(self):
        try:
            self.subscriber.Close()
        except Exception as exc:  # SDK cleanup must not block ROS shutdown.
            rospy.logdebug("Closing Unitree state subscriber failed: %s", str(exc))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bridge Unitree Go2 rt/sportmodestate to ROS1 Odometry."
    )
    parser.add_argument("--network_interface", default="enxec9a0c1bc5be")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--sdk_topic", default="rt/sportmodestate")
    parser.add_argument("--odom_topic", default="/go2/sport_odom")
    parser.add_argument("--odom_frame", default="go2_odom")
    parser.add_argument("--base_frame", default="base_link")
    parser.add_argument("--queue_len", type=int, default=1)
    parser.add_argument("--state_timeout_sec", type=float, default=0.5)
    parser.add_argument("--publish_rate_hz", type=float, default=50.0)
    parser.add_argument("--watchdog_rate_hz", type=float, default=10.0)
    args = parser.parse_args(rospy.myargv()[1:])
    args.start_monotonic = time.monotonic()
    return args


def main():
    args = parse_args()
    rospy.init_node("go2_sport_mode_state_odom", anonymous=False)

    try:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    except ImportError as exc:
        rospy.logfatal("unitree_sdk2py import failed: %s", str(exc))
        raise SystemExit(2)

    try:
        ChannelFactoryInitialize(args.domain_id, args.network_interface or None)
    except Exception as exc:
        rospy.logfatal("Unitree DDS initialization failed: %s", str(exc))
        raise SystemExit(2)

    SportModeStateOdomBridge(args, ChannelSubscriber, SportModeState_)
    rospy.spin()


if __name__ == "__main__":
    main()
