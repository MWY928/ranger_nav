#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish fake polar goals for Falcon bridge smoke testing.

PointStamped convention:
- point.x = r (meters)
- point.y = theta (radians)
- point.z = tag_id (optional debug id)
"""

import argparse
import math

import rospy
from geometry_msgs.msg import PointStamped


def parse_args():
    p = argparse.ArgumentParser(description="Publish fake /tag_polar goals for bridge testing.")
    p.add_argument("--topic", type=str, default="/tag_polar")
    p.add_argument("--frame_id", type=str, default="camera_color_optical_frame")
    p.add_argument("--rate", type=float, default=20.0)

    p.add_argument("--mode", type=str, default="fixed", choices=["fixed", "sweep"])
    p.add_argument("--r", type=float, default=2, help="Fixed/sweep center distance (m)")
    p.add_argument("--theta", type=float, default=3.14, help="Fixed heading (rad)")
    p.add_argument("--tag_id", type=float, default=999.0)

    p.add_argument("--sweep_amp", type=float, default=0.5, help="Sweep amplitude (rad)")
    p.add_argument("--sweep_hz", type=float, default=0.15, help="Sweep frequency (Hz)")
    return p.parse_args()


def main():
    args = parse_args()
    rospy.init_node("fake_polar_goal_pub", anonymous=False)
    pub = rospy.Publisher(args.topic, PointStamped, queue_size=20)

    rate_hz = max(0.1, args.rate)
    loop = rospy.Rate(rate_hz)
    t0 = rospy.Time.now().to_sec()

    rospy.loginfo("fake_polar_goal_pub started.")
    rospy.loginfo("Publish: %s @ %.2f Hz", args.topic, rate_hz)
    rospy.loginfo(
        "Mode: %s | r=%.3f | theta=%.3f | sweep_amp=%.3f | sweep_hz=%.3f",
        args.mode,
        args.r,
        args.theta,
        args.sweep_amp,
        args.sweep_hz,
    )

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        theta = args.theta
        if args.mode == "sweep":
            dt = now.to_sec() - t0
            theta = args.theta + args.sweep_amp * math.sin(2.0 * math.pi * args.sweep_hz * dt)

        msg = PointStamped()
        msg.header.stamp = now
        msg.header.frame_id = args.frame_id
        msg.point.x = float(args.r)
        msg.point.y = float(theta)
        msg.point.z = float(args.tag_id)
        pub.publish(msg)
        loop.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
