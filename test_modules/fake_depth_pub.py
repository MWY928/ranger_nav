#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish fake depth images for falcon_ros_bridge testing.

Default output mimics common RealSense aligned depth stream:
- topic: /camera/aligned_depth_to_color/image_raw
- msg type: sensor_msgs/Image
- encoding: 16UC1 (millimeters)
"""

import argparse
import math

import numpy as np
import rospy
from sensor_msgs.msg import Image


def parse_args():
    p = argparse.ArgumentParser(description="Publish fake depth image stream.")
    p.add_argument("--topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--frame_id", type=str, default="camera_color_optical_frame")
    p.add_argument("--rate", type=float, default=20.0)

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--encoding", type=str, default="16UC1", choices=["16UC1", "32FC1"])

    p.add_argument("--mode", type=str, default="fixed", choices=["fixed", "sweep", "random"])
    p.add_argument("--depth_m", type=float, default=2.0, help="Used by fixed mode.")
    p.add_argument("--min_depth_m", type=float, default=0.2, help="Used by sweep/random mode.")
    p.add_argument("--max_depth_m", type=float, default=8.0, help="Used by sweep/random mode.")
    p.add_argument("--sweep_hz", type=float, default=0.2, help="Used by sweep mode.")
    p.add_argument("--seed", type=int, default=0, help="Used by random mode.")
    return p.parse_args()


def _safe_depth(d: float) -> float:
    return max(0.0, float(d))


def _build_depth_value_m(args, t_sec: float, rng: np.random.Generator) -> float:
    if args.mode == "fixed":
        return _safe_depth(args.depth_m)

    lo = _safe_depth(min(args.min_depth_m, args.max_depth_m))
    hi = _safe_depth(max(args.min_depth_m, args.max_depth_m))

    if args.mode == "sweep":
        mid = 0.5 * (lo + hi)
        amp = 0.5 * (hi - lo)
        return mid + amp * math.sin(2.0 * math.pi * args.sweep_hz * t_sec)

    return float(rng.uniform(lo, hi))


def _to_image_msg(depth_m: float, args, stamp: rospy.Time) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = args.frame_id
    msg.width = int(args.width)
    msg.height = int(args.height)
    msg.is_bigendian = False
    msg.encoding = args.encoding

    if args.encoding == "16UC1":
        depth_mm = int(round(max(0.0, depth_m) * 1000.0))
        depth_mm = max(0, min(depth_mm, 65535))
        arr = np.full((args.height, args.width), depth_mm, dtype=np.uint16)
        msg.step = args.width * 2
    else:
        arr = np.full((args.height, args.width), np.float32(max(0.0, depth_m)), dtype=np.float32)
        msg.step = args.width * 4

    msg.data = arr.tobytes()
    return msg


def main():
    args = parse_args()
    rospy.init_node("fake_depth_pub", anonymous=False)
    pub = rospy.Publisher(args.topic, Image, queue_size=5)
    rate_hz = max(0.1, args.rate)
    loop = rospy.Rate(rate_hz)
    t0 = rospy.Time.now().to_sec()
    rng = np.random.default_rng(args.seed)

    rospy.loginfo("fake_depth_pub started.")
    rospy.loginfo("Publish: %s @ %.2f Hz", args.topic, rate_hz)
    rospy.loginfo(
        "Image: %dx%d encoding=%s frame_id=%s",
        args.width,
        args.height,
        args.encoding,
        args.frame_id,
    )
    rospy.loginfo(
        "Mode=%s depth_m=%.3f range=[%.3f, %.3f] sweep_hz=%.3f",
        args.mode,
        args.depth_m,
        args.min_depth_m,
        args.max_depth_m,
        args.sweep_hz,
    )

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        dt = now.to_sec() - t0
        depth_m = _build_depth_value_m(args, dt, rng)
        msg = _to_image_msg(depth_m, args, now)
        pub.publish(msg)
        loop.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
