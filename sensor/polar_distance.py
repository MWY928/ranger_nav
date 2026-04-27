#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy

from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PointStamped


class TagToPolarNode(object):
    def __init__(self):
        rospy.init_node("tag_to_polar_node", anonymous=False)

        # -----------------------------
        # Parameters
        # -----------------------------
        self.detections_topic = rospy.get_param("~detections_topic", "/tag_detections")
        self.output_topic = rospy.get_param("~output_topic", "/tag_polar")
        self.target_tag_id = int(rospy.get_param("~target_tag_id", 0))
        self.use_first_detection = bool(rospy.get_param("~use_first_detection", False))

        # -----------------------------
        # Angle correction parameters
        # -----------------------------
        # theta_offset_rad:
        #   用来修正相机安装偏角 / tag 正前方测量偏差。
        #
        #   corrected_theta = raw_theta - theta_offset_rad
        #
        #   例如：
        #   如果 tag 物理上在正前方，但测出来 raw_theta = -0.1037 rad，
        #   那么设置：
        #       _theta_offset_rad:=-0.1037
        #   修正后：
        #       corrected_theta = -0.1037 - (-0.1037) = 0
        #
        # theta_deadband_rad:
        #   修正后的角度如果在 [-deadband, +deadband] 内，
        #   就直接输出 0。
        #
        #   例如：
        #       0.03 rad ≈ 1.72 deg
        #       0.05 rad ≈ 2.86 deg
        #       0.0873 rad ≈ 5 deg
        self.theta_offset_rad = float(rospy.get_param("~theta_offset_rad", -0.1))
        self.theta_deadband_rad = float(rospy.get_param("~theta_deadband_rad", 0.0))

        # 坐标解释方式：
        # 对 apriltag_ros 常见输出，pose 是 tag 在 camera frame 下的位姿。
        # 常见相机光学坐标系里：
        #   x: 向右
        #   y: 向下
        #   z: 向前
        #
        # 这里我们取二维平面为 (x, z)：
        #   forward = z
        #   lateral = x
        # 所以：
        #   r = sqrt(x^2 + z^2)
        #   theta = atan2(x, z)
        #
        # theta > 0 表示 tag 在相机前方偏右
        # theta < 0 表示 tag 在相机前方偏左

        self.pub = rospy.Publisher(self.output_topic, PointStamped, queue_size=10)
        self.sub = rospy.Subscriber(
            self.detections_topic,
            AprilTagDetectionArray,
            self.cb,
            queue_size=10
        )

        rospy.loginfo("tag_to_polar_node started.")
        rospy.loginfo("Subscribe: %s", self.detections_topic)
        rospy.loginfo("Publish:   %s", self.output_topic)
        rospy.loginfo(
            "target_tag_id = %d, use_first_detection = %s",
            self.target_tag_id,
            str(self.use_first_detection)
        )
        rospy.loginfo(
            "theta_offset_rad = %.4f rad, theta_deadband_rad = %.4f rad",
            self.theta_offset_rad,
            self.theta_deadband_rad
        )

    def cb(self, msg):
        if not msg.detections:
            return

        chosen_detection = None
        chosen_id = None

        if self.use_first_detection:
            det = msg.detections[0]
            if len(det.id) > 0:
                chosen_detection = det
                chosen_id = det.id[0]
        else:
            for det in msg.detections:
                if len(det.id) == 0:
                    continue

                # 单 tag 情况下一般 det.id[0] 就是标签 ID
                if det.id[0] == self.target_tag_id:
                    chosen_detection = det
                    chosen_id = det.id[0]
                    break

        if chosen_detection is None:
            return

        pose = chosen_detection.pose.pose.pose
        px = pose.position.x
        py = pose.position.y
        pz = pose.position.z

        # -----------------------------
        # 2D polar in camera x-z plane
        # -----------------------------
        r = math.sqrt(px * px + pz * pz)

        # 原始角度：
        # theta_raw > 0: tag 在相机前方偏右
        # theta_raw < 0: tag 在相机前方偏左
        theta_raw = math.atan2(px, pz)

        # -----------------------------
        # Offset correction
        # -----------------------------
        # 如果正前方测出来是 -0.1037 rad，
        # 设置 theta_offset_rad = -0.1037，
        # 则修正后 theta = 0。
        theta = theta_raw - self.theta_offset_rad

        # -----------------------------
        # Deadband / zero zone
        # -----------------------------
        # 修正后角度较小时，直接认为是正前方。
        if abs(theta) < self.theta_deadband_rad:
            theta = 0.0

        out = PointStamped()

        # 优先使用原始 detection array 的时间戳
        if msg.header.stamp != rospy.Time():
            out.header.stamp = msg.header.stamp
        else:
            out.header.stamp = rospy.Time.now()

        if msg.header.frame_id:
            out.header.frame_id = msg.header.frame_id
        else:
            out.header.frame_id = "camera_color_optical_frame"

        # 约定：
        # x = r
        # y = theta
        # z = tag_id
        out.point.x = r
        out.point.y = theta
        out.point.z = float(chosen_id)

        self.pub.publish(out)

        rospy.logdebug(
            "tag_id=%d, px=%.3f, py=%.3f, pz=%.3f -> "
            "r=%.3f, theta_raw=%.4f rad, theta_corrected=%.4f rad",
            chosen_id,
            px,
            py,
            pz,
            r,
            theta_raw,
            theta
        )


if __name__ == "__main__":
    try:
        TagToPolarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
