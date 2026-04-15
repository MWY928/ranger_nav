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
        self.sub = rospy.Subscriber(self.detections_topic, AprilTagDetectionArray, self.cb, queue_size=10)

        rospy.loginfo("tag_to_polar_node started.")
        rospy.loginfo("Subscribe: %s", self.detections_topic)
        rospy.loginfo("Publish:   %s", self.output_topic)
        rospy.loginfo("target_tag_id = %d, use_first_detection = %s",
                      self.target_tag_id, str(self.use_first_detection))

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

        # 2D polar in camera x-z plane
        r = math.sqrt(px * px + pz * pz)
        theta = math.atan2(px, pz)

        out = PointStamped()
        out.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        out.header.frame_id = msg.header.frame_id if msg.header.frame_id else "camera_color_optical_frame"

        # 约定：
        # x = r
        # y = theta
        # z = tag_id
        out.point.x = r
        out.point.y = theta
        out.point.z = float(chosen_id)

        self.pub.publish(out)

        rospy.logdebug("tag_id=%d, px=%.3f, py=%.3f, pz=%.3f -> r=%.3f, theta=%.3f rad",
                       chosen_id, px, py, pz, r, theta)


if __name__ == "__main__":
    try:
        TagToPolarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass