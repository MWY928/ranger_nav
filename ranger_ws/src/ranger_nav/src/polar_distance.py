#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PointStamped


def get_bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class TagToPolarNode(object):
    """
    Convert local apriltag_ros detections into Falcon pointgoal-style polar input.

    Expected deployment for Go2:
    - Go2 Jetson publishes D435 image topics through realsense2_ros.
    - This machine runs apriltag_ros against those remote image topics.
    - This node subscribes to apriltag_ros /tag_detections.

    Output PointStamped:
    - point.x = r, meters after distance_offset
    - point.y = theta, radians
    - point.z = tag_id

    Angle convention:
    - theta > 0: tag is left of the camera optical axis, turn left
    - theta < 0: tag is right of the camera optical axis, turn right
    """

    def __init__(self):
        rospy.init_node("tag_to_polar_node", anonymous=False)

        self.detections_topic = rospy.get_param("~detections_topic", "/tag_detections")
        self.output_topic = rospy.get_param("~output_topic", "/tag_polar")
        self.output_frame_id = rospy.get_param("~output_frame_id", "base_link")

        self.target_tag_id = int(rospy.get_param("~target_tag_id", 0))
        self.use_first_detection = get_bool_param("~use_first_detection", False)

        self.theta_offset_rad = float(rospy.get_param("~theta_offset_rad", 0.0))
        self.enable_theta_offset = get_bool_param("~enable_theta_offset", True)
        self.theta_deadband_rad = float(rospy.get_param("~theta_deadband_rad", 0.0))
        self.enable_theta_deadband = get_bool_param("~enable_theta_deadband", True)

        self.distance_offset = float(rospy.get_param("~distance_offset", 0.6))
        self.min_distance = float(rospy.get_param("~min_distance", 0.0))

        self.pub = rospy.Publisher(self.output_topic, PointStamped, queue_size=10)
        self.sub = rospy.Subscriber(
            self.detections_topic,
            AprilTagDetectionArray,
            self.cb,
            queue_size=10,
        )

        rospy.loginfo("tag_to_polar_node started.")
        rospy.loginfo("detections_topic: %s", self.detections_topic)
        rospy.loginfo("output_topic:     %s", self.output_topic)
        rospy.loginfo("output_frame_id:  %s", self.output_frame_id)
        rospy.loginfo(
            "target_tag_id=%d use_first_detection=%s",
            self.target_tag_id,
            self.use_first_detection,
        )
        rospy.loginfo(
            "theta_offset=%.4f enabled=%s deadband=%.4f enabled=%s",
            self.theta_offset_rad,
            self.enable_theta_offset,
            self.theta_deadband_rad,
            self.enable_theta_deadband,
        )
        rospy.loginfo(
            "distance_offset=%.3f min_distance=%.3f",
            self.distance_offset,
            self.min_distance,
        )

    def select_detection(self, msg):
        if not msg.detections:
            return None, None

        if self.use_first_detection:
            det = msg.detections[0]
            if len(det.id) > 0:
                return det, int(det.id[0])
            return None, None

        for det in msg.detections:
            if len(det.id) == 0:
                continue
            tag_id = int(det.id[0])
            if tag_id == self.target_tag_id:
                return det, tag_id

        return None, None

    def compute_polar(self, det):
        pose = det.pose.pose.pose
        px = float(pose.position.x)
        pz = float(pose.position.z)

        raw_r = math.sqrt(px * px + pz * pz)
        r = max(self.min_distance, raw_r - self.distance_offset)

        # Camera optical frame convention: x right, z forward.
        # Control convention: positive theta means the target is on the left.
        theta = wrap_angle(-math.atan2(px, pz))
        if self.enable_theta_offset:
            theta = wrap_angle(theta - self.theta_offset_rad)
        if self.enable_theta_deadband and abs(theta) < self.theta_deadband_rad:
            theta = 0.0

        return r, theta, px, pz, raw_r

    @staticmethod
    def pick_stamp(msg, det):
        if det.pose.header.stamp != rospy.Time():
            return det.pose.header.stamp
        if msg.header.stamp != rospy.Time():
            return msg.header.stamp
        return rospy.Time.now()

    def cb(self, msg):
        det, tag_id = self.select_detection(msg)
        if det is None:
            return

        r, theta, px, pz, raw_r = self.compute_polar(det)
        out = PointStamped()
        out.header.stamp = self.pick_stamp(msg, det)
        out.header.frame_id = self.output_frame_id
        out.point.x = float(r)
        out.point.y = float(theta)
        out.point.z = float(tag_id)

        self.pub.publish(out)

        rospy.logdebug(
            "tag_id=%d px=%.3f pz=%.3f raw_r=%.3f r=%.3f theta=%.4f",
            tag_id,
            px,
            pz,
            raw_r,
            r,
            theta,
        )


if __name__ == "__main__":
    try:
        TagToPolarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
