#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry


def get_bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class SimplePolarGoalTracker(object):
    """
    AprilTag polar tracker for Go2 + remote realsense2_ros images.

    Expected deployment:
    - Go2 Jetson publishes D435 image topics through realsense2_ros.
    - This machine runs apriltag_ros against those remote image topics.
    - This node subscribes to apriltag_ros /tag_detections.
    - Optional odom fallback can be enabled once Go2 odometry is available.

    Output PointStamped:
    - point.x = r, meters after distance_offset
    - point.y = theta, radians
    - point.z = tag_id

    Angle convention:
    - theta > 0: tag is left of the camera optical axis, turn left
    - theta < 0: tag is right of the camera optical axis, turn right
    """

    def __init__(self):
        rospy.init_node("simple_polar_goal_tracker", anonymous=False)

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

        self.use_odom_fallback = get_bool_param("~use_odom_fallback", False)
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.lost_timeout_sec = float(rospy.get_param("~lost_timeout_sec", 0.30))
        self.predict_timeout_sec = float(rospy.get_param("~predict_timeout_sec", 5.0))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))
        self.alpha = float(rospy.get_param("~alpha", 0.5))

        self.have_odom = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.have_tag_estimate = False
        self.tag_x_odom = 0.0
        self.tag_y_odom = 0.0
        self.last_tag_id = -1
        self.last_detection_time = rospy.Time(0)

        self.pub = rospy.Publisher(self.output_topic, PointStamped, queue_size=10)
        self.sub_det = rospy.Subscriber(
            self.detections_topic,
            AprilTagDetectionArray,
            self.detection_cb,
            queue_size=10,
        )

        self.sub_odom = None
        self.timer = None
        if self.use_odom_fallback:
            self.sub_odom = rospy.Subscriber(
                self.odom_topic,
                Odometry,
                self.odom_cb,
                queue_size=30,
            )
            self.timer = rospy.Timer(
                rospy.Duration(1.0 / max(self.publish_rate_hz, 1.0)),
                self.timer_cb,
            )

        rospy.loginfo("simple_polar_goal_tracker started.")
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
        rospy.loginfo("use_odom_fallback: %s", self.use_odom_fallback)
        if self.use_odom_fallback:
            rospy.loginfo("odom_topic:       %s", self.odom_topic)
            rospy.loginfo("alpha:            %.3f", self.alpha)

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.have_odom = True

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

    def detection_cb(self, msg):
        det, tag_id = self.select_detection(msg)
        if det is None:
            return

        r, theta, px, pz, raw_r = self.compute_polar(det)
        stamp = self.pick_stamp(msg, det)

        self.last_detection_time = stamp
        self.last_tag_id = tag_id

        if self.use_odom_fallback and self.have_odom:
            self.update_tag_estimate_from_polar(r, theta)

        self.publish_polar(r, theta, tag_id, stamp)

        rospy.logdebug(
            "tag_id=%d px=%.3f pz=%.3f raw_r=%.3f r=%.3f theta=%.4f",
            tag_id,
            px,
            pz,
            raw_r,
            r,
            theta,
        )

    def update_tag_estimate_from_polar(self, r, theta):
        global_bearing = wrap_angle(self.robot_yaw + theta)
        meas_x = self.robot_x + r * math.cos(global_bearing)
        meas_y = self.robot_y + r * math.sin(global_bearing)

        if not self.have_tag_estimate:
            self.tag_x_odom = meas_x
            self.tag_y_odom = meas_y
            self.have_tag_estimate = True
            return

        self.tag_x_odom = self.alpha * meas_x + (1.0 - self.alpha) * self.tag_x_odom
        self.tag_y_odom = self.alpha * meas_y + (1.0 - self.alpha) * self.tag_y_odom

    def estimate_polar_from_odom(self):
        dx = self.tag_x_odom - self.robot_x
        dy = self.tag_y_odom - self.robot_y

        r = max(self.min_distance, math.sqrt(dx * dx + dy * dy))
        theta = wrap_angle(math.atan2(dy, dx) - self.robot_yaw)
        if self.enable_theta_deadband and abs(theta) < self.theta_deadband_rad:
            theta = 0.0

        return r, theta

    def timer_cb(self, _event):
        if not self.use_odom_fallback:
            return
        if not self.have_odom:
            return
        if not self.have_tag_estimate:
            return
        if self.last_detection_time == rospy.Time(0):
            return

        now = rospy.Time.now()
        lost_age = (now - self.last_detection_time).to_sec()

        if lost_age <= self.lost_timeout_sec:
            return

        if lost_age > self.predict_timeout_sec:
            rospy.logwarn_throttle(
                2.0,
                "Tag lost for %.2f s, stop odom fallback publishing.",
                lost_age,
            )
            return

        r, theta = self.estimate_polar_from_odom()
        self.publish_polar(r, theta, self.last_tag_id, now)

    def publish_polar(self, r, theta, tag_id, stamp):
        out = PointStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.output_frame_id
        out.point.x = float(r)
        out.point.y = float(theta)
        out.point.z = float(tag_id)
        self.pub.publish(out)


if __name__ == "__main__":
    try:
        SimplePolarGoalTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
