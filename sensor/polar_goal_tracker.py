#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy

from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry


def quat_to_yaw(q):
    """Convert quaternion to planar yaw."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class SimplePolarGoalTracker(object):
    """
    AprilTag polar tracker with odom fallback.

    Output:
        point.x = r
        point.y = theta
        point.z = tag_id

    Angle convention:
        theta > 0: tag is on the left, robot should turn left
        theta < 0: tag is on the right, robot should turn right

    This matches:
        /cmd_vel.angular.z > 0: left turn
        /cmd_vel.angular.z < 0: right turn
    """

    def __init__(self):
        rospy.init_node("simple_polar_goal_tracker", anonymous=False)

        # Topics
        self.detections_topic = rospy.get_param("~detections_topic", "/tag_detections")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.output_topic = rospy.get_param("~output_topic", "/tag_polar")

        # Tag
        self.target_tag_id = int(rospy.get_param("~target_tag_id", 0))
        self.use_first_detection = bool(rospy.get_param("~use_first_detection", False))

        # Angle correction
        # New convention:
        #   theta_raw = -atan2(px, pz)
        #
        # If old front angle was -0.1037 rad,
        # new front angle is +0.1037 rad,
        # so set theta_offset_rad = +0.1037.
        self.theta_offset_rad = float(rospy.get_param("~theta_offset_rad", 0.0))
        self.theta_deadband_rad = float(rospy.get_param("~theta_deadband_rad", 0.03))

        # Fallback behavior
        self.lost_timeout_sec = float(rospy.get_param("~lost_timeout_sec", 0.30))
        self.predict_timeout_sec = float(rospy.get_param("~predict_timeout_sec", 5.0))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))

        # Low-pass filter for tag global position
        # alpha close to 1: trust new detections more
        # alpha close to 0: smoother but slower
        self.alpha = float(rospy.get_param("~alpha", 0.5))

        # Robot odom state
        self.have_odom = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Tag estimate in odom frame
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
        rospy.loginfo("odom_topic:       %s", self.odom_topic)
        rospy.loginfo("output_topic:     %s", self.output_topic)
        rospy.loginfo("theta_offset_rad: %.4f", self.theta_offset_rad)
        rospy.loginfo("theta_deadband_rad: %.4f", self.theta_deadband_rad)
        rospy.loginfo("alpha: %.3f", self.alpha)

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
            if int(det.id[0]) == self.target_tag_id:
                return det, int(det.id[0])

        return None, None

    def detection_cb(self, msg):
        det, tag_id = self.select_detection(msg)
        if det is None:
            return

        pose = det.pose.pose.pose
        px = float(pose.position.x)
        pz = float(pose.position.z)

        # Camera optical frame:
        #   x: right
        #   z: forward
        #
        # Use cmd_vel-compatible sign:
        #   theta > 0: tag left
        #   theta < 0: tag right
        r = math.sqrt(px * px + pz * pz)
        theta_raw = -math.atan2(px, pz)
        theta = wrap_angle(theta_raw - self.theta_offset_rad)
        theta = self.apply_deadband(theta)

        now = rospy.Time.now()
        self.last_detection_time = now
        self.last_tag_id = tag_id

        # If odom exists, update tag global estimate.
        if self.have_odom:
            self.update_tag_estimate_from_polar(r, theta)

        # Publish direct measurement immediately.
        self.publish_polar(r, theta, tag_id, now)

    def update_tag_estimate_from_polar(self, r, theta):
        """
        Convert robot-relative polar measurement to odom-frame tag position.

        theta convention:
            theta > 0 means left.
        """
        global_bearing = wrap_angle(self.robot_yaw + theta)

        meas_x = self.robot_x + r * math.cos(global_bearing)
        meas_y = self.robot_y + r * math.sin(global_bearing)

        if not self.have_tag_estimate:
            self.tag_x_odom = meas_x
            self.tag_y_odom = meas_y
            self.have_tag_estimate = True
        else:
            self.tag_x_odom = self.alpha * meas_x + (1.0 - self.alpha) * self.tag_x_odom
            self.tag_y_odom = self.alpha * meas_y + (1.0 - self.alpha) * self.tag_y_odom

    def estimate_polar_from_odom(self):
        """
        Convert stored odom-frame tag position back to robot-relative polar.
        """
        dx = self.tag_x_odom - self.robot_x
        dy = self.tag_y_odom - self.robot_y

        r = math.sqrt(dx * dx + dy * dy)
        theta = wrap_angle(math.atan2(dy, dx) - self.robot_yaw)
        theta = self.apply_deadband(theta)

        return r, theta

    def timer_cb(self, _event):
        if not self.have_odom:
            return
        if not self.have_tag_estimate:
            return
        if self.last_detection_time == rospy.Time(0):
            return

        now = rospy.Time.now()
        lost_age = (now - self.last_detection_time).to_sec()

        # If tag was just seen, detection_cb has already published it.
        if lost_age <= self.lost_timeout_sec:
            return

        # If tag has been lost for too long, stop publishing fallback.
        if lost_age > self.predict_timeout_sec:
            rospy.logwarn_throttle(
                2.0,
                "Tag lost for %.2f s, stop fallback publishing.",
                lost_age,
            )
            return

        r, theta = self.estimate_polar_from_odom()
        self.publish_polar(r, theta, self.last_tag_id, now)

    def apply_deadband(self, theta):
        if abs(theta) < self.theta_deadband_rad:
            return 0.0
        return theta

    def publish_polar(self, r, theta, tag_id, stamp):
        out = PointStamped()
        out.header.stamp = stamp
        out.header.frame_id = "base_link"

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