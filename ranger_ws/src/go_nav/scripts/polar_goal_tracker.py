#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

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
        self.camera_offset_x_m = float(rospy.get_param("~camera_offset_x_m", 0.0))
        self.camera_offset_y_m = float(rospy.get_param("~camera_offset_y_m", 0.0))

        self.distance_offset = float(rospy.get_param("~distance_offset", 0.6))
        self.min_distance = float(rospy.get_param("~min_distance", 0.0))

        self.use_odom_fallback = get_bool_param("~use_odom_fallback", False)
        self.odom_topic = rospy.get_param("~odom_topic", "/go2/sport_odom")
        self.lost_timeout_sec = float(rospy.get_param("~lost_timeout_sec", 0.12))
        self.predict_timeout_sec = float(rospy.get_param("~predict_timeout_sec", 1.0))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))
        self.alpha = max(0.0, min(1.0, float(rospy.get_param("~alpha", 0.5))))
        self.odom_timeout_sec = float(rospy.get_param("~odom_timeout_sec", 0.25))
        self.max_odom_jump_m = float(rospy.get_param("~max_odom_jump_m", 0.75))
        self.max_odom_yaw_jump_rad = float(
            rospy.get_param("~max_odom_yaw_jump_rad", 1.20)
        )

        self.state_lock = threading.RLock()

        self.have_odom = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_odom_receive_time = rospy.Time(0)

        self.have_tag_estimate = False
        self.tag_x_odom = 0.0
        self.tag_y_odom = 0.0
        self.last_tag_id = -1
        self.last_detection_time = rospy.Time(0)
        self.last_detection_receive_time = rospy.Time(0)

        self.pub = rospy.Publisher(self.output_topic, PointStamped, queue_size=10)
        self.sub_det = rospy.Subscriber(
            self.detections_topic,
            AprilTagDetectionArray,
            self.detection_cb,
            queue_size=1,
        )

        self.sub_odom = None
        self.timer = None
        if self.use_odom_fallback:
            self.sub_odom = rospy.Subscriber(
                self.odom_topic,
                Odometry,
                self.odom_cb,
                queue_size=1,
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
        rospy.loginfo(
            "camera_offset_x=%.3f camera_offset_y=%.3f",
            self.camera_offset_x_m,
            self.camera_offset_y_m,
        )
        rospy.loginfo("use_odom_fallback: %s", self.use_odom_fallback)
        if self.use_odom_fallback:
            rospy.loginfo("odom_topic:       %s", self.odom_topic)
            rospy.loginfo("alpha:            %.3f", self.alpha)
            rospy.loginfo("odom_timeout_sec: %.3f", self.odom_timeout_sec)

    def odom_cb(self, msg):
        robot_x = float(msg.pose.pose.position.x)
        robot_y = float(msg.pose.pose.position.y)
        robot_yaw = quat_to_yaw(msg.pose.pose.orientation)
        if not all(math.isfinite(value) for value in (robot_x, robot_y, robot_yaw)):
            rospy.logwarn_throttle(1.0, "Non-finite odometry pose ignored.")
            return

        now = rospy.Time.now()
        with self.state_lock:
            if self.have_odom:
                jump_m = math.hypot(robot_x - self.robot_x, robot_y - self.robot_y)
                jump_yaw = abs(wrap_angle(robot_yaw - self.robot_yaw))
                position_jump = (
                    self.max_odom_jump_m > 0.0 and jump_m > self.max_odom_jump_m
                )
                yaw_jump = (
                    self.max_odom_yaw_jump_rad > 0.0
                    and jump_yaw > self.max_odom_yaw_jump_rad
                )
                if position_jump or yaw_jump:
                    self.have_tag_estimate = False
                    rospy.logwarn(
                        "Odometry discontinuity (%.3f m, %.3f rad); "
                        "clearing cached tag position.",
                        jump_m,
                        jump_yaw,
                    )

            self.robot_x = robot_x
            self.robot_y = robot_y
            self.robot_yaw = robot_yaw
            self.last_odom_receive_time = now
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

        camera_r = math.sqrt(px * px + pz * pz)

        # Camera optical frame convention: x right, z forward.
        # Control convention: positive theta means the target is on the left.
        theta = wrap_angle(-math.atan2(px, pz))
        if self.enable_theta_offset:
            theta = wrap_angle(theta - self.theta_offset_rad)

        tag_x_base = self.camera_offset_x_m + camera_r * math.cos(theta)
        tag_y_base = self.camera_offset_y_m + camera_r * math.sin(theta)
        raw_r = math.hypot(tag_x_base, tag_y_base)
        theta_for_estimate = math.atan2(tag_y_base, tag_x_base)
        r = max(self.min_distance, raw_r - self.distance_offset)
        theta = theta_for_estimate
        if self.enable_theta_deadband and abs(theta) < self.theta_deadband_rad:
            theta = 0.0

        return r, theta, px, pz, raw_r, theta_for_estimate

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

        r, theta, px, pz, raw_r, theta_for_estimate = self.compute_polar(det)
        stamp = self.pick_stamp(msg, det)
        receive_time = rospy.Time.now()

        with self.state_lock:
            reset_estimate = self.last_detection_receive_time == rospy.Time(0)
            if not reset_estimate:
                detection_gap = (
                    receive_time - self.last_detection_receive_time
                ).to_sec()
                reset_estimate = (
                    detection_gap < 0.0
                    or detection_gap > self.predict_timeout_sec
                )
            self.last_detection_time = stamp
            self.last_detection_receive_time = receive_time
            self.last_tag_id = tag_id

            if self.use_odom_fallback and self.have_odom:
                # Cache the physical tag position. distance_offset is a desired
                # stopping offset and deadband is an output behavior; neither
                # should distort the world-frame tag estimate.
                self.update_tag_estimate_from_polar(
                    raw_r, theta_for_estimate, reset=reset_estimate
                )

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

    def update_tag_estimate_from_polar(self, raw_r, theta, reset=False):
        global_bearing = wrap_angle(self.robot_yaw + theta)
        meas_x = self.robot_x + raw_r * math.cos(global_bearing)
        meas_y = self.robot_y + raw_r * math.sin(global_bearing)

        if reset or not self.have_tag_estimate:
            self.tag_x_odom = meas_x
            self.tag_y_odom = meas_y
            self.have_tag_estimate = True
            return

        self.tag_x_odom = self.alpha * meas_x + (1.0 - self.alpha) * self.tag_x_odom
        self.tag_y_odom = self.alpha * meas_y + (1.0 - self.alpha) * self.tag_y_odom

    def estimate_polar_from_odom(self):
        dx = self.tag_x_odom - self.robot_x
        dy = self.tag_y_odom - self.robot_y

        raw_r = math.sqrt(dx * dx + dy * dy)
        r = max(self.min_distance, raw_r - self.distance_offset)
        theta = wrap_angle(math.atan2(dy, dx) - self.robot_yaw)
        if self.enable_theta_deadband and abs(theta) < self.theta_deadband_rad:
            theta = 0.0

        return r, theta

    def timer_cb(self, _event):
        if not self.use_odom_fallback:
            return
        now = rospy.Time.now()
        with self.state_lock:
            if not self.have_odom:
                rospy.logwarn_throttle(
                    2.0,
                    "Odometry fallback enabled, but no message received on %s.",
                    self.odom_topic,
                )
                return
            if not self.have_tag_estimate:
                return
            if self.last_detection_receive_time == rospy.Time(0):
                return

            odom_age = (now - self.last_odom_receive_time).to_sec()
            if odom_age < 0.0 or odom_age > self.odom_timeout_sec:
                rospy.logwarn_throttle(
                    1.0,
                    "Odometry is stale (%.2f s); suppressing tag prediction.",
                    odom_age,
                )
                return

            lost_age = (now - self.last_detection_receive_time).to_sec()
            if lost_age <= self.lost_timeout_sec:
                return
            if lost_age > self.predict_timeout_sec:
                rospy.logwarn_throttle(
                    2.0,
                    "Tag lost for %.2f s; stop odometry fallback publishing.",
                    lost_age,
                )
                return

            r, theta = self.estimate_polar_from_odom()
            tag_id = self.last_tag_id

        self.publish_polar(r, theta, tag_id, now)
        rospy.loginfo_throttle(
            1.0,
            "AprilTag occluded for %.2f s; publishing odometry prediction.",
            lost_age,
        )

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
