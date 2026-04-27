#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def quat_to_yaw(q):
    # yaw from quaternion (Z-up convention)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class PolarGoalTrackerNode(object):
    """
    Publish pointgoal-like polar target (r, theta):
    - Tag visible: use direct AprilTag observation.
    - Tag lost: use cached global tag position + odom(/imu yaw) for dead reckoning.

    Output convention (same as current pipeline):
    - PointStamped.point.x = r      (meter)
    - PointStamped.point.y = theta  (radian)
    - PointStamped.point.z = tag_id
    """

    def __init__(self):
        rospy.init_node("polar_goal_tracker_node", anonymous=False)

        self.detections_topic = rospy.get_param("~detections_topic", "/tag_detections")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.imu_topic = rospy.get_param("~imu_topic", "/imu/data")
        self.output_topic = rospy.get_param("~output_topic", "/tag_polar")

        self.target_tag_id = int(rospy.get_param("~target_tag_id", 0))
        self.use_first_detection = bool(rospy.get_param("~use_first_detection", False))

        self.theta_offset_rad = float(rospy.get_param("~theta_offset_rad", -0.1))
        self.theta_deadband_rad = float(rospy.get_param("~theta_deadband_rad", 0.0))

        self.use_imu_yaw = bool(rospy.get_param("~use_imu_yaw", False))
        self.fallback_apply_theta_offset = bool(
            rospy.get_param("~fallback_apply_theta_offset", False)
        )
        self.fallback_frame_id = rospy.get_param("~fallback_frame_id", "base_link")

        self.lost_timeout_sec = float(rospy.get_param("~lost_timeout_sec", 0.20))
        self.predict_timeout_sec = float(rospy.get_param("~predict_timeout_sec", 6.0))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))

        self.have_odom = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        self.have_imu = False
        self.imu_yaw = 0.0
        self.imu_yaw_offset = None

        self.have_tag_fix = False
        self.tag_odom_x = 0.0
        self.tag_odom_y = 0.0
        self.last_tag_id = -1
        self.last_detection_time = rospy.Time(0)

        self.pub = rospy.Publisher(self.output_topic, PointStamped, queue_size=20)
        self.sub_det = rospy.Subscriber(
            self.detections_topic, AprilTagDetectionArray, self._det_cb, queue_size=10
        )
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=50)
        self.sub_imu = None
        if self.use_imu_yaw:
            self.sub_imu = rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=50)

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = rospy.Timer(rospy.Duration(period), self._timer_cb)

        rospy.loginfo("polar_goal_tracker_node started.")
        rospy.loginfo("Subscribe detections: %s", self.detections_topic)
        rospy.loginfo("Subscribe odom:       %s", self.odom_topic)
        if self.use_imu_yaw:
            rospy.loginfo("Subscribe imu:        %s", self.imu_topic)
        rospy.loginfo("Publish:              %s", self.output_topic)
        rospy.loginfo(
            "target_tag_id=%d, use_first_detection=%s, use_imu_yaw=%s",
            self.target_tag_id,
            str(self.use_first_detection),
            str(self.use_imu_yaw),
        )

    def _get_heading_yaw(self):
        if self.use_imu_yaw and self.have_imu:
            if self.imu_yaw_offset is not None:
                return wrap_angle(self.imu_yaw + self.imu_yaw_offset)
            return self.imu_yaw
        if self.have_odom:
            return self.odom_yaw
        return None

    def _odom_cb(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.odom_yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.have_odom = True

        if self.use_imu_yaw and self.have_imu and self.imu_yaw_offset is None:
            self.imu_yaw_offset = wrap_angle(self.odom_yaw - self.imu_yaw)
            rospy.loginfo("IMU yaw offset initialized: %.4f rad", self.imu_yaw_offset)

    def _imu_cb(self, msg):
        self.imu_yaw = quat_to_yaw(msg.orientation)
        self.have_imu = True
        if self.have_odom and self.imu_yaw_offset is None:
            self.imu_yaw_offset = wrap_angle(self.odom_yaw - self.imu_yaw)
            rospy.loginfo("IMU yaw offset initialized: %.4f rad", self.imu_yaw_offset)

    def _select_detection(self, msg):
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

    def _publish_polar(self, stamp, frame_id, r, theta, tag_id):
        out = PointStamped()
        out.header.stamp = stamp if stamp != rospy.Time() else rospy.Time.now()
        out.header.frame_id = frame_id if frame_id else self.fallback_frame_id
        out.point.x = float(r)
        out.point.y = float(theta)
        out.point.z = float(tag_id)
        self.pub.publish(out)

    def _det_cb(self, msg):
        det, tag_id = self._select_detection(msg)
        if det is None:
            return

        pose = det.pose.pose.pose
        px = float(pose.position.x)
        pz = float(pose.position.z)

        r = math.sqrt(px * px + pz * pz)
        theta_raw = math.atan2(px, pz)
        theta = theta_raw - self.theta_offset_rad
        if abs(theta) < self.theta_deadband_rad:
            theta = 0.0

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        frame_id = msg.header.frame_id if msg.header.frame_id else "camera_color_optical_frame"
        self._publish_polar(stamp, frame_id, r, theta, tag_id)

        self.last_detection_time = rospy.Time.now()
        self.last_tag_id = tag_id

        yaw = self._get_heading_yaw()
        if self.have_odom and yaw is not None:
            # Cache tag position in odom frame when visible.
            bearing_global = wrap_angle(yaw + theta)
            self.tag_odom_x = self.odom_x + r * math.cos(bearing_global)
            self.tag_odom_y = self.odom_y + r * math.sin(bearing_global)
            self.have_tag_fix = True

        rospy.logdebug(
            "det tag=%d r=%.3f theta=%.4f (raw=%.4f), have_fix=%s",
            tag_id,
            r,
            theta,
            theta_raw,
            str(self.have_tag_fix),
        )

    def _timer_cb(self, _event):
        if not self.have_tag_fix or not self.have_odom:
            return

        if self.last_detection_time == rospy.Time(0):
            return

        now = rospy.Time.now()
        lost_age = (now - self.last_detection_time).to_sec()

        # Recently visible: direct detection publisher is active, no fallback needed.
        if lost_age <= self.lost_timeout_sec:
            return

        # Tag lost too long: stop predicting to avoid unlimited drift.
        if lost_age > self.predict_timeout_sec:
            rospy.logwarn_throttle(
                2.0,
                "Tag lost for %.2fs (>%.2fs), fallback publish stopped.",
                lost_age,
                self.predict_timeout_sec,
            )
            return

        yaw = self._get_heading_yaw()
        if yaw is None:
            return

        dx = self.tag_odom_x - self.odom_x
        dy = self.tag_odom_y - self.odom_y
        r = math.sqrt(dx * dx + dy * dy)
        theta = wrap_angle(math.atan2(dy, dx) - yaw)
        if self.fallback_apply_theta_offset:
            theta = wrap_angle(theta - self.theta_offset_rad)
        if abs(theta) < self.theta_deadband_rad:
            theta = 0.0

        self._publish_polar(now, self.fallback_frame_id, r, theta, self.last_tag_id)
        rospy.logdebug(
            "fallback tag=%d lost_age=%.2f r=%.3f theta=%.4f",
            self.last_tag_id,
            lost_age,
            r,
            theta,
        )


if __name__ == "__main__":
    try:
        PolarGoalTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
