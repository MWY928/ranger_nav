#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import pyrealsense2 as rs

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge


class RealsenseRosNode(object):
    def __init__(self):
        rospy.init_node("realsense_rgbd_node", anonymous=False)

        # -----------------------------
        # ROS params
        # -----------------------------
        self.serial = rospy.get_param("~serial", None)
        self.prefer_fps = int(rospy.get_param("~fps", 30))
        self.enable_align_to_color = bool(rospy.get_param("~align_to_color", True))

        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.color_info_topic = rospy.get_param("~color_info_topic", "/camera/color/camera_info")
        self.depth_info_topic = rospy.get_param("~depth_info_topic", "/camera/aligned_depth_to_color/camera_info")

        self.color_frame_id = rospy.get_param("~color_frame_id", "camera_color_optical_frame")
        self.depth_frame_id = rospy.get_param("~depth_frame_id", "camera_color_optical_frame")  
        # 因为这里发布的是“对齐到彩色图”的深度，所以默认也用 color optical frame

        self.bridge = CvBridge()

        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None

        self.color_pub = rospy.Publisher(self.color_topic, Image, queue_size=10)
        self.depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=10)
        self.color_info_pub = rospy.Publisher(self.color_info_topic, CameraInfo, queue_size=10)
        self.depth_info_pub = rospy.Publisher(self.depth_info_topic, CameraInfo, queue_size=10)

        self.color_camera_info_msg = None
        self.depth_camera_info_msg = None

        self._setup_realsense()

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("realsense_rgbd_node started successfully.")

    @staticmethod
    def list_devices():
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]

    def _setup_realsense(self):
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found.")

        # 若未指定 serial，则默认取第一个设备
        if self.serial is None:
            self.serial = devices[0].get_info(rs.camera_info.serial_number)
            rospy.logwarn("~serial not set, using first RealSense device: %s", self.serial)

        dev = None
        for d in devices:
            if d.get_info(rs.camera_info.serial_number) == self.serial:
                dev = d
                break

        if dev is None:
            raise RuntimeError("Device with serial {} not found.".format(self.serial))

        product_line = dev.get_info(rs.camera_info.product_line)
        rospy.loginfo("Using RealSense serial=%s, product_line=%s", self.serial, product_line)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial)

        # 这里按你原脚本的 L500 配置方式设置
        # 若不是 L500，也给一个通用保守配置
        if product_line == "L500":
            depth_w, depth_h, depth_fps = 640, 480, min(self.prefer_fps, 30)
            color_w, color_h, color_fps = 1280, 720, min(self.prefer_fps, 30)
        else:
            depth_w, depth_h, depth_fps = 640, 480, min(self.prefer_fps, 30)
            color_w, color_h, color_fps = 640, 480, min(self.prefer_fps, 30)

        self.config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
        self.config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)

        self.profile = self.pipeline.start(self.config)

        if self.enable_align_to_color:
            self.align = rs.align(rs.stream.color)
            rospy.loginfo("Depth will be aligned to color stream.")
        else:
            self.align = None
            rospy.loginfo("Alignment disabled.")

        # 预热几帧，避免刚启动时数据不稳定
        for _ in range(10):
            self.pipeline.wait_for_frames()

        self._prepare_camera_info()

    def _prepare_camera_info(self):
        """
        构造 CameraInfo。
        如果深度对齐到彩色图，则深度也使用彩色相机内参。
        """
        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get initial frames for CameraInfo.")

        c_prof = color_frame.get_profile().as_video_stream_profile()
        c_intr = c_prof.get_intrinsics()

        color_info = CameraInfo()
        color_info.width = c_intr.width
        color_info.height = c_intr.height
        color_info.distortion_model = "plumb_bob"
        color_info.D = list(c_intr.coeffs)
        color_info.K = [
            c_intr.fx, 0.0,      c_intr.ppx,
            0.0,      c_intr.fy, c_intr.ppy,
            0.0,      0.0,       1.0
        ]
        color_info.R = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        color_info.P = [
            c_intr.fx, 0.0,      c_intr.ppx, 0.0,
            0.0,      c_intr.fy, c_intr.ppy, 0.0,
            0.0,      0.0,       1.0,       0.0
        ]

        self.color_camera_info_msg = color_info

        if self.align is not None:
            # 对齐到彩色图后，深度图像素坐标系与彩色图一致
            depth_info = CameraInfo()
            depth_info.width = c_intr.width
            depth_info.height = c_intr.height
            depth_info.distortion_model = "plumb_bob"
            depth_info.D = list(c_intr.coeffs)
            depth_info.K = list(color_info.K)
            depth_info.R = list(color_info.R)
            depth_info.P = list(color_info.P)
            self.depth_camera_info_msg = depth_info
        else:
            d_prof = depth_frame.get_profile().as_video_stream_profile()
            d_intr = d_prof.get_intrinsics()

            depth_info = CameraInfo()
            depth_info.width = d_intr.width
            depth_info.height = d_intr.height
            depth_info.distortion_model = "plumb_bob"
            depth_info.D = list(d_intr.coeffs)
            depth_info.K = [
                d_intr.fx, 0.0,      d_intr.ppx,
                0.0,      d_intr.fy, d_intr.ppy,
                0.0,      0.0,       1.0
            ]
            depth_info.R = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
            depth_info.P = [
                d_intr.fx, 0.0,      d_intr.ppx, 0.0,
                0.0,      d_intr.fy, d_intr.ppy, 0.0,
                0.0,      0.0,       1.0,       0.0
            ]
            self.depth_camera_info_msg = depth_info

    def _make_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def publish_once(self):
        frames = self.pipeline.wait_for_frames()

        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            rospy.logwarn_throttle(5.0, "Failed to get color or depth frame.")
            return

        # -----------------------------
        # Color image
        # RealSense给的是BGR，这里转成RGB后按rgb8发布
        # -----------------------------
        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = color_bgr[:, :, ::-1].copy()

        color_msg = self.bridge.cv2_to_imgmsg(color_rgb, encoding="rgb8")
        color_msg.header = self._make_header(self.color_frame_id)

        # -----------------------------
        # Depth image
        # 发布原始 z16 深度，单位通常是相机原始深度单位（常见为毫米尺度对应的uint16）
        # 这样最兼容 ROS 生态
        # -----------------------------
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()  # uint16
        depth_msg = self.bridge.cv2_to_imgmsg(depth_raw, encoding="16UC1")
        depth_msg.header = self._make_header(self.depth_frame_id)

        # -----------------------------
        # CameraInfo
        # -----------------------------
        color_info = CameraInfo()
        color_info.header = color_msg.header
        color_info.width = self.color_camera_info_msg.width
        color_info.height = self.color_camera_info_msg.height
        color_info.distortion_model = self.color_camera_info_msg.distortion_model
        color_info.D = list(self.color_camera_info_msg.D)
        color_info.K = list(self.color_camera_info_msg.K)
        color_info.R = list(self.color_camera_info_msg.R)
        color_info.P = list(self.color_camera_info_msg.P)

        depth_info = CameraInfo()
        depth_info.header = depth_msg.header
        depth_info.width = self.depth_camera_info_msg.width
        depth_info.height = self.depth_camera_info_msg.height
        depth_info.distortion_model = self.depth_camera_info_msg.distortion_model
        depth_info.D = list(self.depth_camera_info_msg.D)
        depth_info.K = list(self.depth_camera_info_msg.K)
        depth_info.R = list(self.depth_camera_info_msg.R)
        depth_info.P = list(self.depth_camera_info_msg.P)

        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.color_info_pub.publish(color_info)
        self.depth_info_pub.publish(depth_info)

    def spin(self):
        rate = rospy.Rate(self.prefer_fps)
        while not rospy.is_shutdown():
            try:
                self.publish_once()
            except Exception as e:
                rospy.logerr_throttle(2.0, "Error while publishing RGB-D frames: %s", str(e))
            rate.sleep()

    def shutdown(self):
        rospy.loginfo("Shutting down realsense_rgbd_node...")
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as e:
            rospy.logwarn("Failed to stop RealSense pipeline cleanly: %s", str(e))


if __name__ == "__main__":
    try:
        node = RealsenseRosNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Fatal error in realsense_rgbd_node: %s", str(e))