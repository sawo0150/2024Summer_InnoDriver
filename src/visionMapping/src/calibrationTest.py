#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import json

class CameraComparison:
    def __init__(self, calibration_file='/home/innodriver/InnoDriver_ws/src/visionMapping/src/calibration_images/calibration_data.json'):
        self.bridge = CvBridge()
        self.image_sub_raw = rospy.Subscriber("/usb_cam/image_raw", Image, self.raw_image_callback)
        self.image_pub_calibrated = rospy.Publisher("/calibrated_image", Image, queue_size=1)
        self.calibration_file = calibration_file
        self.mtx = None
        self.dist = None
        self.load_calibration_data()

    def load_calibration_data(self):
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                self.mtx = np.array(data['mtx'])
                self.dist = np.array(data['dist'])
                rospy.loginfo("Calibration data loaded successfully.")
        except FileNotFoundError:
            rospy.logerr(f"Calibration file '{self.calibration_file}' not found. Cannot perform calibration.")
            rospy.signal_shutdown("Calibration file not found.")

    def raw_image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            calibrated_image = self.undistort_image(cv_image)
            self.publish_calibrated_image(calibrated_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def undistort_image(self, image):
        if self.mtx is None or self.dist is None:
            return image
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
        # x, y, w, h = roi
        # undistorted_image = undistorted_image[y:y+h, x:x+w]
        return undistorted_image

    def publish_calibrated_image(self, image):
        try:
            calibrated_img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.image_pub_calibrated.publish(calibrated_img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

if __name__ == '__main__':
    rospy.init_node('camera_comparison', anonymous=True)
    camera_comparison = CameraComparison()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()
