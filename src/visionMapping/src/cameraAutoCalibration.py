#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import numpy as np
import json
import glob
import threading

class CameraCalibration:
    def __init__(self, image_dir='/home/innodriver/InnoDriver_ws/src/visionMapping/src/calibration_images', calibration_file='/home/innodriver/InnoDriver_ws/src/visionMapping/src/calibration_images/calibration_data.json', checkerboard_size=(6, 9)):
        self.bridge = CvBridge()
        self.image_sub_raw = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.image_count = 0
        self.image_dir = image_dir
        self.calibration_file = calibration_file
        self.checkerboard_size = checkerboard_size
        self.capturing_images = True
        self.calibration_done = False
        self.mtx = None
        self.dist = None

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.start_image_capture()

        rospy.loginfo("Capturing images every 5 seconds. Press 'c' to start calibration when done.")

    def start_image_capture(self):
        self.capture_thread = threading.Thread(target=self.capture_images)
        self.capture_thread.start()

    def capture_images(self):
        while not rospy.is_shutdown() and self.capturing_images:
            if self.image_count < 20:  # 캡처할 이미지 수 제한
                rospy.sleep(5)
                rospy.loginfo(f"Capturing image {self.image_count + 1}")
                self.save_image(self.current_image)
            else:
                self.capturing_images = False
                rospy.loginfo("Captured 20 images, stopping image capture.")
                self.calibrate_camera()

    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.calibration_done:
                undistorted_image = self.undistort_image(self.current_image)
                self.process_and_show_image(undistorted_image)
            else:
                self.process_and_show_image(self.current_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def process_and_show_image(self, image):
        cv2.imshow('Image', image)
        cv2.waitKey(1)

    def save_image(self, image):
        if image is not None:
            image_filename = os.path.join(self.image_dir, f'image_{self.image_count:02d}.png')
            cv2.imwrite(image_filename, image)
            self.image_count += 1
            rospy.loginfo(f"Saved image: {image_filename}")

    def calibrate_camera(self):
        rospy.loginfo("Starting camera calibration...")
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[1], 0:self.checkerboard_size[0]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        images = glob.glob(f'{self.image_dir}/*.png')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                img = cv2.drawChessboardCorners(img, self.checkerboard_size, corners, ret)
                cv2.imshow('Chessboard Corners', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        if len(objpoints) == 0 or len(imgpoints) == 0:
            rospy.logerr("Calibration failed: No valid points were found in images.")
            return

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        calibration_data = {'mtx': self.mtx.tolist(), 'dist': self.dist.tolist()}

        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)

        self.calibration_done = True
        rospy.loginfo("Calibration done. Calibration data saved to " + self.calibration_file)

    def undistort_image(self, image):
        if self.mtx is None or self.dist is None:
            return image
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(self.mtx), np.array(self.dist), (w,h), 1, (w,h))
        undistorted_image = cv2.undistort(image, np.array(self.mtx), np.array(self.dist), None, newcameramtx)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]
        return undistorted_image

if __name__ == '__main__':
    rospy.init_node('camera_calibration', anonymous=True)
    CameraCalibration()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()
