#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import pickle
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# 기존 warp matrix를 로드하는 함수
def load_warp_transform_matrix(directory, file_name='warp_matrix.pkl'):
    file_path = os.path.join(directory, file_name)
    try:
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
        rospy.loginfo("Loaded transform matrix from %s", file_path)
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        transformed_image = warp_transform(cv_image)
        
        # 변환된 이미지를 publish
        transformed_msg = bridge.cv2_to_imgmsg(transformed_image, "bgr8")
        transformed_image_pub.publish(transformed_msg)
    except Exception as e:
        rospy.logerr("Failed to convert and transform image: %s", e)

def warp_transform(cv_image):
    top_view = cv2.warpPerspective(cv_image, warp_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return top_view

if __name__ == "__main__":
    rospy.init_node('image_transform_publisher_node')
    bridge = CvBridge()

    # Directory and file settings
    directory = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
    warp_matrix = load_warp_transform_matrix(directory)

    width = 448
    height = 300

    # Ensure warp_matrix was successfully loaded before subscribing to image topic
    if warp_matrix is not None:
        image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
        transformed_image_pub = rospy.Publisher("/transformed_image", Image, queue_size=1)
        rospy.spin()
    else:
        rospy.logerr("Failed to load warp matrix. Node will shutdown.")
        rospy.signal_shutdown("Failed to load warp matrix.")
