#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import pickle
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

pt1 = [379, 716]
pt2 = [858, 716]
pt3 = [492, 352]
pt4 = [743, 354]
width = 448
height = 300
# meter 단위임
box_width = 0.52
box_height = 0.95
#meter/pixel 비율
resolution = 17/1024  # 17m/4096 픽셀

pt1_warp = [(width/2 - box_width/(resolution*2)), (height)]
pt2_warp = [(width/2 + box_width/(resolution*2)), (height)]
pt3_warp = [(width/2 - box_width/(resolution*2)), (height - box_height/resolution)]
pt4_warp = [(width/2 + box_width/(resolution*2)), (height - box_height/resolution)]

def calculate_perspective_transform_matrix(pts_src, pts_dst, directory, file_name='warp_matrix.pkl'):
    # 경로 검증 및 생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        rospy.loginfo(f"Created directory: {directory}")

    # 포인트 배열을 numpy float32 타입으로 변환
    pts_src = np.array(pts_src, dtype=np.float32)
    pts_dst = np.array(pts_dst, dtype=np.float32)

    # 원근 변환 행렬 계산
    try:
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(matrix, f)
        rospy.loginfo("Transform matrix saved to %s", file_path)
        return matrix
    except cv2.error as e:
        rospy.logerr("Error calculating perspective transform: %s", e)
        return None

def load_warp_transform_matrix(directory, file_name='warp_matrix.pkl'):
    file_path = os.path.join(directory, file_name)
    try:
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
        print("Loaded transform matrix from", file_path)
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        transformed_image = warp_transform(cv_image)
        cv2.imshow("Transformed Image", transformed_image)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr("Failed to convert image: %s", e)

def warp_transform(cv_image):
    top_view = cv2.warpPerspective(cv_image, warp_matrix, (width, height))
    return top_view

if __name__ == "__main__":
    rospy.init_node('image_transformer_node')
    bridge = CvBridge()

    # Define source and destination points
    pts_src = [pt1, pt2, pt3, pt4]  # Adjust these values
    pts_dst = [pt1_warp, pt2_warp, pt3_warp, pt4_warp]  # Adjust these values

    # Directory and file settings
    directory = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
    warp_matrix = calculate_perspective_transform_matrix(pts_src, pts_dst, directory)

    # Ensure warp_matrix was successfully created before subscribing to image topic
    if warp_matrix is not None:
        image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
        rospy.spin()
    else:
        rospy.logerr("Failed to create warp matrix. Node will shutdown.")
        rospy.signal_shutdown("Failed to create warp matrix.")