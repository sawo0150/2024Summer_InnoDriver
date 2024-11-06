#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pickle
import tensorflow as tf

# TensorFlow 초기화 전에 GPU 비활성화 설정
tf.config.set_visible_devices([], 'GPU')

from keras_segmentation.models.unet import mobilenet_unet
from missionRacing.msg import CrosswalkInfo  # 사용할 새로운 메시지 형식

class CrossWalkAnalizer:
    def __init__(self):
        # ROS 초기화
        rospy.init_node('crossWalkAnalizer', anonymous=True)
        # Subscribers
        self.image_sub_raw1 = rospy.Subscriber("/camera1/usb_cam1/image_raw", Image, self.callback_raw1)
        self.image_sub_raw2 = rospy.Subscriber("/camera2/usb_cam2/image_raw", Image, self.callback_raw2)
        self.info_pub = rospy.Publisher('/crosswalk_info', CrosswalkInfo, queue_size=10)
        self.bridge = CvBridge()
        
        # Directory and file settings
        self.directory = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
        self.warp_matrix = self.load_warp_transform_matrix(self.directory)
        
        self.width = 448
        self.height = 300
        self.resolution = 17/1024   # 17m/4096 픽셀
        self.laneWidth = 0.90       # 원래는 0.85m지만, 조금 조정?

        self.carWidth = 0.42
        self.carBoxHeight = 0.54
        # 차량의 위치 및 크기 정의
        self.car_box_dist = 0.61
        self.car_center_x = int(self.width / 2)
        self.car_center_y = self.height - int(self.car_box_dist / self.resolution/2)
        self.car_TR_center_y = self.height + int(self.car_box_dist / self.resolution/2)
        self.car_width = int(self.carWidth / self.resolution)
        self.car_height = int(self.carBoxHeight / self.resolution)
        self.last_1detection_time = rospy.Time.now()
        self.last_2detection_time = rospy.Time.now()

        self.crosswalk_detected = False
        self.crosswalk_distance = -1
        self.traffic_light_detected = False
        self.light_type = 0
        self.lines = None
        self.filtered_lines_num = 0

        self.traffic_output_image = None
        self.crosswalk_output_image = None

        # Load the Unet model
        self.model = mobilenet_unet(n_classes=2, input_height=224, input_width=224)
        self.model.load_weights('/home/innodriver/InnoDriver_ws/Unet_crosswalk_train/checkpoints/mobile_unet.66')


    # 기존 warp matrix를 로드하는 함수
    def load_warp_transform_matrix(self, directory, file_name='warp_matrix.pkl'):
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'rb') as f:
                matrix = pickle.load(f)
            rospy.loginfo("Loaded transform matrix from %s", file_path)
            return matrix
        except FileNotFoundError:
            raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

    def warp_transform(self, cv_image, width, height):
        top_view = cv2.warpPerspective(cv_image, self.warp_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return top_view

    def callback_raw1(self, data):
        if (rospy.Time.now() - self.last_1detection_time).to_sec() < 1:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 이미지 top_view로 변환
        top_view = self.warp_transform(cv_image, self.width, self.height)

        # # cv2.imshow로 결과 보여주기
        # cv2.imshow('Top View', top_view)
        # cv2.waitKey(10)

        # Unet 모델로 예측
        try:
            resized_image = cv2.resize(top_view, (224, 224))

            prediction = self.model.predict_segmentation(inp=resized_image)
        
            if prediction is None:
                rospy.logerr("Segmentation failed, returning")
                return

            prediction_resized = cv2.resize(prediction, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            # Combine masks with different colors
            colored_image = top_view.copy()
            colored_image[prediction_resized == 1] = [255, 0, 0]

            # cv2.imshow로 결과 보여주기
            cv2.imshow('Crosswalk Mask', colored_image)
            cv2.waitKey(10)

            # 횡단보도 존재 여부 판단
            crosswalk_pixels = np.sum(prediction_resized > 0.5)
            self.crosswalk_detected = crosswalk_pixels > 400

            if self.crosswalk_detected:
                crosswalk_coords = np.column_stack(np.where(prediction_resized > 0.5))
                avg_x, avg_y = np.mean(crosswalk_coords, axis=0)
                distance_pixels = ((avg_y - self.car_TR_center_y) ** 2 + (avg_x - self.car_center_x) ** 2) ** 0.5
                # distance_pixels = ((avg_y - self.car_TR_center_y) ** 2 ) ** 0.5
                self.crosswalk_distance = distance_pixels * self.resolution
            else:
                self.crosswalk_distance = -1
            print(self.crosswalk_detected, self.crosswalk_distance)
            self.last_1detection_time = rospy.Time.now()

            # 메시지 발행
            self.publish_info()

        except Exception as e:
            rospy.logerr(f"Error in callback_raw1: {str(e)}")

    def determine_traffic_light_color(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, light_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(light_mask)
        min_size = 800
        large_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_size]
        output_image = np.zeros_like(image)
        height, width = light_mask.shape
        roundness_scores = []

        for i in large_labels:
            x, y, w, h, area = stats[i]
            if y + h < height and x + w < width and x > 0 and y > 0:
                mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = contours[0]
                    perimeter = cv2.arcLength(cnt, True)
                    roundness = (4 * np.pi * area) / (perimeter ** 2)
                    roundness_scores.append((roundness, i))
        
        roundness_scores.sort(reverse=True)
        
        if roundness_scores:
            if roundness_scores[0][0] > 0.6:
                best_label = roundness_scores[0][1]
                best_mask = (labels == best_label).astype(np.uint8) * 255
                M = cv2.moments(best_mask)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                radius = int(np.sqrt(np.sum(best_mask) / (np.pi * 255)))
                rect_mask = np.zeros_like(light_mask)
                rect_width = radius * 6
                rect_height = radius * 2
                top_left = (max(cX - rect_width // 2, 0), max(cY - rect_height // 2, 0))
                bottom_right = (min(cX + rect_width // 2, width), min(cY + rect_height // 2, height))
                cv2.rectangle(rect_mask, top_left, bottom_right, 255, -1)
                inv_black_mask = cv2.bitwise_not(light_mask)
                # inv_rect_mask = cv2.bitwise_not(rect_mask)
                overlap = cv2.bitwise_and(inv_black_mask, rect_mask)
                overlap_coords = np.column_stack(np.where(overlap == 255))
                
                if len(overlap_coords) > 0:
                    avg_overlap_x = np.mean(overlap_coords[:, 1])
                    # print(avg_overlap_x, top_left[0] + rect_width * 2 / 5, bottom_right[0] - rect_width * 2 / 5)
                    if avg_overlap_x < top_left[0] + rect_width * 2 / 5:
                        self.light_type = 2  # Green light
                    elif avg_overlap_x > bottom_right[0] - rect_width * 2 / 5:
                        self.light_type = 1  # Red light
                    else:
                        self.light_type = 3  # Amber light
                    print(self.light_type)
                    self.traffic_light_detected = True

                return overlap, rect_mask, light_mask
        return None, None, None

    def callback_raw2(self, data):
        if (rospy.Time.now() - self.last_2detection_time).to_sec() < 0.5:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        overlap, rect_mask, light_mask = self.determine_traffic_light_color(cv_image)

        # if overlap is not None:
        #     cv2.imshow('Light Mask', light_mask)
        #     cv2.imshow('Rectangular Mask', rect_mask)
        #     cv2.imshow('Overlap', overlap)
        #     cv2.waitKey(10)

        self.last_2detection_time = rospy.Time.now()

        # 메시지 발행
        self.publish_info()

    def publish_info(self):
        info_msg = CrosswalkInfo()
        info_msg.crosswalk_detected = self.crosswalk_detected
        info_msg.crosswalk_distance = self.crosswalk_distance
        info_msg.traffic_light_detected = self.traffic_light_detected
        info_msg.light_type = self.light_type
        self.info_pub.publish(info_msg)

        # Reset the flags
        # self.crosswalk_detected = False
        # self.crosswalk_distance = -1
        # self.traffic_light_detected = False
        # self.light_type = 0
        # self.filtered_lines_num = 0

if __name__ == '__main__':
    try:
        yolo_node = CrossWalkAnalizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
