#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pickle
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

    def keep_only_white_hsv(self, img):
        # 이미지를 HSV 색 공간으로 변환
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 흰색 범위 설정 (HSV) - 그림자까지 포함하도록 임계값 조정
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 30, 255], dtype=np.uint8)

        # 흰색 범위 내의 픽셀을 흰색으로 설정
        mask = cv2.inRange(imgHSV, lower_white, upper_white)

        # 마스크를 통해 흰색 픽셀만 남기고 나머지는 검정색으로 변경
        result = cv2.bitwise_and(img, img, mask=mask)

        # 추가적인 마스크 처리를 통해 작은 잡음을 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)

        return result

    def canny(self, image):
        return cv2.Canny(image, 50, 150)

    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def region_of_interest(self, image):
        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros_like(image)
        top_left = (0, 300)
        bottom_right = (width, 700)

        cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)
        return cv2.bitwise_and(image, mask)

    def cross_walk_detect(self, lines):
        detection = False
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if not (-20 <= angle <= 20):
                    filtered_lines.append(line)

        filtered_lines_num = len(filtered_lines)
        if filtered_lines_num > 23:
            detection = True
        print(filtered_lines_num)
        print(detection)
        
        return detection, filtered_lines

    def callback_raw1(self, data):
        if (rospy.Time.now() - self.last_1detection_time).to_sec() < 0.5:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # OpenCV 기반 횡단보도 감지
        lane_image = np.copy(cv_image)
        ThreshImg = self.keep_only_white_hsv(lane_image)
        canny_image = self.canny(ThreshImg)
        cropped_image = self.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=10)
        self.crosswalk_detected, filtered_lines = self.cross_walk_detect(lines)
        line_image = self.display_lines(lane_image, filtered_lines)
        self.crosswalk_output_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

        # cv2.imshow('crosswalk_output_image', self.crosswalk_output_image)
        # cv2.waitKey(10)
        self.last_1detection_time = rospy.Time.now()

        # 메시지 발행
        self.publish_info()

    def determine_traffic_light_color(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, light_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(light_mask)
        min_size = 1000
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
