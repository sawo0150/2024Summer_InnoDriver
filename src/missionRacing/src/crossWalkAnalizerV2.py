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
        self.laneWidth = 0.90       #원래는 0.85m지만, 조금 조정?

        self.carWidth = 0.42
        self.carBoxHeight = 0.54
        # 차량의 위치 및 크기 정의
        self.car_box_dist = 0.61
        self.car_center_x = int(self.width / 2)
        self.car_center_y = self.height - int(self.car_box_dist / self.resolution/2)
        self.car_TR_center_y = self.height + int(self.car_box_dist / self.resolution/2)
        self.car_width = int(self.carWidth /self.resolution)
        self.car_height = int(self.carBoxHeight /self.resolution)
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
        canny = cv2.Canny(image, 50, 150)
        return canny

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
        top_left = (0, 200)
        bottom_right = (width, 700)

        cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

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
        if filtered_lines_num > 30:
            detection = True
        print(filtered_lines_num)
        
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
        # line_image = self.display_lines(lane_image, lines)
        line_image = self.display_lines(lane_image, filtered_lines)
        self.crosswalk_output_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

        cv2.imshow('Crosswalk Detection', self.crosswalk_output_image)
        cv2.waitKey(10)

        self.last_1detection_time = rospy.Time.now()

        # 메시지 발행
        self.publish_info()

    def process_traffic_light(self, image):
        # OpenCV 기반 신호등 감지 (임의의 예제 코드, 실제로는 신호등을 감지하는 코드 필요)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=30)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                # 임의의 조건: 원의 중심이 특정 영역에 있는 경우 신호등을 감지했다고 가정
                if y < image.shape[0] // 2:
                    self.light_type = 2  # Green
                else:
                    self.light_type = 1  # Red
                self.traffic_light_detected = True

        return image

    def callback_raw2(self, data):
        if (rospy.Time.now() - self.last_2detection_time).to_sec() < 0.5:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # OpenCV 기반 신호등 감지
        self.traffic_output_image = self.process_traffic_light(cv_image)

        # cv2.imshow('Traffic Light Detection', self.traffic_output_image)
        # cv2.waitKey(10)

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
        self.crosswalk_detected = False
        self.crosswalk_distance = -1
        self.traffic_light_detected = False
        self.light_type = 0
        self.filtered_lines_num = 0

        # if self.traffic_output_image is not None:
        #     cv2.imshow('Traffic Light Detection', self.traffic_output_image)
        #     cv2.waitKey(1)
        # if self.crosswalk_output_image is not None:
        #     cv2.imshow('Crosswalk Detection', self.crosswalk_output_image)
        

if __name__ == '__main__':
    try:
        yolo_node = CrossWalkAnalizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
