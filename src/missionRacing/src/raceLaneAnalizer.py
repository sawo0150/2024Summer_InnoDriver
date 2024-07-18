#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import os
import pickle
import time
from missionRacing.msg import LaneWaypoints, LaneObstacleProbabilities

class LaneAnalizer:
    def __init__(self):
        rospy.init_node('lane_analizer', anonymous=True)
        self.bridge = CvBridge()
        self.pub1 = rospy.Publisher('Analized_image', Image, queue_size=2)
        # Publisher
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=2)
        rospy.Subscriber('ultraDistance', Float64MultiArray, self.ultrasonic_callback)

        # Publisher 초기화 부분
        self.pub_lane_obstacle_probabilities = rospy.Publisher('lane_obstacle_probabilities', LaneObstacleProbabilities, queue_size=2)
        self.pub_waypoints = rospy.Publisher('lane_waypoints', LaneWaypoints, queue_size=2)


        self.rate = rospy.Rate(5)
        self.width = 448
        self.height = 300
        self.resolution = 17/1024   # 17m/4096 픽셀
        self.laneWidth = 0.90       #원래는 0.85m지만, 조금 조정?

        self.carWidth = 0.5
        self.carBoxHeight = 0.4
        # 차량의 위치 및 크기 정의
        self.car_center_x = int(self.width / 2)
        self.car_center_y = self.height - int(self.carBoxHeight / self.resolution/2)
        self.car_width = int(self.carWidth /self.resolution)
        self.car_height = int(self.carBoxHeight /self.resolution)

        # 초음파 센서 관련 초기화
        self.num_sensors = 4  # 예제 값, 실제 센서 개수로 변경
        self.sensor_positions = [(0.1, 0.2), (-0.1, 0.2)]  # 예제 값, 실제 센서 위치로 변경
        self.sensor_angles = [0, np.pi / 4, -np.pi / 4, np.pi / 2]  # 예제 값, 실제 센서 각도로 변경
        self.distances = [0] * self.num_sensors

        # 차량 박스 생성
        self.car_box = np.zeros((self.height, self.width), dtype=np.uint8)
        self.top_left_x = max(self.car_center_x - self.car_width // 2, 0)
        self.top_left_y = max(self.car_center_y - self.car_height // 2, 0)
        self.bottom_right_x = min(self.car_center_x + self.car_width // 2, self.width)
        self.bottom_right_y = min(self.car_center_y + self.car_height // 2, self.height)

        self.car_box[self.top_left_y:self.bottom_right_y, self.top_left_x:self.bottom_right_x] = 1
        # Directory and file settings
        self.directory = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
        self.warp_matrix = self.load_warp_transform_matrix(self.directory)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = self.warp_transform(cv_image,self.width, self.height)
            lane1_mask, lane2_mask, corners = self.create_lane_masks(cv_image)
            
            # Combine masks with different colors
            colored_image = cv_image.copy()
            colored_image[lane1_mask == 1] = [255, 0, 0]  # Blue for 1st lane
            colored_image[lane2_mask == 1] = [0, 0, 255]  # Red for 2nd lane

            lPoint, rPoint = self.calculate_waypoints(corners)
            colored_image = self.draw_waypoints_on_mask(colored_image, lPoint, rPoint)
            lane1CP, lane2CP = self.calculate_carLane_probabilities(lane1_mask, lane2_mask)
            # print(time.time()-startTime)
            # print(lane1P, lane2P)
            lane1OP, lane2OP, distanceObs = self.calculate_obstacle_probabilities(lane1_mask, lane2_mask)
            # Convert back to ROS Image message and publish
            transformed_msg = self.bridge.cv2_to_imgmsg(colored_image, "bgr8")
            self.pub1.publish(transformed_msg)

            # 확률 데이터 발행 부분
            lane_obstacle_msg = LaneObstacleProbabilities()
            
            lane_probabilities = Float32MultiArray()
            lane_probabilities.data = [lane1CP, lane2CP]
            lane_obstacle_msg.lane_probabilities = lane_probabilities
            
            obstacleLane1_probabilities = Float32MultiArray()
            obstacleLane2_probabilities = Float32MultiArray()
            obstacleLane1_probabilities.data = lane1OP
            obstacleLane2_probabilities.data = lane2OP
            lane_obstacle_msg.obstacle_Lane1probabilities = obstacleLane1_probabilities
            lane_obstacle_msg.obstacle_Lane2probabilities = obstacleLane2_probabilities
            
            obstacle_distances = Float32MultiArray()
            obstacle_distances.data = distanceObs
            lane_obstacle_msg.obstacle_distances = obstacle_distances

            self.pub_lane_obstacle_probabilities.publish(lane_obstacle_msg)

            # wayPoint 발행 부분
            waypoints_msg = LaneWaypoints()
            waypoints_msg.left_points = [float(val) for point in lPoint for val in point]  # Flatten and convert to float
            waypoints_msg.right_points = [float(val) for point in rPoint for val in point]  # Flatten and convert to float
            self.pub_waypoints.publish(waypoints_msg)

        except CvBridgeError as e:
            rospy.logerr("Failed to convert image: %s", e)

    def ultrasonic_callback(self, msg):
        self.distances = msg.data

    def count_nonzero_neighbors(self, mask, point, radius=5):
        x, y = point
        region = mask[max(y-radius, 0):min(y+radius, mask.shape[0]), max(x-radius, 0):min(x+radius, mask.shape[1])]
        return np.count_nonzero(region)

    def draw_half_lines(self, corners, image_shape):
        # 첫 번째 반직선
        start_point = corners[1]
        end_point = corners[0]
        delta_x = start_point[0] - end_point[0]
        delta_y = start_point[1] - end_point[1]
        
        if delta_y == 0:
            start_y = start_point[1]
            start_x = 0 if delta_x > 0 else image_shape[1]
        else:
            start_y = image_shape[0]
            angle = np.arctan2(delta_y, delta_x)
            start_x = int(start_point[0] + (start_y - start_point[1]) / np.tan(angle))
        
        # 두 번째 반직선
        start_point = corners[-2]
        end_point = corners[-1]
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]

        if delta_y == 0:
            end_y = end_point[1]
            end_x = image_shape[1] if delta_x > 0 else 0
        else:
            end_y = 0
            angle = np.arctan2(delta_y, delta_x)
            end_x = int(end_point[0] + (end_y - end_point[1]) / np.tan(angle))

        return (start_x, start_y), (end_x, end_y)

    def create_lane_masks(self, image):
        # cv2.imshow('Road Mask', image)
        # cv2.waitKey(1000)
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower_black = np.array([0, 0, 0])
        # upper_black = np.array([255, 255, 150])
        # mask = cv2.inRange(hsv, lower_black, upper_black)
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # mask = cv2.erode(mask, kernel1, iterations=1)
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2]
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #  # Otsu의 이진화를 사용하여 이진화
        # _, mask = cv2.threshold(
        #     gray, 
        #     0, 
        #     255, 
        #     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        # )
        # HSV로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 그린 계열 픽셀의 범위 설정
        lower_green = np.array([35, 40, 100])
        upper_green = np.array([85, 255, 255])
        # 그린 계열 픽셀 마스크 생성
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        # 그린 계열 픽셀의 value 값이 일정 이상인 경우 하얀색으로 변환
        hsv[green_mask > 0] = [0, 0, 255]
        # BGR로 다시 변환
        transformedImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # 그레이스케일로 변환
        gray = cv2.cvtColor(transformedImage, cv2.COLOR_BGR2GRAY)
        # 임계값을 사용하여 이진화
        _, mask = cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV)
        # 마스크를 uint8 타입으로 변환
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cv2.imshow('Road Mask', mask)
        cv2.waitKey(10)

        # 각 레이블과 car_box의 교차 영역을 계산합니다.
        max_intersection = 0
        max_label = 0
        for label in range(1, num_labels):
            label_mask = (labels == label).astype(np.uint8)
            intersection = np.sum(label_mask & self.car_box)  # 교차 영역의 픽셀 수를 계산
            if intersection > max_intersection:
                max_intersection = intersection
                max_label = label

        road_mask = np.zeros_like(labels)
        road_mask[labels == max_label] = 1
        road_mask = road_mask.astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        road_mask = cv2.dilate(road_mask.astype(np.uint8), kernel, iterations=1)
        
        # cv2.imshow('Road Mask', road_mask*255)
        # cv2.waitKey(1000)
        # 중앙선 찾기
        # central_mask = 1-cv2.erode(road_mask, kernel, iterations=1)
        central_mask = 1-road_mask
        

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(central_mask, connectivity=8)
        # labeled_image = cv2.cvtColor(central_mask * 255, cv2.COLOR_GRAY2BGR)
        # for label in range(1, num_labels):
        #     labeled_image[labels == label] = [int(j) for j in np.random.randint(0, 255, size=3)]

        # cv2.imshow('labeled Mask', labeled_image)
        # cv2.waitKey(1000)

        central_line_labels = []
        # print(stats)
        # print(centroids)
        for i in range(1, num_labels):
            if (stats[i, cv2.CC_STAT_WIDTH] < 50 or stats[i, cv2.CC_STAT_HEIGHT] < 50) and stats[i, 4]>30:
                if not ((self.width/2-5<centroids[i,0] or centroids[i,0]<self.width/2+5) and self.height-5<centroids[i,1]):
                    x, y, w, h, area = stats[i]
                    if (area/(w*h)) > 0.1:
                        central_line_labels.append(i)
        
        central_mask = np.zeros_like(road_mask)
        for label in central_line_labels:
            central_mask[labels == label] = 1
        
        # cv2.imshow('central_mask ', central_mask*255)
        # cv2.waitKey(1000)

        # Find the corners of the central lane
        corners = []
        for label in central_line_labels:
            x, y, w, h, area = stats[label]
            potential_corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            neighbor_counts = [self.count_nonzero_neighbors(central_mask, point) for point in potential_corners]
            # print(potential_corners)
            # 가장 큰 neighbor count를 가진 point의 index 계산
            max_index = neighbor_counts.index(max(neighbor_counts))
            point1 = potential_corners[max_index]
            point2 = potential_corners[(max_index + 2) % 4]
            
            # 두 점을 3등분하여 4개의 점 추가
            for i in range(4):
                new_point = (
                    point1[0] + (point2[0] - point1[0]) * i // 3,
                    point1[1] + (point2[1] - point1[1]) * i // 3
                )
                corners.append(new_point)

        if not corners:
            rospy.logerr("No corners found for the central lane")
            return np.zeros_like(road_mask), np.zeros_like(road_mask), []
        
        # print(corners)
        max_y_point = max(corners, key=lambda p: p[1])
        # Sort corners by y value, and then by distance from the first corner
        corners = sorted(corners, key=lambda p: (np.hypot(p[0] - max_y_point[0], p[1] - max_y_point[1])))
        
        # Draw the central lane mask
        for i in range(len(corners) - 1):
            cv2.line(road_mask, corners[i], corners[i + 1], 0, 2)

        # 첫 번째 라인 그리기
        startPoint, endPoint  = self.draw_half_lines(corners, image.shape)
        # print(startPoint, endPoint)
        # print(corners)
        cv2.line(road_mask, startPoint, tuple(corners[0]), 0, 2)
        cv2.line(road_mask, tuple(corners[-1]), endPoint, 0, 2)

        # cv2.imshow('road_mask ', road_mask*255)
        # cv2.waitKey(1000)
        # Find contours on the modified road_mask
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ref_point = (self.width / 2, self.height)

        # 평균점과 기준점 사이의 각도 계산 후 정렬
        def calculate_angle(contour):
            M = cv2.moments(contour)
            angle = np.arctan2(np.mean(contour[:,:,1]) - ref_point[1], np.mean(contour[:,:,0]) - ref_point[0])
            if angle < 0:
                angle += np.pi
            return angle
        # Get the two largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        contours = sorted(contours, key=calculate_angle)  # Sort by the mean x value of contours

        if len(contours)<2:
            lane1_mask = np.zeros_like(road_mask)
            cv2.fillPoly(lane1_mask, [contours[0]], 1)
            return lane1_mask
        # Create lane masks
        lane1_mask = np.zeros_like(road_mask)
        lane2_mask = np.zeros_like(road_mask)
        
        cv2.fillPoly(lane1_mask, [contours[0]], 1)
        cv2.fillPoly(lane2_mask, [contours[1]], 1)
        
        return lane1_mask, lane2_mask, corners

    def calculate_waypoints(self, corners):
        rightLanewayPoint = []
        leftLanewayPoint = []

        for i in range(len(corners) - 1):
            x1, y1 = corners[i]
            x2, y2 = corners[i + 1]
            
            # 중점 계산
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # 선분의 방향 벡터
            dx = x2 - x1
            dy = y2 - y1

            # 수직 벡터
            perp_dx = -dy
            perp_dy = dx

            # 수직 벡터의 크기를 laneWidth로 조정
            length = np.sqrt(perp_dx**2 + perp_dy**2) 
            
            if length == 0:
                continue  # length가 0인 경우, 다음 반복으로 넘어감
            unit_perp_dx = (perp_dx / length) * (self.laneWidth/self.resolution / 2)
            unit_perp_dy = (perp_dy / length) * (self.laneWidth/self.resolution / 2)

            # 오른쪽 점과 왼쪽 점 계산
            right_point = (int(mid_x + unit_perp_dx), int(mid_y + unit_perp_dy))
            left_point = (int(mid_x - unit_perp_dx), int(mid_y - unit_perp_dy))

            rightLanewayPoint.append(right_point)
            leftLanewayPoint.append(left_point)

        return leftLanewayPoint, rightLanewayPoint
    def calculate_carLane_probabilities(self, lane1_mask, lane2_mask):

        # 차량 박스와 각 레인 마스크의 겹치는 영역 계산
        intersection_with_lane1 = np.sum(np.logical_and(self.car_box, lane1_mask))
        intersection_with_lane2 = np.sum(np.logical_and(self.car_box, lane2_mask))

        # 차량 박스 내 총 픽셀 수
        total_car_pixels = np.sum(self.car_box)
        # cv2.imshow('carBox Mask', car_box*255)
        # cv2.waitKey(10000)

        # 각 레인에 대한 확률 계산
        lane1_probability = intersection_with_lane1 / total_car_pixels if total_car_pixels > 0 else 0
        lane2_probability = intersection_with_lane2 / total_car_pixels if total_car_pixels > 0 else 0

        return lane1_probability, lane2_probability


    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()


    def draw_waypoints_on_mask(self, mask, leftLanewayPoint, rightLanewayPoint):
        # Draw right lane waypoints and lines
        for i in range(len(rightLanewayPoint) - 1):
            cv2.circle(mask, rightLanewayPoint[i], 5, (255, 0, 0), -1)  # Blue color, filled circle
            cv2.line(mask, rightLanewayPoint[i], rightLanewayPoint[i + 1], (255, 0, 0), 2)  # Blue color line

        # Draw left lane waypoints and lines
        for i in range(len(leftLanewayPoint) - 1):
            cv2.circle(mask, leftLanewayPoint[i], 5, (0, 0, 255), -1)  # Red color, filled circle
            cv2.line(mask, leftLanewayPoint[i], leftLanewayPoint[i + 1], (0, 0, 255), 2)  # Red color line

        # Draw the last points
        if rightLanewayPoint:
            cv2.circle(mask, rightLanewayPoint[-1], 5, (255, 0, 0), -1)
        if leftLanewayPoint:
            cv2.circle(mask, leftLanewayPoint[-1], 5, (0, 0, 255), -1)

        return mask
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

    def warp_transform(self, cv_image,width, height):
        top_view = cv2.warpPerspective(cv_image, self.warp_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return top_view

    def calculate_obstacle_probabilities(self, lane1_mask, lane2_mask):
        # 장애물 위치를 계산하고 Gaussian 분포로 Mask 생성
        obstacle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        sigma = 0.5 / self.resolution  # 1m를 픽셀로 변환한 값

        y, x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')

        lane1_probabilities = []
        lane2_probabilities = []
        distances_within_range = []
        for i in range(self.num_sensors):
            distance = self.distances[i]
            if distance > 4000:
                lane1_probabilities.append(0)
                lane2_probabilities.append(0)
                distances_within_range.append(distance)
                continue  # 4m 이상의 장애물은 무시
            
            sensor_x, sensor_y = self.sensor_positions[i]
            angle = self.sensor_angles[i]
            
            # 장애물의 이미지 상 위치 계산
            obstacle_x = sensor_x + distance * np.cos(angle)
            obstacle_y = sensor_y + distance * np.sin(angle)
            obstacle_x = int(obstacle_x / self.resolution)
            obstacle_y = int(obstacle_y / self.resolution)

            # Gaussian 분포로 Mask 생성
            d = np.sqrt((x - obstacle_x) ** 2 + (y - obstacle_y) ** 2)
            gaussian_mask = np.exp(-(d ** 2) / (2 * sigma ** 2))
            obstacle_mask = np.maximum(obstacle_mask, gaussian_mask * 255)
            
            cv2.imshow('Obstacle Mask', obstacle_mask)
            cv2.waitKey(100)
            # Lane mask와 내적하여 확률 계산
            lane1_intersection = np.sum(lane1_mask * obstacle_mask)
            lane2_intersection = np.sum(lane2_mask * obstacle_mask)
            total_intersection = np.sum(obstacle_mask)

            if total_intersection == 0:
                lane1_probability = 0
                lane2_probability = 0
            else:
                lane1_probability = lane1_intersection / total_intersection
                lane2_probability = lane2_intersection / total_intersection

            lane1_probabilities.append(lane1_probability)
            lane2_probabilities.append(lane2_probability)
            distances_within_range.append(distance)

        return lane1_probabilities, lane2_probabilities, distances_within_range

if __name__ == '__main__':
    try:
        lane_masker = LaneAnalizer()
        lane_masker.run()

        # # 이미지 파일 경로
        # image_path = "/home/innodriver/InnoDriver_ws/src/missionRacing/src/1720785185578291177.jpg"

        # # 이미지 읽기
        # image = cv2.imread(image_path)
        # startTime = time.time()
        # transformedImage = lane_masker.warp_transform(image,lane_masker.width, lane_masker.height)
        # # print(time.time()-startTime)
        # lane1_mask, lane2_mask, corners = lane_masker.create_lane_masks(transformedImage)
        
        # # Combine masks with different colors
        # colored_image = transformedImage.copy()
        # colored_image[lane1_mask == 1] = [128, 0, 0]  # Blue for 1st lane
        # colored_image[lane2_mask == 1] = [0, 0, 128]  # Red for 2nd lane
        # cv2.imshow('Road Mask', colored_image)
        # lPoint, rPoint = lane_masker.calculate_waypoints(corners)
        # colored_image = lane_masker.draw_waypoints_on_mask(colored_image, lPoint, rPoint)
        # lane1P, lane2P = lane_masker.calculate_lane_probabilities(lane1_mask, lane2_mask)
        # print(time.time()-startTime)
        # print(lane1P, lane2P)
        # cv2.imshow('waypoint Mask', colored_image)
        # cv2.waitKey(10000)
    except rospy.ROSInterruptException:
        pass
