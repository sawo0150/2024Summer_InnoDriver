#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class LidarParking:
    def __init__(self):
        rospy.init_node('lidar_parking_node')
        self.resolution = 17.0 / 1024.0  # m/pixel
        self.image_size = 512  # Assuming a square image
        self.radius = self.image_size // 2
        self.center = (self.radius, self.radius)
        self.mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    def scan_callback(self, scan_data):
        self.mask.fill(0)  # Clear the mask
        angle_min = scan_data.angle_min
        angle_increment = scan_data.angle_increment
        
        for i, range in enumerate(scan_data.ranges):
            if range == float('inf') or range == float('-inf') or range == 0.0:
                continue
            angle = angle_min + i * angle_increment
            x = int(self.radius + (range / self.resolution) * np.cos(angle))
            y = int(self.radius + (range / self.resolution) * np.sin(angle))
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                self.mask = cv2.circle(self.mask, (x, y), 10, 255, -1)
        
        free_spaces = self.find_parking_spaces(self.mask)
        if free_spaces:
            entry_point, steering_angle = self.calculate_parking_trajectory(free_spaces[0])
            self.park_car(entry_point, steering_angle)
        
        cv2.imshow("Lidar Mask", self.mask)
        cv2.waitKey(1)
    
    def find_parking_spaces(self, mask):
        # Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        free_spaces = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust area threshold based on your need
                rect = self.get_rotated_rectangle(contour)
                if rect is not None:
                    free_spaces.append(rect)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(mask, [box], 0, (0, 255, 0), 2)
        
        return free_spaces
    
    def get_rotated_rectangle(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        for dx in range(-5, 6):
            for dy in range(-5, 6):
                for angle in np.linspace(-10, 10, 21):
                    M = cv2.getRotationMatrix2D((self.radius + dx, self.radius + dy), angle, 1.0)
                    rotated_box = cv2.transform(np.array([box]), M)[0]
                    if self.is_valid_parking_space(rotated_box):
                        return rect
        return None
    
    def is_valid_parking_space(self, box):
        # 여기서 박스가 유효한 주차 공간인지 확인하는 로직을 추가
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        aspect_ratio = width / height if height > 0 else 0
        return 0.8 < aspect_ratio < 1.2
    
    def calculate_parking_trajectory(self, rect):
        center, size, angle = rect
        entry_point = (int(center[0]), int(center[1]))
        # 주차를 위한 조향 각도 계산 로직
        steering_angle = 0  # Placeholder, 실제 차량 역학을 고려한 계산 필요
        return entry_point, steering_angle
    
    def park_car(self, entry_point, steering_angle):
        # 차량을 주차하기 위한 명령 발행
        move_cmd = Twist()
        move_cmd.linear.x = 0.2  # 앞으로 이동
        move_cmd.angular.z = steering_angle
        self.cmd_pub.publish(move_cmd)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    lidar_parking = LidarParking()
    lidar_parking.run()
