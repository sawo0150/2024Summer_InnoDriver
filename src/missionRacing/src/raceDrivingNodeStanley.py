#!/usr/bin/env python3

import numpy as np
import math
import rospy
import cv2
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from missionRacing.msg import LaneWaypoints 
from cv_bridge import CvBridge

class StanleyController:
    def __init__(self):
        rospy.init_node('stanley_controller', anonymous=True)
        self.bridge = CvBridge()
        self.sub_lane_probabilities = rospy.Subscriber('lane_probabilities', Float32MultiArray, self.probabilities_callback)
        self.sub_waypoints = rospy.Subscriber('lane_waypoints', LaneWaypoints, self.waypoints_callback)
        self.pub_goal_state = rospy.Publisher('goal_state', Float64MultiArray, queue_size=2)

        self.width = 448
        self.height = 300
        self.carWidth = 0.42
        self.carHeigt = 0.54
        self.resolution = 17/1024   # 17m/4096 픽셀
        self.carCenterPointX = int(self.width / 2)
        self.carCenterPointY = int(self.height + self.carHeigt/(2*self.resolution))  # 차량의 하단 가장자리
        self.default_lane = None
        self.current_waypoints = None
        self.K = 0.005  # 경로 추종 감도

        self.angleMax = 23

        self.lane_selected = False
        self.base_lane = None
        self.waypoints = None
        self.rate = rospy.Rate(5)

    def probabilities_callback(self, msg):
        if not self.lane_selected:
            if msg.data[0] > msg.data[1]:
                self.base_lane = 'left'
            else:
                self.base_lane = 'right'
            self.lane_selected = True

    def waypoints_callback(self, msg):
        self.waypoints = msg

    def calculate_steering_and_pulse(self):
        if not self.waypoints or not self.base_lane:
            return

        if self.base_lane == 'left':
            lane_points = self.waypoints.left_points
        else:
            lane_points = self.waypoints.right_points

        lane_points = np.array(lane_points).reshape(-1, 2)
        distances = np.linalg.norm(lane_points - np.array([self.carCenterPointX, self.carCenterPointY]), axis=1)
        points_in_range = lane_points[distances <= 1.5 / (self.resolution)]

        if len(points_in_range) < 2:
            steering = np.radians(self.angleMax) if self.base_lane == 'left' else np.radians(-self.angleMax)
            pulse = 150
        else:
            start_point = points_in_range[0]
            end_point = points_in_range[-1]
            if start_point[0] == end_point[0]:  # y축과 평행한 경우
                reference_line = (np.inf, start_point[0])
            else:
                reference_line = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
                
            steering = self.stanley_algorithm(reference_line)
            steering = np.clip(np.degrees(steering), -self.angleMax, self.angleMax)
            pulse_range = 230 - 150
            pulse = 230 - (abs(steering)/self.angleMax * pulse_range)
        
        # return steering/self.angleMax, pulse
        return steering, pulse

    def stanley_algorithm(self, reference_line):
        # Simplified Stanley algorithm for steering calculation
        k = 0.5  # Control gain
        yaw = np.pi/2  # Assume car is aligned with the y-axis in image coordinates
        
        if np.isinf(reference_line[0]):  # y축과 평행한 경우
            crosstrack_error = reference_line[1] - self.carCenterPointX
            heading_error = 0  # 차량의 heading과 일치하므로 heading error는 0
        else:
            yaw_path = np.arctan(reference_line[0])
            if yaw_path<0:
                yaw_path+=np.pi
            heading_error = yaw_path - yaw

            # 직선과 점 사이의 최단 거리 계산
            a = reference_line[0]
            b = -1
            c = reference_line[1]

            distance_numerator = abs(a * self.carCenterPointX + b * self.carCenterPointY + c)
            distance_denominator = np.sqrt(a**2 + b**2)
            crosstrack_error = distance_numerator / distance_denominator

            # 방향을 고려하여 crosstrack error의 부호 결정
            if (self.carCenterPointY - np.polyval(reference_line, self.carCenterPointX)) > 0:
                crosstrack_error = -crosstrack_error


        steering = heading_error + np.arctan2(k * crosstrack_error, 1)

        # 조향각이 [-π/2, π/2] 범위 내에 있도록 조정
        if steering > np.pi / 2:
            steering -= np.pi
        elif steering < -np.pi / 2:
            steering += np.pi
        
        print(np.degrees(heading_error), crosstrack_error)  # Debugging print
        return steering

    def run(self):
        while not rospy.is_shutdown():
            result = self.calculate_steering_and_pulse()
            if result:
                steering, pulse = result
                msg = Float64MultiArray()
                # msg.data = [steering, pulse / 255.0]
                msg.data = [steering, pulse]
                self.pub_goal_state.publish(msg)
            self.rate.sleep()

if __name__ == '__main__':
    controller = StanleyController()
    controller.run()