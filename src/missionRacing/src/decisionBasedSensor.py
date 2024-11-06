#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32, Bool, Int32, Float64MultiArray
from missionRacing.msg import LaneObstacleProbabilities, CrosswalkInfo
import time
import numpy as np

class DecisionNode:
    def __init__(self):
        rospy.init_node('decision_node', anonymous=True)

        rospy.Subscriber('lane_obstacle_probabilities', LaneObstacleProbabilities, self.lane_obstacle_callback)
        rospy.Subscriber('/crosswalk_info', CrosswalkInfo, self.crosswalk_callback)
        rospy.Subscriber("current_state", Float64MultiArray, self.motor_callback)
        rospy.Subscriber('calculated_goal_state', Float64MultiArray, self.goal_callback)

        self.goal_pub = rospy.Publisher("goal_state", Float64MultiArray, queue_size=2)
        self.flag_pub = rospy.Publisher('/decision_flag', Float32, queue_size=2)
        self.pub_lane_change = rospy.Publisher("goalLanesignal", Int32, queue_size=2)
        
        self.current_flag = 0.0
        self.instance_obstacle_detected = False
        self.obstacle_detected = False
        self.instance_crosswalk_detected = False
        self.crosswalk_detected = False

        self.red_light = False
        self.red_light_instance = False
        self.green_light = False
        self.green_light_instance = False

        self.last_obstacle_time = 0
        self.last_crosswalk_time = 0
        self.last_laneChange_time = 0
        self.last_greenLight_time = 0

        self.current_lane = 0
        self.obstacle_lane = 0
        self.goal_lane = 0

        self.obstacle_distance = float('inf')
        self.obstacle_sequence_detect = 0
        self.lane_angle = 0.0   

        self.current_steering_angle = 0
        self.current_motor_power = 0
        self.goal_steering_angle = 0
        self.goal_motor_power = 0
        self.crosswalk_control = 1

        self.maxAngle = 23.0

        self.isFirst = True
        
        self.rate = rospy.Rate(10)  # 10hz

    def motor_callback(self, msg):
        self.current_steering_angle = msg.data[0]
        self.current_motor_power = msg.data[1]

    def goal_callback(self, msg):
        self.goal_steering_angle = msg.data[0]
        self.goal_motor_power = msg.data[1]

    def lane_obstacle_callback(self, msg):
        # 현재 차선 확인
        lane1_prob = msg.lane_probabilities.data[0]
        lane2_prob = msg.lane_probabilities.data[1]
        if self.isFirst and (lane1_prob+lane2_prob)>0:
            self.isFirst = False
            if lane1_prob > lane2_prob:
                self.goal_lane = 2
            else:
                self.goal_lane = 2

        if lane1_prob > lane2_prob:
            self.current_lane = 1
        else:
            self.current_lane = 2

        # 장애물 정보 확인
        obstacle_lane1_probs = msg.obstacle_Lane1probabilities.data
        obstacle_lane2_probs = msg.obstacle_Lane2probabilities.data
        obstacle_distances = msg.obstacle_distances.data

        self.instance_obstacle_detected = False
        closest_obstacle_index = -1

        for i in range(len(obstacle_distances)):
            distance = obstacle_distances[i]
            obstacle_lane1_prob = obstacle_lane1_probs[i]
            obstacle_lane2_prob = obstacle_lane2_probs[i]

            # 장애물 존재 여부 확인
            if obstacle_lane1_prob > 0.5 or obstacle_lane2_prob > 0.5:
                self.instance_obstacle_detected = True

            # 차선 변경 조건 확인
            if distance < 2.5:  # 3m 안에 있는 장애물
                if self.goal_lane == 1 and obstacle_lane1_prob > 0.5:
                    if closest_obstacle_index == -1 or distance < obstacle_distances[closest_obstacle_index]:
                        closest_obstacle_index = i
                elif self.goal_lane == 2 and obstacle_lane2_prob > 0.5:
                    if closest_obstacle_index == -1 or distance < obstacle_distances[closest_obstacle_index]:
                        closest_obstacle_index = i
        
        lane_change_decision = 0  # 초기 값은 차선 변경 안함 (0)
        if time.time() - self.last_laneChange_time > 5 and not self.crosswalk_detected:
            if closest_obstacle_index != -1:
                if self.obstacle_sequence_detect>1:
                    self.obstacle_sequence_detect=0
                    if self.goal_lane == 1:
                        self.goal_lane = 2  # 오른쪽 차선 변경 (1)
                    elif self.goal_lane == 2:
                        self.goal_lane = 1  # 왼쪽 차선 변경 (2)
                    self.last_laneChange_time = time.time()
                else:
                    self.obstacle_sequence_detect+=1
            else:
                self.obstacle_sequence_detect=0
        # print("lane_obstacle_callBack 실행중")
        lane_number = Int32()
        lane_number.data = int(self.goal_lane)
        self.pub_lane_change.publish(lane_number)

    def crosswalk_callback(self, msg):
        self.crosswalk_detected = msg.crosswalk_detected
        self.crosswalk_distance = msg.crosswalk_distance
        self.traffic_light_detected = msg.traffic_light_detected
        self.red_light = (msg.light_type == 1)
        self.green_light = (msg.light_type == 2)
        print(self.red_light, self.green_light)
        # if self.crosswalk_detected or self.red_light or self.green_light:
            # self.last_crosswalk_time = time.time()

    def update_flag(self):
        current_time = time.time()
         
        if  current_time - self.last_greenLight_time < 1:
            self.current_flag = 1.0
            self.crosswalk_control = 1.0

        elif self.crosswalk_detected and self.green_light:
            self.last_greenLight_time = time.time()
            print("초록불... 횡단보도 앞 출발")
            self.current_flag = 1.0
            self.crosswalk_control = 1.0

        elif self.crosswalk_detected and self.crosswalk_distance <2.3:
            if self.red_light:
                self.last_crosswalk_time = current_time
                print("stop중... 횡단보도 앞")
                self.current_flag = 3.0
                self.crosswalk_control = 0
                # if self.crosswalk_distance <= 0.1:
                #     self.crosswalk_control = 0.0
                # elif self.crosswalk_distance <= 0.5:
                #     self.crosswalk_control = 0.5
                # else:
                #     self.crosswalk_control = 1.0
            else:
                self.last_crosswalk_time = current_time
                self.crosswalk_control = 0
        elif current_time-self.last_crosswalk_time<3:
            self.current_flag = 3.0
            self.crosswalk_control = 0


        # elif self.obstacle_detected or self.instance_obstacle_detected:
        #     self.current_flag = 2.0
        #     self.crosswalk_control = 1.0
        #     if self.instance_obstacle_detected:
        #         self.last_obstacle_time = current_time
        #         self.obstacle_detected = True
        #     elif current_time - self.last_obstacle_time > 3:
        #         self.obstacle_detected = False
        else:
            self.current_flag = 1.0
            self.crosswalk_control = 1.0

        self.flag_pub.publish(self.current_flag)

    def publishGoalState(self):

        # Create the goal_state message
        goal_pulse = (1-np.abs(self.current_steering_angle - self.goal_steering_angle)/2) * self.goal_motor_power *self.crosswalk_control
        msg = Float64MultiArray()
        msg.data = [self.goal_steering_angle, goal_pulse]

        # msg.data = [optimal_steering_angle, pulse / 255]
        # msg.data = [predicted_steering, 0]
        self.goal_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            self.update_flag()
            self.publishGoalState()
            self.rate.sleep()

if __name__ == '__main__':
    node = DecisionNode()
    node.run()
