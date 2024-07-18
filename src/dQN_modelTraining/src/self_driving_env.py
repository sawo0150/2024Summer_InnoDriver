#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2



class SelfDrivingEnv:
    def __init__(self):
        self.bridge = CvBridge()

        rospy.init_node('self_driving_env', anonymous=True)
        # Subscribers
        self.image_sub_raw = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_raw)
        # self.image_sub_compressed = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback_compressed)
        self.steering_motor_sub = rospy.Subscriber("current_state", Float64MultiArray, self.motor_callback)

        # Publishers
        self.pub = rospy.Publisher('goal_state', Float64MultiArray, queue_size=10)
        
        self.rate = rospy.Rate(10)  # 10hz

        # Initialize state
        self.current_image = None
        self.current_steering_angle = 0
        self.current_motor_power = 0
        self.goal_motor_power = 0.5
        self.goal_steering_angle = 0

        self.reset()

    def callback_raw(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    # def callback_compressed(self, data):
    #     try:
    #         np_arr = np.fromstring(data.data, np.uint8)
    #         self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #     except CvBridgeError as e:
    #         print(e)

    def motor_callback(self, msg):
        self.current_steering_angle = msg.data[0]
        self.current_motor_power = msg.data[1]

    def reset(self):
        self.goal_steering_angle = 0
        self.goal_motor_power = 0.5
        self.steering_angle = 0
        return self._get_state()

    def step(self, action):
        self.goal_steering_angle = action[0]
        self.goal_motor_power = action[1]

        # Publish the action to the robot
        msg = Float64MultiArray()
        msg.data = [self.goal_steering_angle, self.goal_motor_power]
        self.pub.publish(msg)
        self.rate.sleep()

        reward = self._calculate_reward()

        done = self._is_done()

        return self._get_state(), reward, done

    def _get_state(self):
        if self.current_image is not None:
            resized_image = cv2.resize(self.current_image, (224, 224))
            # Normalize the image if necessary (values between 0 and 1)
            processed_image = resized_image / 255.0
        else:
            processed_image = np.zeros((224, 224, 3))

        return processed_image, self.current_steering_angle, self.current_motor_power
   
    def calculate_reward(self):
        # 차선 검출
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
        
        if lines is None:
            return -1  # 차선이 검출되지 않으면 보상 낮음
        
        lane_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            lane_angles.append(np.degrees(angle))
        
        # 차선 각도 평균과 자동차 각도 비교
        lane_angle = np.mean(lane_angles)
        angle_diff = abs(lane_angle - np.pi/2)
        
        # 차가 두 차선 사이에 잘 있는지 평가
        lanes_centered = any((self.current_image.shape[1] // 2) in (x1, x2) for line in lines for x1, _, x2, _ in line)
        
        reward = 1.0 - (angle_diff / 90)  # 각도 차이로 보상 계산
        reward += 0.5 if lanes_centered else -0.5  # 차선 중심에 대한 보상
        
        return reward

if __name__ == '__main__':
    env = SelfDrivingEnv()
    rospy.spin()