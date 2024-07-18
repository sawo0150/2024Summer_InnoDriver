#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import csv
import os
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.bridge = CvBridge()

        rospy.init_node('data_collector', anonymous=True)
        
        # Subscribers
        self.image_sub_raw = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_raw)
        self.steering_motor_sub = rospy.Subscriber("current_state", Float64MultiArray, self.motor_callback)
        self.joystick_sub = rospy.Subscriber("goal_state", Float64MultiArray, self.joystick_callback)

        self.rate = rospy.Rate(10)  # 10hz

        # Initialize state
        self.current_image = None
        self.current_steering_angle = 0
        self.current_motor_power = 0

        # Create offPolicyLearning directory if it doesn't exist
        self.data_dir = '/home/innodriver/InnoDriver_ws/src/dQN_modelTraining/offPolicyLearningData'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Create images directory inside offPolicyLearning directory
        self.images_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        # Initialize CSV file
        self.csv_file = self.generate_unique_filename()
        self.init_csv()

    def generate_unique_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.data_dir, f'training_data_{timestamp}.csv')


    def init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["image", "steering_angle", "motor_power", "goal_steering_angle", "goal_motor_power"])

    def callback_raw(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def motor_callback(self, msg):
        self.current_steering_angle = msg.data[0]
        self.current_motor_power = msg.data[1]

    def joystick_callback(self, msg):
        goal_steering_angle = msg.data[0]
        goal_motor_power = msg.data[1]
        self.save_data(goal_steering_angle, goal_motor_power)

    def save_data(self, goal_steering_angle, goal_motor_power):
        if self.current_image is not None:
            image_filename = os.path.join(self.images_dir, f'{rospy.Time.now()}.jpg')
            cv2.imwrite(image_filename, self.current_image)
            
            with open(self.csv_file, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([image_filename, self.current_steering_angle, self.current_motor_power, goal_steering_angle, goal_motor_power])

        # Sleep to maintain the rate
        self.rate.sleep()

if __name__ == '__main__':
    data_collector = DataCollector()
    rospy.spin()
