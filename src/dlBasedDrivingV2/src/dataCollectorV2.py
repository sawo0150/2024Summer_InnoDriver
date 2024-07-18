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
import pickle

class DataCollector:
    def __init__(self):
        self.bridge = CvBridge()

        rospy.init_node('data_collector', anonymous=True)
        
        # Subscribers
        self.image_sub_raw = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_raw)
        self.steering_motor_sub = rospy.Subscriber("current_state", Float64MultiArray, self.motor_callback)
        self.joystick_sub = rospy.Subscriber("goal_state", Float64MultiArray, self.joystick_callback)

        self.rate = rospy.Rate(2)  # 2Hz

        # Initialize state
        self.current_image = None
        self.current_steering_angle = 0
        self.current_motor_power = 0

        # Load warp matrix
        directory = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
        self.warp_matrix = self.load_warp_transform_matrix(directory)
        self.width = 448
        self.height = 300

        # Create Data directory if it doesn't exist
        self.data_base_dir = '/home/innodriver/InnoDriver_ws/Data'
        if not os.path.exists(self.data_base_dir):
            os.makedirs(self.data_base_dir)

        # Create unique directories for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.data_base_dir, timestamp)
        self.images_dir = os.path.join(self.run_dir, 'images')
        os.makedirs(self.images_dir)
        
        self.csv_dir1 = os.path.join(self.run_dir, 'csv1')
        if not os.path.exists(self.csv_dir1):
            os.makedirs(self.csv_dir1)
        self.csv_file1 = os.path.join(self.csv_dir1, f'training_data_{timestamp}.csv')

        self.csv_dir2 = os.path.join(self.run_dir, 'csv2')
        if not os.path.exists(self.csv_dir2):
            os.makedirs(self.csv_dir2)
        self.csv_file2 = os.path.join(self.csv_dir2, f'training_data_ext_{timestamp}.csv')

        self.transformed_images_dir = os.path.join(self.run_dir, 'transformed_images')
        if not os.path.exists(self.transformed_images_dir):
            os.makedirs(self.transformed_images_dir)

        # Initialize CSV files
        self.init_csv()

    def load_warp_transform_matrix(self, directory, file_name='warp_matrix.pkl'):
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'rb') as f:
                matrix = pickle.load(f)
            rospy.loginfo("Loaded transform matrix from %s", file_path)
            return matrix
        except FileNotFoundError:
            rospy.logerr("Transform matrix file not found at %s. Please generate it first.", file_path)
            rospy.signal_shutdown("Transform matrix file not found.")
        except Exception as e:
            rospy.logerr("Failed to load transform matrix: %s", e)
            rospy.signal_shutdown("Failed to load transform matrix.")

    def init_csv(self):
        try:
            # CSV file 1
            with open(self.csv_file1, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["image", "steering_angle", "motor_power", "goal_steering_angle", "goal_motor_power"])

            # CSV file 2
            with open(self.csv_file2, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(["image", "-2", "-1", "0", "1", "2"])
        except Exception as e:
            rospy.logerr("Failed to initialize CSV files: %s", e)
            rospy.signal_shutdown("Failed to initialize CSV files.")

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

    def warp_transform(self, cv_image):
        top_view = cv2.warpPerspective(cv_image, self.warp_matrix, (self.width, self.height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return top_view
    
    def save_data(self, goal_steering_angle, goal_motor_power):
        if self.current_image is not None:
            timestamp = rospy.Time.now().to_nsec()
            image_filename = os.path.join(self.images_dir, f'{timestamp}.jpg')
            transformed_image_filename = os.path.join(self.transformed_images_dir, f'{timestamp}.jpg')

            cv2.imwrite(image_filename, self.current_image)
            transformed_image = self.warp_transform(self.current_image)
            cv2.imwrite(transformed_image_filename, transformed_image)
            
            with open(self.csv_file1, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([image_filename, self.current_steering_angle, self.current_motor_power, goal_steering_angle, goal_motor_power])
            
            with open(self.csv_file2, mode='a') as file:
                writer = csv.writer(file)
                writer.writerow([image_filename, "", "", "", "", ""])

        # Sleep to maintain the rate
        self.rate.sleep()

if __name__ == '__main__':
    try:
        data_collector = DataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass