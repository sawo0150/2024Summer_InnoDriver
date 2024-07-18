#!/usr/bin/env python3

from model.DrivingNetwork import DrivingNetwork
import os
import cv2
import numpy as np
import tensorflow as tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float64MultiArray
import rospy
import pickle

class DrivingNode:
    def __init__(self):
        rospy.init_node('driving_node', anonymous=True)
        
        self.bridge = CvBridge()
        
        # Load the pre-trained model
        self.model = self.load_model()

        # Load the warp transform matrix
        self.warp_matrix = self.load_warp_transform_matrix('/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix')

        # Subscribers
        rospy.Subscriber('current_state', Float64MultiArray, self.current_state_callback)
        rospy.Subscriber('transformed_image', Image, self.image_callback)
        rospy.Subscriber('/decision_flag', Float32, self.decision_flag_callback)
        
        # Publisher
        self.pub = rospy.Publisher('goal_state', Float64MultiArray, queue_size=2)
        
        self.current_pulse = 0.0
        self.transformed_image = None
        self.decision_flag = 0.0

        self.rate = rospy.Rate(5)  # 10Hz

    def load_model(self):
        input_shape = (256, 256, 3)
        driving_network = DrivingNetwork(input_shape)
        model = driving_network.model
        checkpoint_path = '/home/innodriver/InnoDriver_ws/src/dlBasedDriving/src/model/driving_model_checkpoint.h5'
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            print("Loaded pre-trained model.")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        return model

    def load_warp_transform_matrix(self, directory, file_name='warp_matrix.pkl'):
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'rb') as f:
                matrix = pickle.load(f)
            return matrix
        except FileNotFoundError:
            raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

    def warp_transform(self, cv_image, warp_matrix, width=256, height=256):
        top_view = cv2.warpPerspective(cv_image, warp_matrix, (width, height))
        return top_view

    def current_state_callback(self, data):
        self.current_pulse = data.data[1]

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.transformed_image = self.warp_transform(cv_image, self.warp_matrix)
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)

    def decision_flag_callback(self, msg):
        self.decision_flag = msg.data

    def run(self):
        while not rospy.is_shutdown():
            if self.transformed_image is not None:
                if self.decision_flag == 0.0:
                    # Free driving mode
                    image = cv2.resize(self.transformed_image, (256, 256))
                    image = np.expand_dims(image, axis=0)

                    # Get the model output
                    predicted_steering = self.model.predict(image)[0][0]
                    pulse_range = 230 - 150
                    pulse = 230 - (abs(predicted_steering) * pulse_range)
                elif self.decision_flag == 2.0:
                    # Obstacle avoidance mode
                    predicted_steering = 0.0  # 직진
                    pulse = 150  # 느린 속도로 주행
                elif self.decision_flag == 2.1:
                    # Lane change to 1st lane
                    predicted_steering = -0.5  # 왼쪽으로 조향
                    pulse = 200  # 적당한 속도로 주행
                elif self.decision_flag == 2.2:
                    # Lane change to 2nd lane
                    predicted_steering = 0.5  # 오른쪽으로 조향
                    pulse = 200  # 적당한 속도로 주행
                elif self.decision_flag == 3.0:
                    # Crosswalk detection mode
                    predicted_steering = 0.0  # 직진
                    pulse = 150  # 느린 속도로 주행
                elif self.decision_flag == 3.1:
                    # Red light detected
                    predicted_steering = 0.0  # 멈춤
                    pulse = 0  # 정지
                elif self.decision_flag == 3.2:
                    # Green light detected
                    predicted_steering = 0.0  # 직진
                    pulse = 200  # 출발
                
                # Create the goal_state message
                msg = Float64MultiArray()
                msg.data = [predicted_steering, pulse / 255.0]

                # Publish the goal_state
                self.pub.publish(msg)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        driving_node = DrivingNode()
        driving_node.run()
    except rospy.ROSInterruptException:
        pass
