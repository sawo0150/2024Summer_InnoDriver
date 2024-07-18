#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Bool
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import numpy as np
import cv2
import os

class AutonomousDrivingNode:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('autonomous_driving_node', anonymous=True)
        
        # Load trained model
        model_path = '/home/innodriver/InnoDriver_ws/DNv2checkpoints/best_model_val_loss.h5'  # Replace XXX with the epoch number
        self.model = tf.keras.models.load_model(model_path)
        
        # Subscribers
        self.image_sub_raw = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_raw)
        self.control_sub = rospy.Subscriber("control_signal", Bool, self.control_callback)
        
        # Publisher
        self.pub = rospy.Publisher('goal_state', Float64MultiArray, queue_size=2)

        # State variable
        self.running = False
        self.rate = rospy.Rate(5)  # 10 Hz


    def preprocess_image(self, cv_image):
        image = cv2.resize(cv_image, (320, 180))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        binary = binary / 255.0  # Normalize to [0, 1]
        
        cv2.imshow('binary Mask', binary)
        cv2.waitKey(10)
        binary = np.expand_dims(binary, axis=-1)  # Add channel dimension
        return np.expand_dims(binary, axis=0)  # Add batch dimension
    # def preprocess_image(image_path):
        # image = image / 255.0  # Normalize to [0, 1]
        # image = np.expand_dims(image, axis=0)  # Add batch dimension
        # return image
    
    def control_callback(self, msg):
        self.running = msg.data
        rospy.loginfo(f"Received control signal: {'Running' if self.running else 'Stopped'}")

    def callback_raw(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            preprocessed_image = self.preprocess_image(cv_image)
            self.predict_and_publish(preprocessed_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        self.rate.sleep()

    def predict_and_publish(self, preprocessed_image):
        predictions = self.model.predict(preprocessed_image)[0]
        steering_values = np.array([-2, -1, 0, 1, 2])
        predicted_steering = 0-np.sum(predictions * steering_values) / 2
        
        pulse_range = 230 - 150
        pulse = 230 - (abs(predicted_steering) * pulse_range)
        
        # If not running, set pulse to 0
        if not self.running:
            pulse = 0

        # Create the goal_state message
        msg = Float64MultiArray()
        msg.data = [predicted_steering, pulse / 255]
        # msg.data = [predicted_steering, 0]
        self.pub.publish(msg)

if __name__ == '__main__':
    try:
        autonomous_driving_node = AutonomousDrivingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
