#!/usr/bin/env python3

from model.DrivingNetwork import DrivingNetwork
import os
import cv2
import numpy as np
import tensorflow as tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import rospy
import pickle

print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Warp transform 관련 함수 정의
def load_warp_transform_matrix(directory, file_name='warp_matrix.pkl'):
    file_path = os.path.join(directory, file_name)
    try:
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

def warp_transform(cv_image, warp_matrix, width=256, height=256):
    top_view = cv2.warpPerspective(cv_image, warp_matrix, (width, height))
    return top_view

class AutonomousDriving:
    def __init__(self):
        rospy.init_node('autonomous_driving_node', anonymous=True)
        
        self.bridge = CvBridge()
        
        # Load the pre-trained model
        self.model = self.load_model()

        # Load the warp transform matrix
        self.warp_matrix = load_warp_transform_matrix('/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix')

        # Subscribers
        rospy.Subscriber('current_state', Float64MultiArray, self.current_state_callback)
        rospy.Subscriber('transformed_image', Image, self.image_callback)
        
        # Publisher
        self.pub = rospy.Publisher('goal_state', Float64MultiArray, queue_size=2)
        
        self.current_pulse = 0.0
        self.transformed_image = None

        self.rate = rospy.Rate(5)  # 10Hz

    def load_model(self):
        input_shape = (224, 224, 3)
        driving_network = DrivingNetwork(input_shape)
        model = driving_network.model
        checkpoint_path = '/home/innodriver/InnoDriver_ws/src/dlBasedDriving/src/model/driving_model_checkpoint.h5'
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            print("Loaded pre-trained model.")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        return model

    def current_state_callback(self, data):
        self.current_pulse = data.data[1]

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.transformed_image = warp_transform(cv_image, self.warp_matrix)
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)

    def run(self):
        while not rospy.is_shutdown():
            if self.transformed_image is not None:
                # Preprocess the image
                image = cv2.resize(self.transformed_image, (224, 224))
                image = np.expand_dims(image, axis=0)

                # Get the model output
                predicted_steering = self.model.predict(image)[0][0]
                pulse_range = 230 - 150
                pulse = 230 - (abs(predicted_steering) * pulse_range)
                # Create the goal_state message
                msg = Float64MultiArray()
                msg.data = [predicted_steering, pulse/255]

                # Publish the goal_state
                self.pub.publish(msg)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        autonomous_driving = AutonomousDriving()
        autonomous_driving.run()
    except rospy.ROSInterruptException:
        pass
