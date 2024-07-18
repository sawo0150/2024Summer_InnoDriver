#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import glob

class DataPublisher:
    def __init__(self):
        self.bridge = CvBridge()

        rospy.init_node('data_publisher', anonymous=True)
        
        # Publisher
        self.image_pub = rospy.Publisher("/usb_cam/image_raw", Image, queue_size=10)

        self.rate = rospy.Rate(10)  # 10hz

        # Image directory
        self.images_dir = '/home/innodriver/InnoDriver_ws/src/dQN_modelTraining/offPolicyLearningData/images'

        # Get list of image files
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))
        self.current_index = 0

    def publish_images(self):
        while not rospy.is_shutdown() and self.current_index < len(self.image_files):
            image_file = self.image_files[self.current_index]
            try:
                cv_image = cv2.imread(image_file)
                if cv_image is not None:
                    image_message = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                    self.image_pub.publish(image_message)
                    # rospy.loginfo(f'Published {image_file}')
                else:
                    rospy.logwarn(f'Failed to read {image_file}')
            except CvBridgeError as e:
                rospy.logerr(f'CvBridgeError: {e}')
            
            self.current_index += 1
            self.rate.sleep()

if __name__ == '__main__':
    data_publisher = DataPublisher()
    data_publisher.publish_images()
