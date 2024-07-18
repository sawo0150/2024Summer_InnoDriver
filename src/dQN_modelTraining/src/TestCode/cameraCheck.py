#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageViewer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub_raw = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_raw)
        self.image_sub_compressed = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback_compressed)

    def callback_raw(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        self.process_and_show_image(cv_image)

    def callback_compressed(self, data):
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        self.process_and_show_image(cv_image)

    def process_and_show_image(self, image):
        # 이미지 크기 정보 출력
        rospy.loginfo(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # 이미지에 텍스트 추가
        cv2.putText(image, 'innodriver', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 이미지 표시
        cv2.imshow('Image', image)
        cv2.waitKey(1)  # 1ms 동안 대기하여 창을 업데이트합니다.

if __name__ == '__main__':
    rospy.init_node('image_viewer', anonymous=True)
    ImageViewer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()
