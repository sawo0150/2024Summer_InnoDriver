#!/usr/bin/env python3
import os
import tensorflow as tf
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pickle
import numpy as np
from missionRacing.msg import CrosswalkInfo  # 사용할 새로운 메시지 형식

# YOLO 모델 경로 설정
tf_crosswalk_model_path = '/home/innodriver/InnoDriver_ws/yolo_train/crosswalk_best_saved_model'
tf_traffic_light_model_path = '/home/innodriver/InnoDriver_ws/yolo_train/trafficLight_best_saved_model'

class CrossWalkAnalizer:
    def __init__(self):
        # 모델 로드
        self.crosswalk_model = tf.saved_model.load(tf_crosswalk_model_path)
        self.crosswalk_infer = self.crosswalk_model.signatures['serving_default']

        self.traffic_light_model = tf.saved_model.load(tf_traffic_light_model_path)
        self.traffic_light_infer = self.traffic_light_model.signatures['serving_default']
        
        # ROS 초기화
        rospy.init_node('crossWalkAnalizer', anonymous=True)
        # Subscribers
        self.image_sub_raw1 = rospy.Subscriber("/camera1/usb_cam1/image_raw", Image, self.callback_raw1)
        self.image_sub_raw2 = rospy.Subscriber("/camera2/usb_cam2/image_raw", Image, self.callback_raw2)
        self.info_pub = rospy.Publisher('/crosswalk_info', CrosswalkInfo, queue_size=10)
        self.bridge = CvBridge()
        
        # Directory and file settings
        self.directory = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
        self.warp_matrix = self.load_warp_transform_matrix(self.directory)
        
        self.width = 448
        self.height = 300
        self.resolution = 17/1024   # 17m/4096 픽셀
        self.laneWidth = 0.90       #원래는 0.85m지만, 조금 조정?

        self.carWidth = 0.42
        self.carBoxHeight = 0.54
        # 차량의 위치 및 크기 정의
        self.car_box_dist = 0.61
        self.car_center_x = int(self.width / 2)
        self.car_center_y = self.height - int(self.car_box_dist / self.resolution/2)
        self.car_TR_center_y = self.height + int(self.car_box_dist / self.resolution/2)
        self.car_width = int(self.carWidth /self.resolution)
        self.car_height = int(self.carBoxHeight /self.resolution)
        self.last_1detection_time = rospy.Time.now()
        self.last_2detection_time = rospy.Time.now()

        self.crosswalk_detected = False
        self.crosswalk_distance = -1
        self.traffic_light_detected = False
        self.light_type = 0

    # 기존 warp matrix를 로드하는 함수
    def load_warp_transform_matrix(self, directory, file_name='warp_matrix.pkl'):
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'rb') as f:
                matrix = pickle.load(f)
            rospy.loginfo("Loaded transform matrix from %s", file_path)
            return matrix
        except FileNotFoundError:
            raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

    def warp_transform(self, cv_image, width, height):
        top_view = cv2.warpPerspective(cv_image, self.warp_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return top_view

    @tf.function
    def yolo_predict(self, model, input_tensor):
        return model(input_tensor)

    def preprocess(self, image):
        input_tensor = cv2.resize(image, (640, 640))
        input_tensor = input_tensor / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        return tf.convert_to_tensor(input_tensor)

    def postprocess_traffic_light(self, image, detections):
        detection_boxes = detections['output_0'].numpy()
        h, w, _ = image.shape

        for i in range(detection_boxes.shape[1]):
            box = detection_boxes[0, i, :4]
            score = detection_boxes[0, i, 4]
            class_id = int(detection_boxes[0, i, 5])
            if score < 0.5:
                continue

            y1, x1, y2, x2 = box
            print(box)
            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)

            if class_id == 0:  # Green light
                self.light_type = 2
                self.traffic_light_detected = True
            elif class_id == 3:  # Red light
                self.light_type = 1
                self.traffic_light_detected = True

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Class: {class_id}, Score: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def callback_raw1(self, data):
        if (rospy.Time.now() - self.last_1detection_time).to_sec() < 1.0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        input_tensor = self.preprocess(cv_image)
        detections = self.yolo_predict(self.crosswalk_infer, input_tensor)
        output_image = self.postprocess_crosswalk(cv_image, detections)

        cv2.imshow('Crosswalk Detection', output_image)
        cv2.waitKey(3)

        self.last_1detection_time = rospy.Time.now()

        # 메시지 발행
        self.publish_info()

    def callback_raw2(self, data):
        if (rospy.Time.now() - self.last_2detection_time).to_sec() < 1.0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        input_tensor = self.preprocess(cv_image)
        detections = self.yolo_predict(self.traffic_light_infer, input_tensor)
        output_image = self.postprocess_traffic_light(cv_image, detections)

        cv2.imshow('Traffic Light Detection', output_image)
        cv2.waitKey(1)

        self.last_2detection_time = rospy.Time.now()

        # 메시지 발행
        self.publish_info()

    def publish_info(self):
        info_msg = CrosswalkInfo()
        info_msg.crosswalk_detected = self.crosswalk_detected
        info_msg.crosswalk_distance = self.crosswalk_distance
        info_msg.traffic_light_detected = self.traffic_light_detected
        info_msg.light_type = self.light_type
        self.info_pub.publish(info_msg)

        # Reset the flags
        self.crosswalk_detected = False
        self.crosswalk_distance = -1
        self.traffic_light_detected = False
        self.light_type = 0

if __name__ == '__main__':
    try:
        yolo_node = CrossWalkAnalizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
