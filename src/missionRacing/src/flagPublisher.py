#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tensorflow as tf
from missionRacing.msg import DetectedObject  # 커스텀 메시지 가져오기

# YOLOv5 모델 로드 (TensorFlow로 변환된 모델 사용)
# model = tf.saved_model.load('path_to_your_tensorflow_yolov5_model')

bridge = CvBridge()
obstacle_pub = rospy.Publisher('/detection/obstacle', DetectedObject, queue_size=10)
crosswalk_pub = rospy.Publisher('/detection/crosswalk', DetectedObject, queue_size=10)
traffic_light_pub = rospy.Publisher('/detection/traffic_light', DetectedObject, queue_size=10)

def preprocess_image(image):
    image = tf.image.resize(image, (640, 640))  # YOLOv5 입력 크기
    image = image / 255.0
    image = tf.expand_dims(image, 0)
    return image

def image_callback(msg):
    try:
        # ROS 이미지를 OpenCV 이미지로 변환
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # YOLOv5 모델로 이미지 처리
    input_tensor = preprocess_image(cv_image)
    detections = model(input_tensor)

    # 결과 처리
    results = detections['output_0']  # 모델 출력의 키는 실제 모델에 따라 다를 수 있음
    results = np.array(results)

    # 결과를 저장할 딕셔너리 초기화
    detected_objects = {
        'obstacle': {'count': 0, 'boxes': []},
        'crosswalk': {'count': 0, 'boxes': []},
        'traffic light': {'count': 0, 'boxes': []}
    }

    for detection in results:
        class_id = int(detection[5])
        class_name = model.names[class_id]  # 클래스 이름 매핑
        x_min = int(detection[0])
        y_min = int(detection[1])
        x_max = int(detection[2])
        y_max = int(detection[3])

        if class_name == 'traffic light':
            detected_objects['traffic light']['count'] += 1
            detected_objects['traffic light']['boxes'].append((x_min, y_min, x_max, y_max))
        elif class_name == 'obstacle':
            detected_objects['obstacle']['count'] += 1
            detected_objects['obstacle']['boxes'].append((x_min, y_min, x_max, y_max))
        elif class_name == 'crosswalk':
            detected_objects['crosswalk']['count'] += 1
            detected_objects['crosswalk']['boxes'].append((x_min, y_min, x_max, y_max))

    # 각 토픽으로 메시지 퍼블리시
    for obj_type in detected_objects:
        msg = DetectedObject()
        msg.count = detected_objects[obj_type]['count']
        msg.x_min = [box[0] for box in detected_objects[obj_type]['boxes']]
        msg.y_min = [box[1] for box in detected_objects[obj_type]['boxes']]
        msg.x_max = [box[2] for box in detected_objects[obj_type]['boxes']]
        msg.y_max = [box[3] for box in detected_objects[obj_type]['boxes']]

        if obj_type == 'obstacle':
            obstacle_pub.publish(msg)
        elif obj_type == 'crosswalk':
            crosswalk_pub.publish(msg)
        elif obj_type == 'traffic light':
            traffic_light_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('yolo_v5_detector', anonymous=True)
    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
    rospy.spin()
