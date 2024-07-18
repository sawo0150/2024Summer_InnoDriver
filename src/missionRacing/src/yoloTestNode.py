import os
import tensorflow as tf
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# 경로 설정
tf_model_path = '/home/innodriver/InnoDriver_ws/yolo_train/best_saved_model'

# ROS 노드 클래스 정의
class YOLOv5Node:
    def __init__(self):
        # 모델 로드
        self.model = tf.saved_model.load(tf_model_path)
        self.infer = self.model.signatures['serving_default']
        
        # ROS 초기화
        rospy.init_node('yolov5_node', anonymous=True)
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # 이미지를 OpenCV 형식으로 변환
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        # 전처리
        input_tensor = self.preprocess(cv_image)

        # 추론
        detections = self.infer(input_tensor)

        # 후처리 및 결과 시각화
        output_image = self.postprocess(cv_image, detections)
        cv2.imshow('YOLOv5 Detection', output_image)
        cv2.waitKey(3)

    def preprocess(self, image):
        # 이미지 전처리
        input_tensor = cv2.resize(image, (640, 640))
        input_tensor = input_tensor / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        return tf.convert_to_tensor(input_tensor)

    def postprocess(self, image, detections):
        # 추론 결과 후처리
        detection_boxes = detections['output_0'].numpy()

        h, w, _ = image.shape
        for i in range(detection_boxes.shape[1]):
            box = detection_boxes[0, i, :4]
            score = detection_boxes[0, i, 4]
            class_id = int(detection_boxes[0, i, 5])
            if score < 0.5:
                continue
            y1, x1, y2, x2 = box
            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Class: {class_id}, Score: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image  # 박스 그린 이미지 반환

if __name__ == '__main__':
    # ROS 노드 실행
    try:
        yolo_node = YOLOv5Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
