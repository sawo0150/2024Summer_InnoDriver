#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pickle
import os
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
import tf
import time
from scipy.spatial.distance import cosine

class AutonomousCarLocalizer:
    def __init__(self):
        # Node 초기화
        rospy.init_node('autonomous_car_localization')
        self.bridge = CvBridge()
        self.width = 224
        self.height = 224

        # 전역 변수로 행렬 불러오기
        self.directory = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
        self.warp_matrix = self.load_warp_transform_matrix(self.directory)
        self.map_image = self.load_map_image(os.path.join(self.directory, "map_image.png"))
        self.map_imageBinary = self.changeImageForSimilarity(self.map_image)

        # 맵의 크기 설정
        self.map_width = 612  # 맵의 가로 크기
        self.map_height = 455  # 맵의 세로 크기
        
        self.top_view = None
        # 파라미터 설정
        self.num_particles = 500
        self.particles = np.zeros((self.num_particles, 3))  # x, y, theta
        self.weights = np.ones(self.num_particles) / self.num_particles
        # 맵 전체에 파티클들을 균일하게 분포
        self.uniform_initialize_particles()

        self.updateFrame = 1

        self.box_body_dist = 0.61
        # meter/pixel 비율
        self.resolution = 17/612  # 17m/612 픽셀
        self.wheelbase = 0.54
        self.velocity = 0
        self.steering_angle = 0
        self.maxSteeringAngle = 14.0
        self.lastTime = time.time()
        self.delX = 0
        self.delY = 0
        self.delTheta = 0

        # 이미지, 드라이브, 포즈 메시지 구독
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.steering_motor_sub = rospy.Subscriber("current_state", Float64MultiArray, self.motor_callback)
        self.pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.pose_callback)
        # 위치 및 방향 발행자
        self.pose_pub = rospy.Publisher("/car/pose", PoseWithCovarianceStamped, queue_size=1)
    
    def uniform_initialize_particles(self):
        self.particles[:, 0] = np.random.uniform(0, self.map_width, self.num_particles)  # x 좌표
        self.particles[:, 1] = np.random.uniform(0, self.map_height, self.num_particles)  # y 좌표
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)  # theta (방향)
        print("Particles initialized uniformly across the map.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.top_view = self.warp_transform(cv_image)
            # cv2.imshow("Transformed Image", self.top_view)
            # cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)
    def pwm2velosity(self, pwm):
        vel = 0.77 *(0.7+0.3*pwm/255)
        return vel

    def motor_callback(self, msg):
        self.steering_angle = self.maxSteeringAngle *msg.data[0]
        self.velocity = self.pwm2velosity(255*msg.data[1])
        # delT = time.time() - self.lastTime
        # self.delX += self.velocity * np.cos(theta) * delT
        # --> theta가 뭔지를 알아야 계속 잘 update를 할 수 있는데 particle마다 theta가 다르니까 움직일 수 없음

    def load_map_image(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No map image found at {file_path}.")
        map_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if map_image is None:
            raise IOError("Failed to load map image.")
        print("Map image loaded successfully.")
        return map_image

    def load_warp_transform_matrix(self, directory, file_name='warp_matrix.pkl'):
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'rb') as f:
                matrix = pickle.load(f)
            print("Loaded transform matrix from", file_path)
            return matrix
        except FileNotFoundError:
            raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

    def warp_transform(self, cv_image):
        # tempImage = self.changeImageForSimilarity(cv_image)
        top_view = cv2.warpPerspective(cv_image, self.warp_matrix, (self.width, self.height))
        return top_view

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        theta = self.euler_from_quaternion(quaternion)[2]  # yaw 각도 추출

        # 파티클 초기화
        self.initialize_particles(x, self.map_height-y, theta)

    def euler_from_quaternion(self, quat):
        import tf.transformations as transformations
        return transformations.euler_from_quaternion(quat)

    def initialize_particles(self, x, y, theta, std_dev=50, angle_std_dev=np.pi / 6):
        self.particles[:, 0] = x + np.random.normal(0, std_dev, self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, std_dev, self.num_particles)
        self.particles[:, 2] = theta + np.random.normal(0, angle_std_dev, self.num_particles)
    
    def motion_update(self, dt):
        for i in range(self.num_particles):
            theta = self.particles[i][2]
            self.particles[i][0] += self.velocity * np.cos(theta) * dt / self.resolution
            self.particles[i][1] += self.velocity * np.sin(theta) * dt / self.resolution
            self.particles[i][2] += self.velocity * np.tan(self.steering_angle) / self.wheelbase * dt 

    def measurement_update(self, map_image, top_view_image):
        particles = self.particles
        for i in range(self.num_particles):
            x, y, theta = particles[i]
            transformed_image, areaList = self.transform_image(top_view_image, x, y, theta, map_image.shape)
            self.weights[i] = self.calculate_similarity(transformed_image, areaList)
        

    def transform_image(self, image, x, y, theta, output_shape):    # 출력 이미지 초기화
        # transformed_image = np.zeros(output_shape, dtype=np.uint8)

        # 원본 이미지 크기와 회전 후의 이미지 크기를 기반으로 한 대각 길이 계산
        diagonal = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
        output_size = (diagonal, diagonal)

        # 회전을 위한 변환 행렬 계산 (회전 중심은 원본 이미지의 중심)
        center_x, center_y = diagonal / 2, diagonal / 2
        M_rotate = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(theta)-90, 1)

        # 평행 이동 변환 행렬 생성 (새 이미지의 중심으로 이동)
        center_offset_x = (output_size[1] - image.shape[1]) / 2
        center_offset_y = (output_size[0] - image.shape[0]) / 2
        # M_translate = np.array([[1, 0, center_offset_x],
        #                         [0, 1, center_offset_y]])
        # print(center_offset_x, center_offset_y)
        M_translate = np.array([[1, 0, center_offset_x],
                        [0, 1, center_offset_y],
                        [0, 0, 1]])

        # 변환 행렬 결합
        M_combined = np.dot(M_rotate, M_translate)

        # 변환 적용
        rotated_image = cv2.warpAffine(image, M_combined[:2], output_size)

        # cv2.imshow("Transformed Image", rotated_image)
        # cv2.waitKey(10)
        # 이미지를 새 위치에 복사
        start_x = int(x + (self.box_body_dist/self.resolution+image.shape[0]/2) * np.cos(theta) - diagonal/2)
        start_y = int(y - (self.box_body_dist/self.resolution+image.shape[0]/2) * np.sin(theta) - diagonal/2)
        end_x = start_x + rotated_image.shape[1]
        end_y = start_y + rotated_image.shape[0]

        # 범위 확인 및 유효한 영역 계산
        valid_start_x = max(start_x, 0)
        valid_start_y = max(start_y, 0)
        valid_end_x = min(end_x, output_shape[1])
        valid_end_y = min(end_y, output_shape[0])

        # # 유효한 영역의 이미지만 복사
        if valid_start_x < valid_end_x and valid_start_y < valid_end_y:
            transformed_image = rotated_image[
                (valid_start_y - start_y):(valid_start_y - start_y + valid_end_y - valid_start_y),
                (valid_start_x - start_x):(valid_start_x - start_x + valid_end_x - valid_start_x)
            ]
        else:
            transformed_image = rotated_image
        

        
        # cv2.imshow("Transformed Image", transformed_image)
        # cv2.waitKey(10)
        # print(start_x, start_y, end_x, end_y)
        # cv2.circle(transformed_image, (int(start_x), int(start_y)), 10, (0,0,255), 3)
        # cv2.circle(transformed_image, (int(end_x), int(end_y)), 10, (255,0,0), 3)
 
        return transformed_image, [valid_start_x, valid_start_y, valid_end_x, valid_end_y]

    def changeImageForSimilarity(self, image):
        # 이미지를 그레이스케일로 변환
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # # Canny edge 검출
        # edges1 = cv2.Canny(gray1, 150, 150)

        # # Morphology 연산으로 에지 굵기 증가
        # kernel = np.ones((2, 2), np.uint8)  # 에지를 두껍게 하기 위한 커널
        # edges1 = cv2.dilate(edges1, kernel, iterations=1)

        # # Otsu's thresholding으로 이진화
        # _, binary1 = cv2.threshold(edges1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        
        # Hue와 Value 채널 추출
        hue1, _, value1 = cv2.split(hsvImage)

        # adaptive_thresh = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
        # inverted_image = cv2.bitwise_not(adaptive_thresh)
        return hue1, value1
    def crop_to_match(image1, image2):
        # 두 이미지의 크기를 가져옵니다.
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # 이미지를 크롭할 시작점을 계산합니다.
        if h1 > h2 or w1 > w2:
            start_y = (h1 - h2) // 2
            start_x = (w1 - w2) // 2
            cropped_image1 = image1[start_y:start_y + h2, start_x:start_x + w2]
            return cropped_image1, image2
        elif h2 > h1 or w2 > w1:
            start_y = (h2 - h1) // 2
            start_x = (w2 - w1) // 2
            cropped_image2 = image2[start_y:start_y + h1, start_x:start_x + w1]
            return image1, cropped_image2
        else:
            # 이미지 크기가 같으면 그대로 반환
            return image1, image2
    def calculate_similarity(self, image, areaList):
        valid_start_x, valid_start_y, valid_end_x, valid_end_y = areaList
        mapImageHue = self.map_imageBinary[0].copy()
        mapImageValue = self.map_imageBinary[1].copy()
        # 유효한 영역의 이미지만 복사
        if valid_start_x < valid_end_x and valid_start_y < valid_end_y:
            mapImageHue = mapImageHue[valid_start_y:valid_end_y, valid_start_x:valid_end_x] 
            mapImageValue = mapImageValue[valid_start_y:valid_end_y, valid_start_x:valid_end_x] 
        # mapImage, image = self.crop_to_match(image, mapImage)


        
        hueImage, valueImage = self.changeImageForSimilarity(image)
        # binary1 = image

        # 두 이진 이미지의 교집합 계산
        # intersection = cv2.bitwise_and(binary1, mapImage)

        # tot = np.sum(binary1==255)
        # if tot != 0:
        #     # 교집합에서 1인 픽셀의 개수 계산
        #     similarity = np.sum(intersection == 255) / tot
        # else:
        # similarity = np.sum(intersection == 255)
        
        # print(similarity)
        # cv2.imshow("Transformed Image", binary1)
        # cv2.waitKey(10)

        # Hue 유사도 계산
        hue_difference = np.abs(mapImageHue.astype(int) - hueImage.astype(int))
        hue_difference = np.minimum(hue_difference, 360 - hue_difference)  # 원형 거리 고려
        hue_similarity = 1 - (hue_difference / 180.0)  # 0-180 범위를 0-1로 변환
        
        # Value 유사도 계산 (코사인 유사도)
        value_similarity = 1 - cosine(mapImageValue.flatten(), valueImage.flatten())
        
        # 최종 유사도는 Hue와 Value 유사도의 평균
        final_similarity = np.mean(hue_similarity)*0.8 + value_similarity*0.2
        return final_similarity
    def resample(self):
        # 가중치를 정규화합니다.
        sum_weights = np.sum(self.weights)
        if sum_weights == 0:
            rospy.logwarn("Sum of particle weights is zero, reinitializing weights.")
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= sum_weights

        try:
            indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
            self.particles[:] = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)
        except ValueError as e:
            rospy.logerr("Error in resampling: {0}".format(e))

    def publish_pose(self):
        if np.sum(self.weights) == 0:
            rospy.logwarn("All particle weights are zero.")
            return

        # 가중치에 따른 평균 위치 계산
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        mean_theta = np.average(self.particles[:, 2], weights=self.weights)

        # 메시지 생성
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = mean_x
        pose_msg.pose.pose.position.y = self.map_height - mean_y
        pose_msg.pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, mean_theta))

        # 가중치에 따른 방향(orientation) 계산 및 설정
        covariance = np.zeros((6, 6))
        variance = 1.0 / np.sum(self.weights)  # Simple approximation
        covariance[0, 0] = variance  # x variance
        covariance[1, 1] = variance  # y variance
        covariance[5, 5] = variance  # theta variance

        pose_msg.pose.covariance = tuple(covariance.flatten())

        transformed_image = self.transform_image(self.top_view , mean_x, mean_y, mean_theta, self.map_image.shape)
        output_image = self.map_imageBinary[0].copy()
        cv2.circle(output_image, (int(mean_x), int(mean_y)), 10, 255, 3)
        transformed_image = self.changeImageForSimilarity(transformed_image[0])[0]
        cv2.imshow("Transformed Image", transformed_image)
        cv2.waitKey(10)
        # 위치 정보 발행
        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Published pose: (%f, %f, %f)", mean_x, mean_y, mean_theta)

    def run(self):
        self.updateFrame = 1
        rate = rospy.Rate(self.updateFrame)  # 10Hz
        while not rospy.is_shutdown():
            if self.top_view is not None:
                print("mapping 실행중")
                initTime = time.time()
                self.motion_update(1/self.updateFrame)  # 예시 업데이트, 실제 사용시 입력 받아야 함
                self.measurement_update(self.map_image, self.top_view)  # 맵 이미지와 최신 top view 이미지 필요
                self.resample()
                self.publish_pose()
                rate.sleep()
                print(time.time()-initTime)


if __name__ == "__main__":
    localizer = AutonomousCarLocalizer()
    localizer.run()
