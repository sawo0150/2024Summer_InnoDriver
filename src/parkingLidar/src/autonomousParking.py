#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
import numpy as np
import cv2

class AutonomousParking:
    def __init__(self):
        rospy.init_node('autonomous_parking_node')
        self.resolution = 17.0 / 1024.0  # m/pixel
        self.image_size = 512  # Assuming a square image
        self.radius = self.image_size // 2
        self.center = (self.radius, self.radius)
        self.mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        self.car_width = 0.45
        self.car_length = 0.7
        self.goal_pub = rospy.Publisher("goal_state", Float64MultiArray, queue_size=2)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.current_steering_angle = 0.0
        self.goal_steering_angle = 0.0
        self.goal_motor_power = 0.0
        self.parking_space_found = False
        self.parking_complete = False
        self.driving_forward = True
        self.full_mask = self.create_full_mask()
        self.trajectory_masks = self.create_trajectory_masks()


        self.parkingDirection = 0       #1: 처음 기준 왼쪽, 2: 처음 기준 오른쪽에 존재

    def scan_callback(self, scan_data):
        self.mask.fill(0)
        angle_min = scan_data.angle_min
        angle_increment = scan_data.angle_increment
        for i, range in enumerate(scan_data.ranges):
            if range == float('inf') or range == float('-inf') or range == 0.0:
                continue
            angle = angle_min + i * angle_increment
            x = int(self.radius + (range / self.resolution) * np.cos(angle))
            y = int(self.radius + (range / self.resolution) * np.sin(angle))
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                self.mask = cv2.circle(self.mask, (x, y), 10, 255, -1)
        
        if self.driving_forward:
            self.drive_forward(scan_data)
        elif not self.parking_space_found:
            self.find_parking_space()
        else:
            self.park_car()
        
        cv2.imshow("Lidar Mask", self.mask)
        cv2.waitKey(1)
    
    def drive_forward(self, scan_data):
        ranges = np.array(scan_data.ranges)
        left_min = np.min(ranges[:len(ranges)//2])
        right_min = np.min(ranges[len(ranges)//2:])
        if left_min < 6 or right_min < 6:
            self.publishGoalState(0.0, 0.0)  # Stop
            rospy.sleep(0.5)
            if left_min < 6:
                self.parkingDirection = 1
                self.publishGoalState(-15.0, 0.2)  # Turn right
            else:
                self.parkingDirection = 2
                self.publishGoalState(15.0, 0.2)  # Turn left
            rospy.sleep(3)
            self.driving_forward = False
            self.publishGoalState(0.0, 0.0)  # Stop
            rospy.sleep(0.5)
        else:
            self.publishGoalState(0.0, 0.2)  # Move forward

    def create_full_mask(self):
        full_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        car1_center = (self.radius- int(0.95 / self.resolution), self.radius)
        car2_center = (self.radius + int(0.95 / self.resolution), self.radius)

        cv2.rectangle(full_mask, 
                      (car1_center[0] - int(self.car_width / (2 * self.resolution)), car1_center[1] - int(self.car_length / (2 * self.resolution))),
                      (car1_center[0] + int(self.car_width / (2 * self.resolution)), car1_center[1] + int(self.car_length / (2 * self.resolution))),
                      255, 3)

        cv2.rectangle(full_mask, 
                      (car2_center[0] - int(self.car_width / (2 * self.resolution)), car2_center[1] - int(self.car_length / (2 * self.resolution))),
                      (car2_center[0] + int(self.car_width / (2 * self.resolution)), car2_center[1] + int(self.car_length / (2 * self.resolution))),
                      255, 3)
        
        cv2.imshow("full Mask", full_mask)
        cv2.waitKey(1000)
        return full_mask
    
    def find_parking_space(self):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.mask, connectivity=8)
        if num_labels > 2:
            car_stats = []
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 500:  # Assuming this is the car's size
                    car_stats.append((stats[i], centroids[i]))

            if len(car_stats) >= 2:
                car1, car2 = car_stats[:2]
                self.estimate_parking_space(car1, car2)
                self.parking_space_found = True
    
    def estimate_parking_space(self, car1, car2):
        car1_center = (int(car1[1][0]), int(car1[1][1]))
        car2_center = (int(car2[1][0]), int(car2[1][1]))
        avg_center = ((car1_center[0] + car2_center[0]) // 2, (car1_center[1] + car2_center[1]) // 2)
        
        direction_vector = np.array([car2_center[0] - car1_center[0], car2_center[1] - car1_center[1]])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        self.target_mask = np.zeros_like(self.mask)
        start_point = (self.radius, self.radius)
        end_point = (self.radius + int(direction_vector[0] * self.image_size), self.radius + int(direction_vector[1] * self.image_size))
        
        cv2.line(self.target_mask, start_point, end_point, 255, int(self.car_width / self.resolution))
    
    def create_trajectory_masks(self):
        car_position = (self.radius, self.radius)
        trajectory_masks = [self.create_trajectory_mask(angle, car_position, self.car_width + 0.6, self.car_length, (self.image_size, self.image_size)) for angle in range(-20, 21)]
        return trajectory_masks
    
    def create_trajectory_mask(self, angle, car_position, car_width, car_height, image_size, decay_factor=1.2):
        larger_size = (image_size[0] * 2, image_size[1] * 2)
        mask = np.zeros(larger_size, dtype=np.float32)
        cx, cy = car_position
        offset_x, offset_y = image_size[1] // 2, image_size[0] // 2

        if angle == 0:
            for t in np.arange(0, 1.8 / self.resolution, 0.5):
                y = int(cy - t) + offset_y
                x1 = int(cx - car_width // 2) + offset_x
                x2 = int(cx + car_width // 2) + offset_x
                if y < 0 or x1 < 0 or x2 >= larger_size[1]:
                    break
                mask[y, x1:x2] = np.exp(-decay_factor * t * self.resolution)
        else:
            radius = np.abs(car_height / np.tan(np.radians(angle)))
            center_x = cx + (radius if angle > 0 else -radius) + offset_x

            for t in np.arange(0, 1.8 / self.resolution, 0.5):
                theta = t / radius
                y = int(cy - radius * np.sin(theta)) + offset_y
                x_center = int(cx - radius * (1 - np.cos(theta))) + offset_x

                if y < 0 or x_center < 0 or x_center >= larger_size[1]:
                    break

                if angle > 0:
                    left_radius = radius - car_width / 2
                    right_radius = radius + car_width / 2
                    x_left = int(center_x - left_radius * np.cos(theta))
                    x_right = int(center_x - right_radius * np.cos(theta))
                    y_left = int(cy - left_radius * np.sin(theta)) + offset_y
                    y_right = int(cy - right_radius * np.sin(theta)) + offset_y
                else:
                    left_radius = radius + car_width / 2
                    right_radius = radius - car_width / 2
                    x_left = int(center_x + left_radius * np.cos(theta))
                    x_right = int(center_x + right_radius * np.cos(theta))
                    y_left = int(cy - left_radius * np.sin(theta)) + offset_y
                    y_right = int(cy - right_radius * np.sin(theta)) + offset_y

                if x_left < 0 or x_right >= larger_size[1] or y_left < 0 or y_right >= larger_size[0]:
                    break

                cv2.line(mask, (x_left, y_left), (x_right, y_right), np.exp(-decay_factor * t * self.resolution), thickness=1)

        mask = mask[offset_y:image_size[0] + offset_y, offset_x:image_size[1] + offset_x]
        mask = (mask * 255).astype(np.uint8)
        return mask
    
    def calculate_optimal_steering(self):
        max_dot_product = -1
        optimal_angle = 0

        for angle, mask in zip(range(-20, 21), self.trajectory_masks):
            dot_product = np.sum(self.target_mask * mask)
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                optimal_angle = angle
        
        return optimal_angle

    def park_car(self):
        if not self.parking_complete:
            optimal_angle = self.calculate_optimal_steering()
            self.publishGoalState(optimal_angle, -1.0)  # Reverse with optimal steering angle
            rospy.sleep(1)
            self.parking_complete = True
            rospy.sleep(3)  # Wait for 3 seconds
            self.exit_parking_space()
    
    def exit_parking_space(self):
        self.publishGoalState(0.0, 1.0)  # Move forward for 3 seconds
        rospy.sleep(3)
        self.publishGoalState(-15.0 if self.current_steering_angle > 0 else 15.0, 1.0)  # Turn in opposite direction
        rospy.sleep(3)
        self.publishGoalState(0.0, 0.0)  # Stop
        rospy.sleep(3)
    
    def publishGoalState(self, goal_steering_angle, goal_motor_power):
        self.goal_steering_angle = goal_steering_angle
        self.goal_motor_power = goal_motor_power
        goal_pulse = (1 - np.abs(self.current_steering_angle - self.goal_steering_angle) / 2) * self.goal_motor_power
        msg = Float64MultiArray()
        msg.data = [self.goal_steering_angle, goal_pulse]
        self.goal_pub.publish(msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    autonomous_parking = AutonomousParking()
    autonomous_parking.run()
