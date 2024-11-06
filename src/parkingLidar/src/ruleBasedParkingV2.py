#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Int32
import cv2

class AutonomousParking:
    def __init__(self):
        rospy.init_node('autonomous_parking', anonymous=True)
        self.pub = rospy.Publisher('obstacle_detected', Int32, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.rate = rospy.Rate(10)  # 10Hz
        self.obstacle_detected = False

        self.width = 448
        self.height = 300

        self.resolution = 17/1024   # 17m/4096 픽셀
        self.carWidth = 0.42
        self.carBoxHeight = 0.54
        # 차량의 위치 및 크기 정의
        self.car_box_dist = 0.61
        self.car_center_x = int(self.width / 2)
        self.car_center_y = self.height - int(self.car_box_dist / self.resolution/2)
        self.car_TR_center_y = self.height + int(self.car_box_dist / self.resolution/2)
        self.car_width = int(self.carWidth /self.resolution)
        self.car_height = int(self.carBoxHeight /self.resolution)

        self.obstacle_radius = 0.5
        self.obstacle_detected_num = 0
    def scan_callback(self, scan_data):
        self.lidar_masks = []
        self.obstacle_distances = []
        output_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        angle_min = scan_data.angle_min
        angle_increment = scan_data.angle_increment
        
        front_lidar_points = []
        
        for i, range in enumerate(scan_data.ranges):
            if range == float('inf') or range == float('-inf') or range == 0.0:
                continue
            angle = np.pi +angle_min - i * angle_increment
            if -np.pi/2+4.5*np.pi/12 <= angle <= -np.pi/2+1*np.pi/2:  # -30 degrees to 30 degrees
                x = self.car_center_x+int(((self.obstacle_radius +range) / self.resolution) * np.cos(angle))
                y = self.car_TR_center_y+int(((self.obstacle_radius +range)/ self.resolution) * np.sin(angle))
                if 0 <= x < self.width and 0 <= y < self.height:
                    front_lidar_points.append((x, y, range, angle))
        
        if front_lidar_points:
            for x, y, distance, angle in front_lidar_points:
                obstacle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.circle(obstacle_mask, (x, y), 20, 255, -1)
                self.lidar_masks.append((obstacle_mask, distance))
                cv2.circle(output_mask, (x, y), 20, 255, -1)
        
        cv2.imshow("Lidar Mask", output_mask)
        cv2.waitKey(1)
        self.isObstacleDetected(front_lidar_points)

    def isObstacleDetected(self, points):
        for x, y, distance, angle in points:
            if distance < 2.0:  # If obstacle is within 2 meters
                self.obstacle_detected = True
                print('obstacle detected')
                self.obstacle_detected_num +=1
                return
        self.obstacle_detected = False
        self.publish_obstacle_status()

    def publish_obstacle_status(self):
        msg = Int32()
        msg.data = self.obstacle_detected_num
        self.pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    try:
        print("Starting autonomous parking node...")
        parking_controller = AutonomousParking()
        parking_controller.run()
    except rospy.ROSInterruptException:
        pass
