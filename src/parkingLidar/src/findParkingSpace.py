#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
import cv2

class LidarMask:
    def __init__(self):
        rospy.init_node('lidar_mask_node')
        self.resolution = 17.0 / 1024.0  # m/pixel
        self.image_size = 512  # Assuming a square image
        self.radius = self.image_size // 2
        self.center = (self.radius, self.radius)
        self.mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
    
    def scan_callback(self, scan_data):
        self.mask.fill(0)  # Clear the mask
        angle_min = scan_data.angle_min
        angle_increment = scan_data.angle_increment
        print(angle_min, angle_increment)
        
        for i, range in enumerate(scan_data.ranges):
            if range == float('inf') or range == float('-inf') or range == 0.0:
                continue
            angle = np.pi +angle_min - i * angle_increment
            x = int(self.radius + (range / self.resolution) * np.cos(angle))
            y = int(self.radius + (range / self.resolution) * np.sin(angle))
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                self.mask = cv2.circle(self.mask, (x, y), 10, 255, -1)
        
        cv2.imshow("Lidar Mask", self.mask)
        cv2.waitKey(1)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    lidar_mask = LidarMask()
    lidar_mask.run()
