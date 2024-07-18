#!/usr/bin/env python3
import sys
print(sys.executable)
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img

class OffLearningEnv:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.index = 0

    def reset(self):
        self.index = 0
        return self._get_state(self.index)

    def step(self):
        self.index += 1
        if self.index >= len(self.data):
            done = True
            self.index = 0
        else:
            done = False
        
        state = self._get_state(self.index)
        reward = self._calculate_reward(self.index)
        
        return state, reward, done

    def _get_state(self, index):
        row = self.data.iloc[index]
        image_path = row['image']
        steering_angle = row['steering_angle']
        motor_power = row['motor_power']

        image = img_to_array(load_img(image_path, target_size=(224, 224))) / 255.0
        
        return image, steering_angle, motor_power

    def _calculate_reward(self, index):
        row = self.data.iloc[index]
        goal_steering_angle = row['goal_steering_angle']
        goal_motor_power = row['goal_motor_power']
        
        # 예시로 간단한 보상 함수
        reward = 1.0 - abs(row['steering_angle'] - goal_steering_angle) / 90.0
        reward += 1.0 - abs(row['motor_power'] - goal_motor_power) / 0.5
        
        return reward
