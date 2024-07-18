#!/usr/bin/env python3

from model.DrivingNetworkV3 import drivingNetworkV3
import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def preprocess_image(image_path, target_size=(320, 180)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (target_size[1], target_size[0]))  # Resize to (180, 320) to match (width, height) order
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

def load_data(base_dir):
    images = []
    labels = []

    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if dir == 'csv1':
                csv_dir = os.path.join(root, dir)
                for csv_file in os.listdir(csv_dir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(csv_dir, csv_file)
                        print(csv_path)
                        with open(csv_path, mode='r') as file:
                            reader = csv.DictReader(file)
                            rows = list(reader)
                            for i in range(1, len(rows) - 1):
                                image_path = rows[i]["image"]
                                goal_steering_angle = float(rows[i]["goal_steering_angle"])
                                goal_steering_angle_prev = float(rows[i - 1]["goal_steering_angle"])
                                goal_steering_angle_next = float(rows[i + 1]["goal_steering_angle"])

                                label = (goal_steering_angle_prev + 2 * goal_steering_angle + goal_steering_angle_next) / 4
                                
                                image = preprocess_image(image_path)
                                images.append(image)
                                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


if __name__ == "__main__":
    input_shape = (320, 180, 3)
    base_dir = '/home/innodriver/InnoDriver_ws/Data'
    images, labels = load_data(base_dir)

    model = drivingNetworkV3(input_shape).model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    checkpoint_dir = '/home/innodriver/InnoDriver_ws/DNv3checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_val_path = os.path.join(checkpoint_dir, 'best_model_val_loss.h5')
    checkpoint_train_path = os.path.join(checkpoint_dir, 'best_model_train_loss.h5')

    # Checkpoint settings
    checkpoint_val = ModelCheckpoint(checkpoint_val_path, monitor='val_loss', save_best_only=True, mode='min')
    checkpoint_train = ModelCheckpoint(checkpoint_train_path, monitor='loss', save_best_only=True, mode='min')

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # history = model.fit(images, labels, epochs=50, batch_size=32, validation_split=0.2, 
    #                     callbacks=[checkpoint_val, checkpoint_train, early_stopping])

    history = model.fit(images, labels, epochs=50, batch_size=32, validation_split=0.2, 
                        callbacks=[checkpoint_val, checkpoint_train])
    
    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.h5')
    model.save(final_model_path)