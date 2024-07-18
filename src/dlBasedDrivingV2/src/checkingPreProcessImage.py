#!/usr/bin/env python3

import os
import csv
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (320, 180))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    binary = binary / 255.0  # Normalize to [0, 1]
    binary = np.expand_dims(binary, axis=-1)  # Add channel dimension
    return binary

def display_preprocessed_images(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if dir == 'csv2':
                csv_dir = os.path.join(root, dir)
                for csv_file in os.listdir(csv_dir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(csv_dir, csv_file)
                        with open(csv_path, mode='r') as file:
                            reader = csv.DictReader(file)
                            for row in reader:
                                image_path = row["image"]
                                preprocessed_image = preprocess_image(image_path)
                                cv2.imshow('Preprocessed Image', preprocessed_image)
                                cv2.waitKey(100)  # Display each image for 100 ms
                break  # Stop further os.walk since we only want the csv2 directory of the current folder

if __name__ == "__main__":
    base_dir = '/home/innodriver/InnoDriver_ws/Data'
    display_preprocessed_images(base_dir)
    cv2.destroyAllWindows()
