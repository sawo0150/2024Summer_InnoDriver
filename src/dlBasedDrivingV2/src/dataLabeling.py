#!/usr/bin/env python3

import os
import csv
import cv2
import numpy as np

resolution = 17.0/1024  #m/pixel
carWidth  = 0.425
carHeight = 0.54

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (320, 180))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    binary = binary / 255.0  # Normalize to [0, 1]
    binary = np.expand_dims(binary, axis=-1)  # Add channel dimension
    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (320, 180))
    # image = image / 255.0  # Normalize to [0, 1]
    # return image
    return binary

def add_car_silhouette(transformed_image, carWidth, carHeight, resolution):
    height, width, _ = transformed_image.shape
    new_height = height + int(carHeight / resolution)
    
    # Create a new image with the additional space for the car silhouette
    new_image = np.zeros((new_height, width, 3), dtype=np.uint8)
    new_image[:height, :] = transformed_image
    
    # Define the car silhouette rectangle
    car_width_px = int(carWidth / resolution)
    car_height_px = int(carHeight / resolution)
    center_x = width // 2
    # center_y = height + car_height_px // 2
    
    # Draw the car silhouette
    top_left = (center_x - car_width_px // 2, height)
    bottom_right = (center_x + car_width_px // 2, height + car_height_px)
    cv2.rectangle(new_image, top_left, bottom_right, (255, 255, 255), -1)  # White rectangle
    
    return new_image

def display_images_and_label(csv_path, image_dir, transformed_image_dir):
    rows = []
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    
    start_index = 0
    for i, row in enumerate(rows):
        if row['-2'] == '1' or row['-1'] == '1' or row['0'] == '1' or row['1'] == '1' or row['2'] == '1':
            start_index = i + 1
    
    for i, row in enumerate(rows[start_index:], start=start_index):
        image_path = os.path.join(image_dir, os.path.basename(row["image"]))
        transformed_image_path = os.path.join(transformed_image_dir, os.path.basename(image_path))
        
        original_image = cv2.imread(image_path)
        preprocessed_image = preprocess_image(image_path)
        transformed_image = cv2.imread(transformed_image_path)
        
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Preprocessed Image', preprocessed_image)
        cv2.imshow('Transformed Image', add_car_silhouette(transformed_image, carWidth, carHeight, resolution))
        
        cv2.waitKey(60)
        # if key == ord('1'):
        #     rows[i]['-2'] = '1'
        # elif key == ord('2'):
        #     rows[i]['-1'] = '1'
        # elif key == ord('3'):
        #     rows[i]['0'] = '1'
        # elif key == ord('4'):
        #     rows[i]['1'] = '1'
        # elif key == ord('5'):
        #     rows[i]['2'] = '1'
        # else:
        #     print("Invalid key pressed. Please press 1, 2, 3, 4, or 5.")
        #     continue
        
        valid_input = False
        while not valid_input:
            user_input = input("Enter label (-2: 1, -1: 2, 0: 3, 1: 4, 2: 5): ")
            if user_input in ['1', '2', '3', '4', '5']:
                valid_input = True
                if user_input == '1':
                    rows[i]['-2'] = '1'
                elif user_input == '2':
                    rows[i]['-1'] = '1'
                elif user_input == '3':
                    rows[i]['0'] = '1'
                elif user_input == '4':
                    rows[i]['1'] = '1'
                elif user_input == '5':
                    rows[i]['2'] = '1'
            else:
                print("Invalid input. Please enter 1, 2, 3, 4, or 5.")
        

        with open(csv_path, mode='w', newline='') as file:
            fieldnames = ['image', '-2', '-1', '0', '1', '2']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    cv2.destroyAllWindows()

def main():
    base_dir = '/home/innodriver/InnoDriver_ws/Data'
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    print("Available folders:")
    for i, folder in enumerate(folders):
        print(f"{i}: {folder}")
    
    index = int(input("Select the folder index: "))
    selected_folder = folders[index]
    
    csv_dir = os.path.join(base_dir, selected_folder, 'csv2')
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return
    
    csv_path = os.path.join(csv_dir, csv_files[0])
    image_dir = os.path.join(base_dir, selected_folder, 'images')
    transformed_image_dir = os.path.join(base_dir, selected_folder, 'transformed_images')
    
    display_images_and_label(csv_path, image_dir, transformed_image_dir)

if __name__ == "__main__":
    main()
