#!/usr/bin/env python3

from model.DrivingNetwork import DrivingNetwork
import os
import cv2
import csv
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tqdm import tqdm
from sklearn.model_selection import train_test_split


print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Warp transform 관련 함수 정의
def load_warp_transform_matrix(directory, file_name='warp_matrix.pkl'):
    file_path = os.path.join(directory, file_name)
    try:
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"Transform matrix file not found at {file_path}. Please generate it first.")

def warp_transform(cv_image, warp_matrix, width=224, height=224):
    top_view = cv2.warpPerspective(cv_image, warp_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return top_view

# 데이터 로드 및 전처리 함수 정의
def load_data_from_multiple_csv(data_dir, warp_matrix):
    images = []
    angles = []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for csv_file in csv_files:
        with open(os.path.join(data_dir, csv_file), 'r') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader, desc=f"Loading {csv_file}"):
                image_path = row['image']
                image = cv2.imread(image_path)
                transformed_image = warp_transform(image, warp_matrix)
                images.append(transformed_image)
                angles.append(float(row['goal_steering_angle']))
    return np.array(images), np.array(angles)

def main():
    # 데이터 경로 설정
    data_dir = '/home/innodriver/InnoDriver_ws/src/dQN_modelTraining/offPolicyLearningData'
    warp_matrix_dir = '/home/innodriver/InnoDriver_ws/src/visionMapping/src/warpMatrix'
    
    # Warp transform matrix 로드
    warp_matrix = load_warp_transform_matrix(warp_matrix_dir)

    # 데이터 로드 및 전처리
    images, angles = load_data_from_multiple_csv(data_dir, warp_matrix)

    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.2, random_state=42)

    # 모델 초기화
    input_shape = (224, 224, 3)
    driving_network = DrivingNetwork(input_shape)
    model = driving_network.model

    # 체크포인트 설정
    checkpoint_path = '/home/innodriver/InnoDriver_ws/src/dlBasedDriving/src/model/driving_model_checkpoint.h5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

    # 로그 설정
    log_dir = '/home/innodriver/InnoDriver_ws/src/dlBasedDriving/src/model/training_log.csv'
    csv_logger = CSVLogger(log_dir, append=True)

    # 모델 컴파일
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    # 미리 학습된 모델이 있으면 로드
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print("Loaded pre-trained model.")

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=32,
        callbacks=[checkpoint, csv_logger],
        verbose=1
    )

    # 학습 과정 시각화
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()