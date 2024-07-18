#!/usr/bin/env python3

from model.DrivingNetworkV2 import drivingNetworkV2
import os
import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_accuracy', mode='max', save_best_only=True):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_val_accuracy = -float('inf') if mode == 'max' else float('inf')
        self.best_train_accuracy = -float('inf') if mode == 'max' else float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        current_train_accuracy = logs.get('accuracy')
        
        if self.mode == 'max':
            if (current_val_accuracy is not None and current_val_accuracy > self.best_val_accuracy) or \
               (current_train_accuracy is not None and current_train_accuracy > self.best_train_accuracy):
                self.best_val_accuracy = max(current_val_accuracy, self.best_val_accuracy)
                self.best_train_accuracy = max(current_train_accuracy, self.best_train_accuracy)
                self.model.save(os.path.join(self.filepath.format(epoch=epoch + 1)))
        else:
            if (current_val_accuracy is not None and current_val_accuracy < self.best_val_accuracy) or \
               (current_train_accuracy is not None and current_train_accuracy < self.best_train_accuracy):
                self.best_val_accuracy = min(current_val_accuracy, self.best_val_accuracy)
                self.best_train_accuracy = min(current_train_accuracy, self.best_train_accuracy)
                self.model.save(os.path.join(self.filepath.format(epoch=epoch + 1)))

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (320, 180))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary Mask', binary)
    # cv2.waitKey(10000)
    binary = binary / 255.0  # Normalize to [0, 1]
    binary = np.expand_dims(binary, axis=-1)  # Add channel dimension
    return binary
    # image = image / 255.0  # Normalize to [0, 1]
    # return image

def load_data(base_dir):
    images = []
    labels = []
    label_map = {str(i): i+2 for i in range(-2, 3)}
    print(label_map)
    
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if dir == 'csv2':
                csv_dir = os.path.join(root, dir)
                print(csv_dir)
                for csv_file in os.listdir(csv_dir):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(csv_dir, csv_file)
                        with open(csv_path, mode='r') as file:
                            reader = csv.DictReader(file)
                            for row in reader:
                                image_path = row["image"]
                                image = preprocess_image(image_path)
                                label = None
                                for i in range(-2, 3):
                                    if row[str(i)] == '1':
                                        label = i
                                        break
                                if label is not None:
                                    images.append(image)
                                    labels.append(label_map[str(label)])
                                else:
                                    print(f"No valid label found in row: {row}")
                break  # Stop further os.walk since we only want the csv2 directory of the current folder
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=5)
    return images, labels

def plot_loss(history, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_graph.png'))
    plt.show()

if __name__ == "__main__":
    input_shape = (180, 320, 1)  # 320x180 resized and binary images
    model_class = drivingNetworkV2(input_shape)
    model = model_class.model

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load data
    base_dir = '/home/innodriver/InnoDriver_ws/Data'
    images, labels = load_data(base_dir)

    checkpoint_dir = '/home/innodriver/InnoDriver_ws/DNv2checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_val_path = os.path.join(checkpoint_dir, 'best_model_val_loss.h5')
    checkpoint_train_path = os.path.join(checkpoint_dir, 'best_model_train_loss.h5')

    # Usage
    # Checkpoint settings
    checkpoint_val = ModelCheckpoint(checkpoint_val_path, monitor='val_loss', save_best_only=True, mode='min')
    checkpoint_train = ModelCheckpoint(checkpoint_train_path, monitor='loss', save_best_only=True, mode='min')

    csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'training_log.csv'))

    # Train the model
    history = model.fit(images, labels, validation_split=0.2, epochs=100, batch_size=64, callbacks=[checkpoint_val,checkpoint_train, csv_logger])

    # Plot and save the loss graph
    plot_loss(history, checkpoint_dir)
