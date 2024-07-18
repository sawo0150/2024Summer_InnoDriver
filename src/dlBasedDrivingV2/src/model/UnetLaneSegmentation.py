import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
class UnetLaneSegmentation:
    def __init__(self, input_size=(640, 640, 3), num_classes=3, learning_rate=1e-4):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(self.input_size)
    
        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(self.num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

        # return model

    def train(self, train_generator, val_generator, batch_size=4, epochs=50, checkpoint_dir='./checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_cb = callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_loss:.2f}.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True
        )
        
        history = self.model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=[checkpoint_cb])
        
        self.plot_history(history)
    
    def plot_history(self, history):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.show()

    def predict(self, image):
        return self.model.predict(np.expand_dims(image, axis=0))

    def save(self, path='unet_model.h5'):
        self.model.save(path)

    def load(self, path='unet_model.h5'):
        self.model.load_weights(path)

# 사용 예시

if __name__ == "__main__":
    # 모델 생성
    unet_model = UnetLaneSegmentation(input_size=(448, 300, 3), num_classes=2, learning_rate=1e-4)

    # 예시 데이터 로드 (사용자의 데이터로 대체 필요)
    train_images = np.random.rand(10, 448, 300, 3)  # 예시 데이터
    train_masks = np.random.rand(10, 448, 300, 2)  # 예시 데이터 (one-hot encoded)

    # 모델 학습
    unet_model.train(train_images, train_masks, batch_size=4, epochs=50)

    # 모델 저장
    unet_model.save('unet_model.h5')

    # 모델 로드
    unet_model.load('unet_model.h5')

    # 예시 이미지로 예측
    image = np.random.rand(448, 300, 3)
    pred = unet_model.predict(image)

    # 추론 시간 측정
    import time
    start_time = time.time()
    pred = unet_model.predict(image)
    end_time = time.time()

    print(f"Prediction time: {end_time - start_time} seconds")

    # 결과 이미지 시각화
    import cv2
    pred_mask = np.argmax(pred[0], axis=-1)
    cv2.imshow("Prediction", pred_mask.astype(np.uint8) * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
