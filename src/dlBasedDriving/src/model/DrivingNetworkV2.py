import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

class drivingNetworkV2:
    def __init__(self, input_shape):
        self.model = self.create_network(input_shape)

    def create_network(self, input_shape):
        image_input = Input(shape=input_shape)

        # Layer 1
        x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(image_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Layer 2
        x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Layer 3
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)

        # Layer 4
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)

        # Layer 5
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # Flatten
        x = Flatten()(x)

        # Fully connected layer 1
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Fully connected layer 2
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Output layer
        output = Dense(5, activation='softmax', name='output')(x)

        model = Model(inputs=image_input, outputs=output)
        return model

if __name__ == "__main__":
    input_shape = (320, 180, 1)  # 이진화된 이미지의 예시 입력 크기
    custom_alexnet = CustomAlexNet(input_shape)
    custom_alexnet.model.summary()
