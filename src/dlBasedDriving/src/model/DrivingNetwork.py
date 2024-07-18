import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

class DrivingNetwork:
    def __init__(self, input_shape):
        self.model = self.create_network(input_shape)

    def create_network(self, input_shape):
        image_input = Input(shape=input_shape)
        resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)

        # 마지막 몇 개 층을 학습 가능하도록 설정
        for layer in resnet.layers[:-10]:
            layer.trainable = False

        x = Flatten()(resnet.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Decision network
        fc1 = Dense(256, activation='relu')(x)
        fc1 = Dropout(0.5)(fc1)

        steering_output = Dense(1, activation='tanh', name='steering_output')(fc1)

        model = Model(inputs=image_input, outputs=steering_output)
        return model

if __name__ == "__main__":
    input_shape = (224, 224, 3)  # 예시 입력 크기
    driving_network = DrivingNetwork(input_shape)
    driving_network.model.summary()
