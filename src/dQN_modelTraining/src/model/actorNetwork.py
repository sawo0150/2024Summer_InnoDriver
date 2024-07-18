import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

class ActorNetwork:
    def __init__(self, input_shape):
        self.model = self.create_network(input_shape)

    def create_network(self, input_shape):
        image_input = Input(shape=input_shape)
        resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
        for layer in resnet.layers:
            layer.trainable = False
        x = Flatten()(resnet.output)

        motor_power_input = Input(shape=(1,))
        steering_angle_input = Input(shape=(1,))

        concatenated = Concatenate()([x, motor_power_input, steering_angle_input])
        fc1 = Dense(256, activation='relu')(concatenated)
        fc2 = Dense(256, activation='relu')(fc1)

        motor_output = Dense(1, activation='sigmoid', name='motor_output')(fc2)
        steering_output = Dense(1, activation='tanh', name='steering_output')(fc2)

        model = Model(inputs=[image_input, motor_power_input, steering_angle_input], outputs=[motor_output, steering_output])
        return model
