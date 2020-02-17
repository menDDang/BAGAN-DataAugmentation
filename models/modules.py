from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf

def create_encoder():
    encoder = Sequential([
        Conv2D(16, (2, 2), activation='relu', padding='same'),
        MaxPooling2D((3, 2), padding='same'),
        Conv2D(32, (2, 2), activation='relu', padding='same'),
        MaxPooling2D((3, 2), padding='same'),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        Flatten(),
        Dense(256)])

    return encoder


def create_decoder():
    decoder = Sequential([
        Dense(5 * 10 * 128),
        Reshape((5, 10, 128)),
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        UpSampling2D((3, 2)),
        Conv2D(32, (2, 2), activation='relu', padding='same'),
        UpSampling2D((3, 2)),
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')])

    return decoder


class Resnet(tf.keras.Model):
    def __init__(self, filters):
        super(Resnet, self).__init__()
        self.net = tf.keras.Sequential([
            Conv2D(filters, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
        ])

    def call(self, x):
        return self.net(x) + x

