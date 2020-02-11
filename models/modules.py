from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from tensorflow.keras.models import Sequential, Model


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