from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Sequential


def create_encoder(embed_dims):
    encoder = Sequential([
        Conv2D(16, (2, 2), activation='relu', padding='same'),
        MaxPooling2D((3, 2), padding='same'),
        Conv2D(32, (2, 2), activation='relu', padding='same'),
        MaxPooling2D((3, 2), padding='same'),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        Flatten(),
        Dense(embed_dims)
    ])

    return encoder


def create_decoder(time_length, feat_dim):
    decoder = Sequential([
        Dense((5 * 10 * 128)),
        Reshape((5, 10, 128)),
        Conv2D(128, (2, 2), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (2, 2), activation='relu', padding='same'),
        UpSampling2D((3, 2)),
        Conv2D(32, (2, 2), activation='relu', padding='same'),
        UpSampling2D((3, 2)),
        Conv2D(1, (3, 3), activation='relu', padding='same'),

        # To fit shape
        Flatten(),
        Dense(time_length * feat_dim, activation='sigmoid'),
        Reshape((time_length, feat_dim, 1))
    ])
    return decoder

