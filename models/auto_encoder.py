from .modules import *
import tensorflow as tf


class AutoEncoder(tf.keras.Model):
    def __init__(self, hp):
        super(AutoEncoder, self).__init__()

        # Build model
        self.encoder = create_encoder()
        self.decoder = create_decoder()

        # Set loss function & optimizer
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.train.autoencoder_learning_rate)

    def call(self, inputs):
        embeddings = self.encoder(inputs)
        reconstructed = self.decoder(embeddings)

        return reconstructed


    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, embeddings):
        return self.decoder(embeddings)