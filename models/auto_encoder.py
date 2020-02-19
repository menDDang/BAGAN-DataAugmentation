from .modules import *
import tensorflow as tf
import numpy as np

class AutoEncoder(tf.keras.Model):
    def __init__(self, hp):
        super(AutoEncoder, self).__init__()

        # Build model
        self.encoder = create_encoder(embed_dims=hp.autoencoder.embed_dims)
        self.decoder = create_decoder(time_length=hp.audio.time_length, feat_dim=hp.audio.n_mels)

        self.optimizer = None
        self.loss_fn = None

    def compile(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, inputs):
        # Encoding
        inputs = tf.expand_dims(inputs, -1)
        embeddings = self.encoder(inputs)

        # Decoding
        reconstructed = self.decoder(embeddings)
        reconstructed = tf.squeeze(reconstructed, -1)

        return reconstructed

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, embeddings):
        return self.decoder(embeddings)

    def train_on_batch(self, x):
        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            loss = self.loss_fn(y_true=x, y_pred=y_pred)
            trainable_variables = self.encoder.trainable_variables
            trainable_variables += self.decoder.trainable_variables
            grads = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(grads, trainable_variables))

            return loss

    def save(self, encoder_name, decoder_name):
        self.encoder.save_weights(encoder_name)
        self.decoder.save_weights(decoder_name)

    def evaluate(self, dataset):
        loss_list = []
        for key, x in dataset.x.items():
            x_rec = self.call(x)
            loss = self.loss_fn(y_pred=x_rec, y_true=x)
            loss_list.append(loss)

        return tf.reduce_mean(loss_list)