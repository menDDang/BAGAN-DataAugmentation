import tensorflow as tf
from models.modules import *
import numpy as np
import os


class BAGAN(tf.keras.Model):
    def __init__(self, hp):
        super(BAGAN, self).__init__()
        self.hp = hp
        self.embed_dims = hp.autoencoder.embed_dims
        self.key_word_num = 10  # number of keyword
        self.output_num = self.key_word_num + 1  # one for fake label

        # Set generator
        self.G = self.create_generator()

        # Set discriminator
        self.D = self.create_discriminator()

        # Set empty dictionary for embeddings
        self.embeddings = dict()

    def generate(self, embeddings):
        return self.G(embeddings)

    def discriminate(self, sample):
        if len(sample.shape) == 3:
            sample = tf.expand_dims(sample, -1)

        return self.D(sample)

    def create_generator(self):
        decoder = create_decoder(time_length=self.hp.audio.time_length, feat_dim=self.hp.audio.n_mels)
        return decoder

    def create_discriminator(self):
        encoder = create_encoder(embed_dims=self.embed_dims)
        encoder.add(tf.keras.layers.Dense(self.output_num, activation='softmax'))
        return encoder

    def load_embeddings(self, key, embedding_path):
        mean_vector_path = os.path.join(embedding_path, key + '-mean.npy')
        mean = np.load(mean_vector_path)
        covariance_vector_path = os.path.join(embedding_path, key + '-variance.npy')
        cov = np.load(covariance_vector_path)
        self.embeddings[key] = {'mean': mean, 'cov': cov}

    def generate_noise(self, size):
        noise = []
        for key in self.embeddings:
            mean = self.embeddings[key]['mean']
            cov = self.embeddings[key]['cov']
            #print(np.random.multivariate_normal(mean=mean, cov=cov).shape)
            noise += [np.random.multivariate_normal(mean=mean, cov=cov) for i in range(int(size))]

        noise = np.stack(noise, 0)
        return noise

    def train_on_batch(self, x_real, y):
        # Generate noise
        size = int(len(x_real) / self.key_word_num)
        noise = self.generate_noise(size)

        # Generate fake samples
        x_fake = self.generate(noise)

        # Train discriminator
        with tf.GradientTape() as tape:
            # Discriminate fake samples
            y_fake = np.zeros_like(y)
            y_fake[:, -1] = 1  # labeled as fake
            D_loss_fake = self.loss_fn(y_pred=self.discriminate(x_fake), y_true=y_fake)

            # Discriminate real samples
            D_loss_real = self.loss_fn(y_pred=self.discriminate(x_real), y_true=y)

            # Calculate gradients & update discriminator
            D_loss = (D_loss_fake + D_loss_real) / 2
            grads = tape.gradient(D_loss, self.D.trainable_variables)
            self.D_optimizer.apply_gradients(zip(grads, self.D.trainable_variables))

        # Train generator
        with tf.GradientTape() as tape:
            x_fake = self.generate(noise)
            y_pred = self.discriminate(x_fake)
            G_loss = self.loss_fn(y_pred=y_pred, y_true=y)

            # Calculate gradients & update discriminator
            grads = tape.gradient(G_loss, self.G.trainable_variables)
            self.G_optimizer.apply_gradients(zip(grads, self.G.trainable_variables))

        return D_loss, G_loss

    def evaluate(self, x_real, y):
        # Generate noise
        size = len(x_real) / self.key_word_num
        noise = self.generate_noise(size)

        # Generate fake samples
        x_fake = self.generate(noise)

        y_fake = np.zeros_like(y)
        y_fake[:, -1] = 1

        # Calculate D_loss
        D_fake = self.discriminate(x_fake)
        D_loss_fake = self.loss_fn(y_pred=D_fake, y_true=y_fake)
        D_real = self.discriminate(x_real)
        D_loss_real = self.loss_fn(y_pred=D_real, y_true=y)
        D_loss = (D_loss_fake + D_loss_real) / 2

        # Calculate D_acc
        D_acc_fake = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(D_fake, axis=1), tf.argmax(y_fake, axis=1)), tf.float32))
        D_acc_real = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(D_real, axis=1), tf.argmax(y, axis=1)), tf.float32))
        D_acc = (D_acc_fake + D_acc_real) / 2

        # Calculate G_loss
        G_loss = self.loss_fn(y_pred=D_fake, y_true=y)

        # Calculate G_acc
        G_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(D_fake, axis=1), tf.argmax(y, axis=1)), tf.float32))

        return D_loss, D_acc, G_loss, G_acc

    def compile(self, D_optimizer, G_optimizer, loss_fn):
        self.D_optimizer = D_optimizer
        self.G_optimizer = G_optimizer
        self.loss_fn = loss_fn

    def save(self, D_path, G_path):
        self.D.save_weights(D_path)
        self.G.save_weights(G_path)