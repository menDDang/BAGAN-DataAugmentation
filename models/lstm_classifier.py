import numpy as np
import tensorflow as tf


class classifier(tf.keras.Model):
    def __init__(self, hp):
        super(classifier, self).__init__()
        self.hp = hp

        lstm_neuron_num = hp.model.lstm_neuron_num
        time_length = hp.audio.time_length
        num_mels = hp.audio.n_mels
        output_num = 10 + 1 # 10 : number of wuw, 1 : unknown

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(lstm_neuron_num, input_shape=[time_length, num_mels], return_sequences=True),
            tf.keras.layers.LSTM(lstm_neuron_num, return_sequences=True),
            tf.keras.layers.LSTM(lstm_neuron_num),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(56, activation='relu'),
            tf.keras.layers.Dense(output_num)
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.train.lstm_learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, x):
        return self.model(x)

    def train_on_batch(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            y_pred = self.call(batch_x)
            train_loss = self.loss_fn(batch_y, y_pred)
            grad = tape.gradient(train_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        return train_loss

    def evaluate(self, dataset):
        # predict
        y_pred = dict()
        for key, data in dataset.x.items():
            y_pred[key] = self.call(data)

        loss_list = []
        error_rate_list = []
        for key, data in y_pred.items():
            # set label
            y_true = np.zeros(shape=[data.shape[0], 11], dtype=np.float32)
            if key == 'unknown':
                y_true[:, 10] = 1
            else:
                idx = dataset.commands[key]
                y_true[:, idx] = 1

            # calculate loss
            loss_list.append(self.loss_fn(y_true=y_true, y_pred=data))

            # calculate error rate
            error_rate_list.append(1 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(data, axis=1), tf.argmax(y_true, axis=1)), tf.float32)))


        mean_loss = tf.reduce_mean(loss_list)
        mean_error_rate = tf.reduce_mean(error_rate_list)

        return mean_loss, mean_error_rate