import numpy as np
import tensorflow as tf
from models.modules import Resnet

class classifier(tf.keras.Model):
    def __init__(self, hp):
        super(classifier, self).__init__()
        self.hp = hp

        lstm_neuron_num = hp.model.lstm_neuron_num
        time_length = hp.audio.time_length
        num_mels = hp.audio.n_mels
        output_num = 10 + 1 # 10 : number of wuw, 1 : unknown

        lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=hp.train.lstm_learning_rate,
            decay_steps=hp.train.lstm_train_epoch_num,
            end_learning_rate=0.00001
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        #self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(45, (3, 3), padding='same', activation='relu'))
        for i in range(6):
            self.model.add(Resnet(45))
        self.model.add(tf.keras.layers.Conv2D(45, (3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.Conv2D(45, (3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.AveragePooling2D((3, 3)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(output_num, activation='softmax'))

        """    
        self.model.add()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8, (2,2), padding='same', activation='relu'),     # [90, 80, 8]
            tf.keras.layers.MaxPool2D(2),                                            # [45, 40, 8]
            tf.keras.layers.Conv2D(16, (2,2), padding='same', activation='relu'),    # [45, 40, 16]
            tf.keras.layers.MaxPool2D((3,2)),                                        # [15, 20, 16]
            tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu'),   # [15, 20, 32]
            tf.keras.layers.MaxPool2D((3, 2)),                                       # [5, 10, 32]
            tf.keras.layers.Conv2D(64, (2, 2), padding='same', activation='relu'),   # [5, 10, 64]
            tf.keras.layers.Conv2D(128, (2, 2), padding='same', activation='relu'),  # [5, 10, 128]
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_num, activation='softmax')
        ])
        """
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(lstm_neuron_num, input_shape=[time_length, num_mels], return_sequences=True),
            tf.keras.layers.LSTM(lstm_neuron_num, return_sequences=True),
            tf.keras.layers.LSTM(lstm_neuron_num),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_num, activation='softmax')
        ])
        """



    def call(self, x):
        x = tf.expand_dims(x, -1)
        return self.model(x)

    def train_on_batch(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            y_pred = self.call(batch_x)
            train_loss = self.loss_fn(batch_y, y_pred)
            grad = tape.gradient(train_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        return train_loss

    def evaluate(self, x, y_true):
        loss_list, error_rate_list, i = [], [], 0
        while (i + 100 < len(x)):
            y_pred = self.call(x[i:i+100])
            loss = self.loss_fn(y_true=y_true[i:i+100], y_pred=y_pred)
            loss_list.append(loss)
            error_rate = 1 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_true[i:i+100], axis=1)), tf.float32))
            error_rate_list.append(error_rate)
            i += 100

        return tf.reduce_mean(loss_list), tf.reduce_mean(error_rate_list)



    def evaluate0(self, dataset):
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
                y_true[:, -1] = 1
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