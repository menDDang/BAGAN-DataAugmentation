import numpy as np
import tensorflow as tf


class ResNetClassifier(tf.keras.Model):
    def __init__(self):
        super(ResNetClassifier, self).__init__()

        # Build model
        self.conv1 = tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu')
        self.res1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(2, 2)),
            tf.keras.layers.BatchNormalization(),
        ])
        self.res2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(2, 2,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(2, 2)),
            tf.keras.layers.BatchNormalization(),
        ])
        self.res3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(4, 4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(4, 4)),
            tf.keras.layers.BatchNormalization(),
        ])
        self.res4 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(4, 4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(4, 4)),
            tf.keras.layers.BatchNormalization(),
        ])
        self.res5 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(4, 4)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(4, 4)),
            tf.keras.layers.BatchNormalization(),
        ])
        self.res6 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(8, 8)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(8, 8)),
            tf.keras.layers.BatchNormalization(),
        ])
        self.conv2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(16, 16)),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(16, 16)),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu',
                                   dilation_rate=(16, 16)),
            tf.keras.layers.Conv2D(filters=45, kernel_size=(3, 3), padding='same', activation='relu',
                                   dilation_rate=(16, 16)),
            tf.keras.layers.AveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(11, activation='softmax')
        ])

        #
        self.optimizer = None
        self.loss_fn = None

    def compile(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, x):
        x = tf.expand_dims(x, -1)
        x = self.conv1(x)
        x = self.res1(x) + x
        x = self.res2(x) + x
        x = self.res3(x) + x
        x = self.res4(x) + x
        x = self.res5(x) + x
        x = self.res6(x) + x
        x = self.conv2(x)
        return x

    def train_on_batch(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            y_pred = self.call(batch_x)
            train_loss = self.loss_fn(y_true=batch_y, y_pred=y_pred)

            variable_list = self.conv1.trainable_variables
            variable_list += self.res1.trainable_variables
            variable_list += self.res2.trainable_variables
            variable_list += self.res3.trainable_variables
            variable_list += self.res4.trainable_variables
            variable_list += self.res5.trainable_variables
            variable_list += self.res6.trainable_variables
            variable_list += self.conv2.trainable_variables

            grad = tape.gradient(train_loss, variable_list)
            self.optimizer.apply_gradients(zip(grad, variable_list))

        return train_loss

    def evaluate_old(self, x, y_true):
        loss_list, error_rate_list, i = [], [], 0
        while (i + 100 < len(x)):
            y_pred = self.call(x[i:i+100])
            loss = self.loss_fn(y_true=y_true[i:i+100], y_pred=y_pred)
            loss_list.append(loss)
            error_rate = 1 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_true[i:i+100], axis=1)), tf.float32))
            error_rate_list.append(error_rate)
            i += 100

        return tf.reduce_mean(loss_list), tf.reduce_mean(error_rate_list)

    def evaluate(self, dataset):
        # Predict
        y_pred = dict()
        for key, x in dataset.x.items():
            tmp = []
            i = 0
            while True:
                if i + 100 < len(x):
                    tmp += [np.array(self.call(x[i:i + 100]), dtype=np.float32)]
                    i += 100
                else:
                    tmp += [np.array(self.call(x[i:]), dtype=np.float32)]
                    break

            tmp = np.vstack(tmp)
            y_pred[key] = tmp

        # Calculate loss
        loss_list = []
        for key in y_pred:
            y_true = np.zeros(shape=[len(y_pred[key]), 11])
            if key == 'unknown':
                y_true[:, -1] = 1
            else:
                y_true[:, dataset.commands[key]] = 1

            loss_list.append(self.loss_fn(y_true, y_pred[key]))
        loss = np.mean(loss_list)

        # Calculate mean EER
        mean_eer = []
        for key, x in dataset.x.items():
            if key == 'unknown':
                continue

            y_true = dataset.commands[key]
            eer, diff = 1, 1
            for threshold in [i * 0.01 for i in range(100)]:
                correct_num, error_num = 0, 0
                for j in range(len(y_pred[key])):
                    if y_pred[key][j, y_true] > threshold:
                        correct_num += 1
                    else:
                        error_num += 1
                FAR = float(error_num) / len(y_pred[key])

                correct_num, error_num = 0, 0
                for j in range(len(y_pred['unknown'])):
                    if y_pred['unknown'][j, y_true] > threshold:
                        error_num += 1
                    else:
                        correct_num += 1
                FFR = float(error_num) / len(y_pred['unknown'])

                if diff > abs(FAR - FFR):
                    diff = abs(FAR - FFR)
                    eer = (FAR + FFR) / 2

            # print(key, eer)
            mean_eer.append(eer)
        mean_eer = np.mean(mean_eer)

        return loss, mean_eer

