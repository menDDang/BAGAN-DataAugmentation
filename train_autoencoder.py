import tensorflow as tf
from models.auto_encoder import AutoEncoder
import argparse
from utils.hparams import HParam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')

    args = parser.parse_args()
    hp = HParam(args.config)

    autoencoder = AutoEncoder(hp)

    x = tf.zeros(shape=[100, 100, 100, 100], dtype=tf.float32)
    y = autoencoder(x)

    print(y.shape)