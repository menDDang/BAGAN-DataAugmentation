import argparse
from utils.data_loader import *
from utils.hparams import HParam
from models.lstm_classifier import classifier
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')
    parser.add_argument('--train_data_ratio', default=1, type=float, help='ratio of data to use for training, 0<r<=1')

    args = parser.parse_args()
    hp = HParam(args.config)

    # Set log directory & summary writer
    if args.train_data_ratio == 1:
        logdir = "./logs/lstm_classifier/fullset"
    else:
        logdir = "./logs/lstm_classifier/subset(r=" + str(args.train_data_ratio) + ")"
    writer = tf.summary.create_file_writer(logdir)

    # Create data sets
    train_dataset = Dataset(mode='training', hp=hp, r=args.train_data_ratio)
    valid_dataset = Dataset(mode='validation', hp=hp)

    # Build LSTM end-to-end classifier
    model = classifier(hp)

    for epoch in range(hp.train.lstm_train_epoch_num):
        batch_x, batch_y = train_dataset.get_batch(hp.train.lstm_batch_size)
        train_loss = model.train_on_batch(batch_x, batch_y)

        # Write summary
        with writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)

        # Validate model
        if epoch % hp.train.lstm_valid_interval == 0:
            valid_loss, valid_error_rate = model.evaluate(valid_dataset)
            with writer.as_default():
                tf.summary.scalar('valid_loss', valid_loss, step=epoch)
                tf.summary.scalar('valid_error_rate', valid_error_rate, step=epoch)

        print("Epoch : {}, Train Loss : {}".format(epoch, '%1.4f' % train_loss))


