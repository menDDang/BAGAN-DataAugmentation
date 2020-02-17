import argparse
from utils.data_loader import *
from utils.hparams import HParam
from models.lstm_classifier import classifier
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')
    parser.add_argument('--train_data_ratio', default=1, type=float, help='ratio of data to use for training, 0<r<=1')
    parser.add_argument('--logdir', required=True, type=str, help='directory to store summaries')
    parser.add_argument('--chkpt_dir', default='chkpt/classifier', type=str)

    args = parser.parse_args()
    hp = HParam(args.config)

    # Set log directory & summary writer
    logdir = args.logdir
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
            loss_list = []
            error_rate_list = []
            for key, x in valid_dataset.x.items():
                y_true = np.zeros(shape=[x.shape[0], 11], dtype=np.float32)
                if key == 'unknown':
                    y_true[:, -1] = 1
                else:
                    idx = valid_dataset.commands[key]
                    y_true[:, idx] = 1
                valid_loss, valid_error_rate = model.evaluate(x, y_true)
                loss_list.append(valid_loss)
                error_rate_list.append(valid_error_rate)

            err = ""
            for error_rate in error_rate_list:
                err += str(error_rate) + " "
            print("Epoch : {}, Valid error rate : {}".format(epoch, err))

            with writer.as_default():
                tf.summary.scalar('valid_loss', tf.reduce_mean(loss_list), step=epoch)
                tf.summary.scalar('valid_error_rate', tf.reduce_mean(error_rate_list), step=epoch)

        # Save check point
        if epoch != 0 and epoch % hp.train.lstm_chkpt_interval == 0:
            path = os.path.join(args.chkpt_dir, "chkpt-" + str(epoch))
            model.save_weights(path)

        if epoch % 1000 == 0:
            print("Epoch : {}, Train Loss : {}".format(epoch, '%1.4f' % train_loss))

    path = os.path.join(args.chkpt_dir, "chkpt-" + str(hp.train.lstm_train_epoch_num))
    model.save_weights(path)
