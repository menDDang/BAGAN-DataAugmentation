import argparse
from utils.data_loader import *
from utils.hparams import HParam
from models.classifier import *
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')
    parser.add_argument('--train_data_ratio', default=1, type=float, help='ratio of data to use for training, 0<r<=1')
    parser.add_argument('--log_dir', required=True, type=str, help='directory to store summaries')
    parser.add_argument('--chkpt_dir', default='chkpt/classifier', type=str)

    args = parser.parse_args()
    hp = HParam(args.config)

    # Set log directory & summary writer
    logdir = args.log_dir
    writer = tf.summary.create_file_writer(logdir)

    # Create data sets
    train_dataset = Dataset(mode='training', hp=hp, r=args.train_data_ratio)
    valid_dataset = Dataset(mode='validation', hp=hp)

    # Build end-to-end classifier.py
    model = ResNetClassifier()

    # Set optimizer & loss function
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.classifier.initial_learning_rate,
        decay_steps=hp.classifier.train_epoch_num,
        decay_rate=hp.classifier.lr_decay_rate
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Compile model
    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    for epoch in range(hp.classifier.train_epoch_num):
        batch_x, batch_y = train_dataset.get_batch(hp.classifier.batch_size)
        train_loss = model.train_on_batch(batch_x, batch_y)

        # Write summary
        with writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)

        # Validate model
        if epoch % hp.classifier.valid_interval == 0:
            valid_loss, valid_mean_eer = model.evaluate(dataset=valid_dataset)
            with writer.as_default():
                tf.summary.scalar('valid_loss', valid_loss, step=epoch)
                tf.summary.scalar('valid_mean_eer', valid_mean_eer, step=epoch)

        # Save check point
        if epoch != 0 and epoch % hp.classifier.chkpt_interval == 0:
            path = os.path.join(args.chkpt_dir, "chkpt-" + str(epoch))
            model.save_weights(path)

        print("Epoch : {}, Train Loss : {}".format(epoch, '%1.4f' % train_loss))

    # Validate last
    valid_loss, valid_mean_eer = model.evaluate(dataset=valid_dataset)
    with writer.as_default():
        tf.summary.scalar('valid_loss', valid_loss, step=hp.classifier.train_epoch_num)
        tf.summary.scalar('valid_mean_eer', valid_mean_eer, step=hp.classifier.train_epoch_num)

    # Save last
    path = os.path.join(args.chkpt_dir, "chkpt-" + str(hp.classifier.train_epoch_num))
    model.save_weights(path)

    print("Optimization is Done!")