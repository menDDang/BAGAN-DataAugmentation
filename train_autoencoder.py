import tensorflow as tf
from models.auto_encoder import AutoEncoder
import argparse
from utils.hparams import HParam
from utils.data_loader import BAGANDataset
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')
    parser.add_argument('--train_data_ratio', default=1, type=float, help='ratio of data to use for training, 0<r<=1')
    parser.add_argument('--log_dir', required=True, type=str, help='directory to store summaries')
    parser.add_argument('--chkpt_dir', default='chkpt/autoencoder', type=str)

    args = parser.parse_args()
    hp = HParam(args.config)


    # Set log directory & summary writer
    logdir = args.log_dir
    writer = tf.summary.create_file_writer(logdir)

    # Create data sets
    train_dataset = BAGANDataset(mode='training', hp=hp, r=args.train_data_ratio)
    valid_dataset = BAGANDataset(mode='validation', hp=hp)

    # Build model
    model = AutoEncoder(hp)

    # Set loss function & optimizer
    loss_fn = tf.keras.losses.MeanSquaredError()
    #loss_fn = tf.keras.losses.CategoricalCrossentropy()
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.autoencoder.initial_learning_rate,
        decay_steps=hp.autoencoder.train_epoch_num,
        decay_rate=hp.autoencoder.lr_decay_rate
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    # Compile model
    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    # Do training
    for epoch in range(hp.autoencoder.train_epoch_num):
        x = train_dataset.get_batch(hp.autoencoder.batch_size)
        train_loss = model.train_on_batch(x)


        if epoch % hp.autoencoder.valid_interval == 0:
            valid_loss = model.evaluate(dataset=valid_dataset)
            with writer.as_default():
                tf.summary.scalar('valid_loss', valid_loss, step=epoch)

        if epoch % hp.autoencoder.chkpt_interval == 0:
            encoder_path = os.path.join(args.chkpt_dir, "enc", "chkpt-" + str(epoch))
            decoder_path = os.path.join(args.chkpt_dir, "dec", "chkpt-" + str(epoch))
            model.save(encoder_path, decoder_path)

        if epoch % 10 == 0:
            with writer.as_default():
                tf.summary.scalar('train_loss', train_loss, step=epoch)
            print("Epoch : {}, Train Loss : {}".format(epoch, '%1.4f' % train_loss))

    # Validate last
    valid_loss = model.evaluate(dataset=valid_dataset)
    with writer.as_default():
        tf.summary.scalar('valid_loss', valid_loss, step=hp.autoencoder.train_epoch_num)

    # Save last
    encoder_path = os.path.join(args.chkpt_dir, "enc", "chkpt-" + str(hp.autoencoder.train_epoch_num))
    decoder_path = os.path.join(args.chkpt_dir, "dec", "chkpt-" + str(hp.autoencoder.train_epoch_num))
    model.save(encoder_path, decoder_path)

    print("Optimization is Done!")

    x = valid_dataset.get_batch(1)
    x_rec = model(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.imshow(x[0])
    ax2.imshow(x_rec[0])
    plt.savefig('sample1.png')
    plt.close()