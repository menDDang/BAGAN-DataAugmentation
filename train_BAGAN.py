import tensorflow as tf
from models.balancingGAN import BAGAN
import argparse
from utils.hparams import HParam
from utils.data_loader import BAGANDataset
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')
    parser.add_argument('--train_data_ratio', default=0.2, type=float, help='ratio of data to use for training, 0<r<=1')
    parser.add_argument('--log_dir', required=True, type=str, help='directory to store summaries')
    parser.add_argument('--chkpt_dir', default='chkpt/bagan', type=str)
    parser.add_argument('--embedding_path', required=True, type=str)

    args = parser.parse_args()
    hp = HParam(args.config)

    # Set log directory & summary writer
    logdir = args.log_dir
    writer = tf.summary.create_file_writer(logdir)

    # Create data sets
    train_dataset = BAGANDataset(mode='training', hp=hp, r=args.train_data_ratio)
    valid_dataset = BAGANDataset(mode='validation', hp=hp)

    # Build model
    model = BAGAN(hp)

    # Load pre-trained encoder & decoder
    encoder_path = 'chkpt/autoencoder/r=0.2-v1/enc/chkpt-10000'
    decoder_path = 'chkpt/autoencoder/r=0.2-v1/dec/chkpt-10000'
    model.D.load_weights(encoder_path).expect_partial()
    model.G.load_weights(decoder_path).expect_partial()

    # Load mean & co-variance vectors
    for key in train_dataset.commands:
        model.load_embeddings(key=key, embedding_path=args.embedding_path)

    # Set loss function & optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    """
    D_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.bagan.initial_D_learning_rate,
        decay_steps=hp.bagan.train_epoch_num,
        decay_rate=hp.bagan.lr_decay_rate
    )
    G_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.bagan.initial_G_learning_rate,
        decay_steps=hp.bagan.train_epoch_num,
        decay_rate=hp.bagan.lr_decay_rate
    )
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=D_lr_scheduler)
    G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler)
    """
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.bagan.initial_D_learning_rate)
    G_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.bagan.initial_G_learning_rate)

    # Compile model
    model.compile(D_optimizer=D_optimizer, G_optimizer=G_optimizer, loss_fn=loss_fn)
    
    # Do training
    for epoch in range(hp.bagan.train_epoch_num):
        x, y = train_dataset.get_batch(hp.bagan.batch_size)
        D_loss, G_loss = model.train_on_batch(x, y)
        
        if epoch % 10 == 0:
            with writer.as_default():
                tf.summary.scalar('Train D Loss', D_loss, step=epoch)
                tf.summary.scalar('Train G Loss', G_loss, step=epoch)
            print('Epoch : {}, Train D Loss : {}, Train G Loss : {}'.format(epoch, '%1.4f' % D_loss, '%1.4f' % G_loss))
            
        if epoch % hp.bagan.valid_interval == 0:
            valid_x, valid_y = valid_dataset.get_batch(hp.bagan.batch_size)
            D_loss, D_acc, G_loss, G_acc = model.evaluate(x_real=valid_x, y=valid_y)
            with writer.as_default():
                tf.summary.scalar('Valid D Loss', D_loss, step=epoch)
                tf.summary.scalar('Valid D Accuracy', D_acc, step=epoch)
                tf.summary.scalar('Valid G Loss', G_loss, step=epoch)
                tf.summary.scalar('Valid G Accuracy', G_acc, step=epoch)
                
        if epoch % hp.bagan.chkpt_interval == 0:
            generator_path = os.path.join(args.chkpt_dir, "generator", "chkpt-" + str(epoch))
            discriminator_path = os.path.join(args.chkpt_dir, "discriminator", "chkpt-" + str(epoch))
            model.save(discriminator_path, generator_path)

    # Valid last
    valid_x, valid_y = valid_dataset.get_batch(hp.bagan.batch_size)
    D_loss, D_acc, G_loss, G_acc = model.evaluate(x_real=valid_x, y=valid_y)
    with writer.as_default():
        tf.summary.scalar('Valid D Loss', D_loss, step=hp.bagan.train_epoch_num)
        tf.summary.scalar('Valid D Accuracy', D_acc, step=hp.bagan.train_epoch_num)
        tf.summary.scalar('Valid G Loss', G_loss, step=hp.bagan.train_epoch_num)
        tf.summary.scalar('Valid G Accuracy', G_acc, step=hp.bagan.train_epoch_num)

    # Save last
    generator_path = os.path.join(args.chkpt_dir, "generator", "chkpt-" + str(hp.bagan.train_epoch_num))
    discriminator_path = os.path.join(args.chkpt_dir, "discriminator", "chkpt-" + str(hp.bagan.train_epoch_num))
    model.save(discriminator_path, generator_path)

    print("Optimization is Done!")