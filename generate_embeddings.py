import os
import numpy as np
import argparse
from utils.hparams import HParam
from utils.data_loader import AutoencoderDataset
from models.modules import create_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')
    parser.add_argument('--train_data_ratio', default=0.2, type=float, help='ratio of data to use for training, 0<r<=1')
    parser.add_argument('--chkpt_dir', required=True, type=str)

    args = parser.parse_args()
    hp = HParam(args.config)

    # Create data sets
    train_dataset = AutoencoderDataset(mode='training', hp=hp, r=args.train_data_ratio)
    valid_dataset = AutoencoderDataset(mode='validation', hp=hp)

    # Build encoder
    encoder = create_encoder(hp.autoencoder.embed_dims)

    # Load weights
    encoder.load_weights(args.chkpt_dir).expect_partial()

    # Generate embeddings for train set
    os.makedirs(hp.path.embed_dir + '/training', exist_ok=True)
    for key, x in train_dataset.x.items():
        embeddings = []

        i = 0
        while True:
            if i + 100 < len(x):
                inputs = np.expand_dims(x[i:i+100], -1)
                embeddings += [np.array(encoder(inputs), dtype=np.float32)]
                i += 100
            else:
                inputs = np.expand_dims(x[i:], -1)
                embeddings += [np.array(encoder(inputs), dtype=np.float32)]
                break

        embeddings = np.vstack(embeddings)
        mean = np.mean(embeddings, axis=0)
        var = np.cov(np.transpose(embeddings))

        save_path = os.path.join(hp.path.embed_dir, 'training', key + '-mean')
        np.save(save_path, mean)
        save_path = os.path.join(hp.path.embed_dir, 'training', key + '-variance')
        np.save(save_path, var)

    # Generate embeddings for valid set
    os.makedirs(hp.path.embed_dir + '/validation', exist_ok=True)
    for key, x in valid_dataset.x.items():
        embeddings = []

        i = 0
        while True:
            if i + 100 < len(x):
                inputs = np.expand_dims(x[i:i + 100], -1)
                embeddings += [np.array(encoder(inputs), dtype=np.float32)]
                i += 100
            else:
                inputs = np.expand_dims(x[i:], -1)
                embeddings += [np.array(encoder(inputs), dtype=np.float32)]
                break

        embeddings = np.vstack(embeddings)
        mean = np.mean(embeddings, axis=0)

        save_path = os.path.join(hp.path.embed_dir, 'validation', key + '-mean')
        np.save(save_path, mean)
        save_path = os.path.join(hp.path.embed_dir, 'validation', key + '-variance')
        np.save(save_path, var)
