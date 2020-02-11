from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from utils.audio import Audio


import argparse
from utils.hparams import HParam


def build_from_path(in_dir, out_dir, hp, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    audio = Audio(hp)

    labels = os.listdir(in_dir)
    for label in tqdm(labels):
        mel_vectors = []
        futures = []

        wavfiles = [in_dir + label + '/' + wavfile for wavfile in os.listdir(in_dir + label)]
        for wavfile in wavfiles:
            futures.append(executor.submit(
                partial(audio.melspectrogram, wavfile)))

        for future in tqdm(futures):
            mel = future.result()

            # To make fixed size inputs for models
            while mel.shape[0] > 90:
                mel = np.delete(mel, 1, 0)
            while mel.shape[0] < 90:
                padding = np.zeros(hp.audio.)
                mel = np.vstack([mel, padding])

            # print('mel shape', np.shape(mel))
            mel_vectors.append(mel)

        print('mel shape: ', str(np.shape(mel_vectors)))
        np.save(out_dir + label + '.npy', mel_vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')

    args = parser.parse_args()
    hp = HParam(args.config)

    for task in ['train', 'test', 'valid']:
        # Set directories
        indir = os.path.join(hp.path.data_dir, task)
        outdir = os.path.join(hp.path.feat_dir, task)
        # Create directory to store features
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        build_from_path(indir, outdir, hp)

