import numpy as np
import os
from utils.audio import Audio
from tqdm import tqdm

import argparse
from utils.hparams import HParam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')

    args = parser.parse_args()
    hp = HParam(args.config)

    audio = Audio(hp)

    for task in ['training', 'testing', 'validation']:
        for word in os.listdir(os.path.join(hp.path.data_dir, task)):
            print("Extracting {}-{}".format(task, word))

            # Set directories
            indir = os.path.join(hp.path.data_dir, task, word)
            f_out_name = os.path.join(hp.path.feat_dir, task, word + ".npy")
            os.makedirs(os.path.join(hp.path.feat_dir, task), exist_ok=True)

            # Set filelist
            filelist = [filename for filename in os.listdir(indir)]

            # Extract
            mfcc_vectors = []
            for filename in tqdm(filelist):
                f_in_name = os.path.join(indir, filename)

                wav = audio.load_wav(f_in_name)
                wav = audio.rescale(wav)
                mfcc = audio.mfcc(wav).T

                # To make fixed size inputs for models
                while mfcc.shape[0] > hp.audio.time_length:
                    mfcc = np.delete(mfcc, 1, 0)
                while mfcc.shape[0] < hp.audio.time_length:
                    padding = np.zeros(hp.audio.n_mfcc)
                    mfcc = np.vstack([mfcc, padding])
                
                mfcc_vectors.append(mfcc)

            mfcc_vectors = np.array(mfcc_vectors)
            print(mfcc_vectors.shape)
            np.save(f_out_name, mfcc_vectors)
