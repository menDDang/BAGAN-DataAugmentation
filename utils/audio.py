import librosa
import numpy as np
import lws

class Audio:
    def __init__(self, hp):
        self.hp = hp
        self.mel_basis = librosa.filters.mel(hp.audio.sampling_rate,
                                             hp.audio.n_fft,
                                             fmin=hp.audio.fmin,
                                             fmax=hp.audio.fmax,
                                             n_mels=hp.audio.n_mels)

    def load_wav(self, filename):
        return librosa.core.load(filename, sr=self.hp.audio.sampling_rate)[0]

    def melspectrogram(self, y):
        D = lws.lws(self.hp.audio.n_fft, self.hp.audio.hop_length, mode='speech').stft(y).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hp.audio.ref_level_db

        return self.normalize(S)


    def _linear_to_mel(self, spectrogram):
        return np.dot(self.mel_basis, spectrogram)

    def _amp_to_db(self, x):
        min_level = np.exp(self.hp.audio.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def normalize(self, S):
        return np.clip((S - self.hp.audio.min_level_db) / -1 * self.hp.audio.min_level_db, 0, 1)

    def denormalize(self, S):
        return (np.clip(S, 0, 1) * -1 * self.hp.audio.min_level_db) + self.hp.audio.min_level_db


if __name__ == "__main__":
    import argparse
    import hparams

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml', help='confuguration file')

    args = parser.parse_args()
    hp = hparams.HParam(args.config)

    audio = Audio(hp)

    filename = "datasets/testing/backward/0c540988_nohash_0.wav"

    wav = audio.load_wav(filename)
    print('length of wav : {}'.format(len(wav)))

    mel = audio.melspectrogram(wav)
    print(mel.shape)