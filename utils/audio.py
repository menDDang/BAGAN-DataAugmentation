import librosa
import numpy as np


class Audio:
    def __init__(self, hp):
        self.hp = hp

    def melspectrogram(self, wav):
        D = self.stft(wav)
        S = self.amp_to_db(np.abs(D)) - self.hp.audio.ref_level_db
        return self.normalize(S)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def normalize(self, S):
        return np.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, normalized_S):
        return (np.clip(normalized_S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db

