import librosa
import numpy as np

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

    def rescale(self, wav):
        return wav / np.abs(wav).max() * self.hp.audio.rescaling_max

    def stft(self, wav):
        return librosa.stft(y=wav,
                            n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    def linear_to_mel(self, spectrum):
        return np.dot(self.mel_basis, spectrum)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, np.abs(x)))


    def melspectrogram(self, y):
        D = self.stft(y)
        mel = self.linear_to_mel(np.abs(D))
        S = self.amp_to_db(mel) - self.hp.audio.ref_level_db

        return self._normalize(S)

    def mfcc(self, y):
        return librosa.feature.mfcc(y=y,
                                    sr=self.hp.audio.sampling_rate,
                                    n_mfcc=self.hp.audio.n_mfcc,
                                    n_fft=self.hp.audio.n_fft,
                                    hop_length=self.hp.audio.hop_length,
                                    win_length=self.hp.audio.win_length
                                    )

    def _normalize(self, S):
        return np.clip((S - self.hp.audio.min_level_db) / -self.hp.audio.min_level_db, 0, 1)
