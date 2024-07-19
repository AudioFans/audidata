import numpy as np
import random
import librosa


class StartCrop:
    r"""Crop a clip from the start of an audio."""
    def __init__(self, clip_duration: float, sr: int):
        self.clip_samples = round(clip_duration * sr)

    def __call__(self, x: np.ndarray) -> np.ndarray:

        clip = librosa.util.fix_length(
            data=x[:, 0 : self.clip_samples], 
            size=self.clip_samples, 
            axis=-1
        )

        return clip


class RandomCrop:
    r"""Crop a clip from the start of an audio."""
    def __init__(self, clip_duration: float, sr: int):
        self.clip_samples = round(clip_duration * sr)

    def __call__(self, x: np.ndarray) -> np.ndarray:

        len_x = x.shape[-1]

        if len_x < self.clip_samples:
            start_sample = 0

        else:
            start_sample = random.randint(0, len_x - self.clip_samples)

        clip = librosa.util.fix_length(
            data=x[:, start_sample : start_sample + self.clip_samples], 
            size=self.clip_samples, 
            axis=-1
        )

        return clip