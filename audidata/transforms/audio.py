import numpy as np
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