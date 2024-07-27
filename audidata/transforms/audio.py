import numpy as np
import random
import librosa


class ToMono:
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        data["audio"] = np.mean(data["audio"], axis=0, keepdims=True)
        return data