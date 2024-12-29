import numpy as np
import random
import librosa
import soundfile as sf


class ToMono:
    r"""Converts multi-channel audio to mono by averaging all channels.

    This transformation takes a multi-channel audio input and returns a mono audio
    by computing the mean across all channels.

    Args:
        None

    Returns:
        dict: A dictionary containing the mono audio under the key 'audio'.
    """

    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        data["audio"] = np.mean(data["audio"], axis=0, keepdims=True)
        return data


class Normalize:
    r"""Normalizes the audio to have a maximum absolute value of 1.

    This transformation scales the audio so that its maximum absolute value is 1.

    Args:
        None

    Returns:
        dict: A dictionary containing the normalized audio under the key 'audio'.
    """

    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        audio = data["audio"]
        audio = audio / np.max(np.abs(audio))
        data["audio"] = audio
        return data


class PitchShift:
    r"""Applies pitch shifting to the audio.

    This transformation shifts the pitch of the audio by a specified number of semitones.

    Args:
        n_steps (int): The number of semitones to shift the pitch.
        sr (int): The sampling rate of the audio.

    Returns:
        dict: A dictionary containing the pitch-shifted audio under the key 'audio'.
    """

    def __init__(self, n_steps: int, sr: int):
        self.n_steps = n_steps
        self.sr = sr

    def __call__(self, data: dict) -> dict:
        audio = data["audio"]
        shifted = np.apply_along_axis(lambda x: librosa.effects.pitch_shift(x, sr=self.sr, n_steps=self.n_steps), axis=-1, arr=audio)
        data["audio"] = shifted
        return data


class TimeStretch:
    r"""Applies time stretching to the audio.

    This transformation stretches or compresses the audio in time without changing its pitch.

    Args:
        rate (float): The rate at which to stretch the audio. Values > 1 speed up the audio,
                      while values < 1 slow it down.

    Returns:
        dict: A dictionary containing the time-stretched audio under the key 'audio'.
    """

    def __init__(self, rate: float):
        self.rate = rate

    def __call__(self, data: dict) -> dict:
        audio = data["audio"]
        stretched = np.apply_along_axis(lambda x: librosa.effects.time_stretch(x, rate=self.rate), axis=-1, arr=audio)
        data["audio"] = stretched
        return data


class AddGaussianNoise:
    r"""Adds Gaussian noise to the audio.

    This transformation adds random Gaussian noise to the audio signal.

    Args:
        noise_level (float): The standard deviation of the Gaussian noise to be added.

    Returns:
        dict: A dictionary containing the noisy audio under the key 'audio'.
    """

    def __init__(self, noise_level=0.005):
        self.noise_level = noise_level

    def __call__(self, data: dict) -> dict:
        audio = data["audio"]
        noise = np.random.normal(0, self.noise_level, audio.shape)
        noisy_audio = audio + noise
        data["audio"] = noisy_audio
        return data


class RandomGain:
    r"""Applies random gain to the audio.

    This transformation multiplies the audio by a random gain factor.

    Args:
        min_gain (float): The minimum gain factor.
        max_gain (float): The maximum gain factor.

    Returns:
        dict: A dictionary containing the gain-adjusted audio under the key 'audio'.
    """

    def __init__(self, min_gain=0.5, max_gain=1.5):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, data: dict) -> dict:
        audio = data["audio"]
        gain = random.uniform(self.min_gain, self.max_gain)
        data["audio"] = audio * gain
        return data


class RandomImpulseReverb:
    r"""Applies random impulse reverb to the audio.

    This transformation creates a random impulse response and applies it to the audio
    to simulate reverb. The generated impulse response is normalized to prevent loudness changes.

    Args:
        rt60 (float): The RT60 (reverberation time) in seconds.
        room_scale (float): The scale of the simulated room.
        sr (int): The sampling rate of the audio.
        mix_ratio (float): The maximum ratio of reverb to original signal in the output.

    Returns:
        dict: A dictionary containing the reverberated audio under the key 'audio'.
    """

    def __init__(self, rt60=0.8, sr=22050, mix_ratio=0.5):
        self.rt60 = rt60
        self.sr = sr
        self.mix_ratio = mix_ratio

    def __call__(self, data: dict) -> dict:
        audio = data["audio"]
        reverb_audio = np.apply_along_axis(self.apply_random_impulse_reverb, axis=-1, arr=audio) # apply to each channel
        data["audio"] = reverb_audio
        return data

    def apply_random_impulse_reverb(self, audio):
        reverb_len = int(self.rt60 * self.sr)
        decay_constant = -np.log(0.001) / (self.rt60 * self.sr)
        time_points = np.arange(reverb_len) / self.sr
        decay = np.exp(-decay_constant * time_points) # exponential decay

        noise = np.random.randn(reverb_len)
        impulse_response = noise * decay
        impulse_response = impulse_response / np.linalg.norm(impulse_response) # normalize by power to prevent loudness changes

        reverb_audio = np.convolve(audio, impulse_response, mode='same')
        
        mixed_audio = (1 - self.mix_ratio) * audio + self.mix_ratio * reverb_audio
        return mixed_audio


def test_transformations(audio_path):
    r"""
    Testing function.
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]  # Add channel dimension if mono, shape: (channels, audio_samples)
    original_data = {"audio": audio, "sr": sr}
    
    augmentations = [
        ToMono(),
        Normalize(),
        PitchShift(n_steps=2, sr=sr),
        TimeStretch(rate=0.5),
        AddGaussianNoise(noise_level=0.005),
        RandomGain(min_gain=0.5, max_gain=1.5),
        RandomImpulseReverb(rt60=0.1, sr=sr, mix_ratio=0.9)
    ]

    for i, aug in enumerate(augmentations):
        aug_data = aug(original_data.copy())
        aug_audio = aug_data["audio"]
        output_filename = f"augmented_{aug.__class__.__name__}.wav"

        sf.write(output_filename, aug_audio.T, sr)


if __name__ == "__main__":
    test_transformations("test.m4a")  # Provide path to an audio file for testing all transformations