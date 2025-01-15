from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono
from audidata.utils import call
from torch.utils.data import Dataset


class Shutterstock(Dataset):
    r"""Shutterstock is a private audio-text pairs music dataset consisting of 
    28,558 audio clips. The total duration is 1,250 hours. Each audio file has 
    1 caption, with captions containing 5 to 37 words. The durations of the 
    audio files range from 5 seconds to 1,415 seconds, with most audio samples 
    lasting between 1 and 3 minutes. The audio files are stereo and sampled at 
    48,000 Hz. After decompression, the dataset size is 409 GB.

    After decompression, the dataset looks like:

        dataset_root (409 GB)
        ├── flac (28,558 files)
        └── 28kdescriptions.csv

    """

    URL = "https://www.shutterstock.com/"

    DURATION = 4502350.60  # Dataset duration (s), 1,250 hours

    def __init__(
        self, 
        root: str = None, 
        sr: float = 48000,  # Sampling rate
        crop: None | callable = RandomCrop(clip_duration=10.),
        transform: None | callable = Mono(),
        target_transform: None | callable = None
    ) -> None:
    
        self.root = root
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        self.meta_csv = Path(self.root, "28kdescriptions.csv")
        self.audios_dir = Path(self.root, "flac")

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {Shutterstock.URL}")

        self.meta_dict = self.load_meta(self.meta_csv, self.audios_dir)

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        caption = self.meta_dict["caption"][index]
        
        full_data = {
            "dataset_name": "Shutterstock",
            "audio_path": audio_path,
        }

        # Load audio data
        audio_data = self.load_audio_data(path=audio_path)
        full_data.update(audio_data)

        # Load target data
        target_data = self.load_target_data(caption=caption)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str, audios_dir: str) -> dict:

        audio_names_dict = {str(Path(name).stem): True for name in os.listdir(audios_dir)}

        df = pd.read_csv(meta_csv, sep=',')
        meta_dict = {"audio_name": [], "audio_path": [], "title": [], "caption": []}
        
        for n in range(len(df)):
            
            audio_name = str(df["idds"].values[n])

            if audio_name not in audio_names_dict.keys():
                continue

            audio_name = "{}.flac".format(audio_name)
            audio_path = str(Path(self.audios_dir, audio_name))

            meta_dict["audio_name"].append(audio_name)
            meta_dict["audio_path"].append(audio_path)
            meta_dict["title"].append(df["titles"].values[n])
            meta_dict["caption"].append(df["descri"].values[n])

        return meta_dict

    def load_audio_data(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)

        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.
            clip_duration = audio_duration

        # Load a clip
        audio = load(
            path=path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )  # shape: (channels_num, audio_samples)

        # Transform audio
        if self.transform is not None:
            audio = call(transform=self.transform, x=audio)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration
        }

        return data

    def load_target_data(self, caption: str) -> np.ndarray:

        target = caption

        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)
            # target: (classes_num,)

        data = {
            "caption": caption,
            "target": target
        }

        return data