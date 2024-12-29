from __future__ import annotations
import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from torch.utils.data import Dataset

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono


class Shutterstock(Dataset):
    r"""Shutterstock is a private audio-text pairs music dataset consisting 
    28,558 audio clips. Each audio sample has 1 captions. Audio samples are of 
    5 to 1415 s. Most audio samples are of 1 ~ 3 minutes. Audios are sampled at 
    48,000 Hz. After decompression, the dataset size is 409 GB.

    After decompression, the dataset looks like:

        dataset_root (409 GB)
        ├── flac (28,558 files)
        └── 28kdescriptions.csv

    """

    duration = 4502350.60  # Dataset duration (s), 1250 hours

    def __init__(
        self, 
        root: str = None, 
        sr: float = 16000,  # Sampling rate
        crop: Optional[callable] = RandomCrop(clip_duration=10.),
        transform: Optional[callable] = Mono(),
        target_transform: Optional[callable] = None
    ) -> None:
    
        self.root = root
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        self.meta_csv = Path(self.root, "28kdescriptions.csv")
        self.audios_dir = Path(self.root, "flac")

        self.meta_dict = self.load_meta(self.meta_csv, self.audios_dir)

        if not Path(root).exists():
            raise "Please download the Shutterstock dataset from {}".format(Shutterstock.url)

    def __getitem__(self, index: int) -> dict:

        caption = self.meta_dict["caption"][index]
        audio_name = self.meta_dict["audio_name"][index]
        audio_path = Path(self.audios_dir, audio_name)

        full_data = {
            "dataset_name": "Shutterstock",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load target
        target_data = self.load_target(caption=caption)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str, audios_dir: str) -> dict:

        audio_names_dict = {str(Path(name).stem): True for name in os.listdir(audios_dir)}

        df = pd.read_csv(meta_csv, sep=',')
        meta_dict = {"audio_name": [], "title": [], "caption": []}
        
        for n in range(len(df)):
            
            audio_name = str(df["idds"].values[n])

            if audio_name in audio_names_dict.keys():
                meta_dict["audio_name"].append("{}.flac".format(audio_name))
                meta_dict["title"].append(df["titles"].values[n])
                meta_dict["caption"].append(df["descri"].values[n])

        return meta_dict

    def load_audio(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)

        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.
            duration = None

        audio = load(
            path=path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )
        # shape: (channels, audio_samples)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration if clip_duration else audio_duration
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_target(self, caption: str) -> dict:

        data = {"caption": caption}

        if self.target_transform:
            data = self.target_transform(data)

        return data