from __future__ import annotations
import os
import re
import pandas as pd
from pathlib import Path
from typing import NoReturn
from typing_extensions import Literal

import librosa
import numpy as np
from torch.utils.data import Dataset

from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call


class Clotho(Dataset):
    r"""Clotho [1] is an audio caption dataset consisting 4981 audio samples. 
    Each audio sample has 5 captions. Audio samples are of 15 to 30s. Captions 
    are 8 to 20 words long. Audios are sampled at 44,100 Hz. After decompression, 
    the dataset size is 7.3 GB.

    [1] K. Drossos, et al., Clotho: An Audio Captioning Dataset, ICASSP, 2020

    The dataset looks like:

        dataset_root (7.3 GB)
        ├── clotho_audio_development (2894 wavs)
        ├── clotho_audio_evaluation (1046 wavs)
        ├── clotho_captions_development.csv
        ├── clotho_captions_evaluation.csv
        ├── clotho_metadata_development.csv
        ├── clotho_metadata_evaluation.csv
        └── LICENSE
    """

    URL = "https://zenodo.org/records/3490684"

    DURATION = 88366.30  # Dataset duration (s), 24.5 hours, including development and evaluation.

    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "test"] = "train",
        sr: float = 44100,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=10.),
        transform: None | callable = Mono(),
        target_transform: None | callable = None
    ) -> None:
    
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        if split == "train":
            self.meta_csv = Path(self.root, "clotho_captions_development.csv")
            self.audios_dir = Path(self.root, "clotho_audio_development")

        elif split == "test":
            self.meta_csv = Path(self.root, "clotho_captions_evaluation.csv")
            self.audios_dir = Path(self.root, "clotho_audio_evaluation")

        else:
            raise ValueError(split)

        self.meta_dict = self.load_meta(self.meta_csv)

        if not Path(root).exists():
            raise "Please download the Clotho dataset from {}".format(Clotho.url)

    def __getitem__(self, index: int) -> dict:

        caption = self.meta_dict["caption"][index]
        audio_name = self.meta_dict["audio_name"][index]
        audio_path = Path(self.audios_dir, audio_name)

        full_data = {
            "dataset_name": "Clotho",
            "audio_path": str(audio_path),
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

    def load_meta(self, meta_csv: str) -> dict:

        meta_dict = {"audio_name": [], "caption": []}

        df = pd.read_csv(meta_csv, sep=',')

        for n in range(len(df)):
            for i in range(1, 6):
                meta_dict["audio_name"].append(df["file_name"][n])
                meta_dict["caption"].append(df["caption_{}".format(i)][n])

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
        )
        # shape: (channels, audio_samples)

        # Transform audio
        if self.transform is not None:
            audio = call(transform=self.transform, x=audio)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration
        }

        return data

    def load_target_data(self, caption: str) -> dict:

        target = caption

        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)

        data = {
            "caption": caption,
            "target": target,
        }

        return data