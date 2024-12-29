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


class AudioCaps(Dataset):
    r"""AudioCaps [1] is an audio caption dataset containing 51,308 audio clips.
    Each audio clip contains one caption. Audio samples are of ~10 seconds. 
    Audios are sampled at 32,000 kHz. After decompression, the dataset size is 131 GB. 

    [1] C. D. Kim, et al. AudioCaps: Generating Captions for Audios in The Wild, NAACL-HLT 2019

    The dataset looks like:

        dataset_root (131 GB)
        ├── train (49274 files)
        ├── val (494 files)
        ├── test (957 files)
        ├── train.csv
        ├── val.csv
        ├── test.csv
        ├── LICENSE.txt
        └── README.md
    """

    URL = "https://github.com/cdjkim/audiocaps"
    
    DURATION = 501667.11  # Dataset duration (s), 139 hours, including training, 
    # validation, and testing

    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "val" "test"] = "train",
        sr: float = 32000,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=10.),
        transform: None | callable = Mono(),
        target_transform: None | callable = None
    ) -> NoReturn:
    
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        self.meta_csv = Path(self.root, "{}.csv".format(self.split))
        self.audios_dir = Path(self.root, self.split)

        self.meta_dict = self.load_meta(self.meta_csv)

        if not Path(root).exists():
            raise "Please download the AudioCaps dataset from {}".format(AudioCaps.url)

    def __getitem__(self, index: int) -> dict:

        caption = self.meta_dict["caption"][index]
        audio_name = self.meta_dict["audio_name"][index]
        audio_path = Path(self.audios_dir, audio_name)

        full_data = {
            "dataset_name": "AudioCaps",
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

        meta_dict = {"audiocap_id": [], "audio_name": [], "caption": []}

        df = pd.read_csv(meta_csv, sep=',')

        for n in range(len(df)):
            meta_dict["audiocap_id"].append(df["audiocap_id"][n])
            meta_dict["audio_name"].append("Y{}.wav".format(df["youtube_id"][n]))
            meta_dict["caption"].append(df["caption"][n])

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