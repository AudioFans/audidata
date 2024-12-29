from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from torch.utils.data import Dataset

from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call


class WavCaps(Dataset):
    r"""WavCaps [1] is an audio caption dataset containing 402,958 audio samples. 
    Each audio has 1 caption. Most audio samples are less than 10 s, with the 
    longest audio of 12.39 hours. Audios are sampled at 32000 Hz. After 
    decompression, the dataset is 770 GB.

    [1] X. Mei, et al., WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio 
    Captioning Dataset for Audio-Language Multimodal Research, 2023

    After decompression, the dataset looks like:

        dataset_root (131 GB)
        ├── Zip_files
        │   ├── AudioSet_SL (108,137 flac)
        │   ├── BBC_Sound_Effects (31,201 flac)
        │   ├── FreeSound (262,300 flac)
        │   └── SoundBible (1,320 flac)
        ├── Zip_files
        │   ├── AudioSet_SL
        │   │   └── as_final.json
        │   ├── BBC_Sound_Effects
        │   │   └── bbc_final.json
        │   ├── FreeSound
        │   │   └── fsd_final.json
        │   ├── SoundBible
        │   │   └── json_files_SoundBible_sb_final.json
        │   └── blacklist
        │       ├── blacklist_exclude_all_ac.json
        │       ├── blacklist_exclude_test_ac.json
        │       └── blacklist_exclude_ub8k_esc50_vggsound.json
        └── README.md
    """

    URL = "https://zenodo.org/records/3490684"

    DURATION = 27161470.93  # Dataset duration (s), 7545 hours

    def __init__(
        self, 
        root: str = None, 
        sr: float = 32000,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=10.),
        transform: None | callable = None,
        target_transform: None | callable = None
    ) -> None:
    
        self.root = root
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        self.audios_dir = Path(self.root, "Zip_files")
        self.jsons_dir = Path(self.root, "json_files")
        paths = sorted(list(Path(self.jsons_dir).rglob("*.json")))

        json_paths = [path for path in paths if "blacklist" not in str(path)]
        blacklist_paths = [path for path in paths if "blacklist" in str(path)]

        self.meta_dict = self.load_meta(json_paths, blacklist_paths)

        if not Path(root).exists():
            raise "Please download the WavCaps dataset from {}".format(WavCaps.url)

    def __getitem__(self, index: int) -> dict:

        caption = self.meta_dict["caption"][index]
        audio_name = self.meta_dict["audio_name"][index]
        subdataset = self.meta_dict["subdataset"][index]
        audio_path = Path(self.audios_dir, subdataset, audio_name)

        full_data = {
            "dataset_name": "WavCaps",
            "subdataset": subdataset,
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

    def load_meta(self, json_paths: list[str], blacklist_paths: list[str]) -> dict:

        black_names = self.get_black_names(blacklist_paths)
        # black_names: dict

        meta_dict = {"subdataset": [], "audio_name": [], "caption": []}

        for json_path in json_paths:
            subdataset = Path(json_path).parent.name
            with open(json_path, "r") as f:
                json_obj = json.load(f)
                for item in json_obj["data"]:
                    name = Path(item["id"]).stem
                    if name not in black_names.keys():
                        meta_dict["subdataset"].append(subdataset)
                        meta_dict["audio_name"].append("{}.flac".format(name))
                        meta_dict["caption"].append(item["caption"])

        return meta_dict

    def get_black_names(self, blacklist_paths: list[str]) -> dict:

        black_names = {}

        for path in blacklist_paths:

            with open(path, "r") as f:

                json_obj = json.load(f)

                for subdataset in json_obj.keys():

                    subnames = json_obj[subdataset]
                    subnames = [Path(name).stem for name in subnames]

                    for name in subnames:

                        black_names[name] = True

        return black_names

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