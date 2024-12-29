from __future__ import annotations
from pathlib import Path
from typing import Optional, NoReturn

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.io.midi import read_single_track_midi
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll
from audidata.utils import call
from audidata.collate.base import collate_list_fn


default_collate_fn_map.update({list: collate_list_fn})


class VoicebankDemand(Dataset):
    r"""Voicebank-Demand [1] contains paired noisy and clean speech for training
    speech enhancement algorithms. Voicebank-Demand contains 20 hours of 24,793
    audio files. Audios are sampled at 16,000 Hz. After decompression, the 
    dataset is 2.5 GB.

    [1] Thiemann, J., et al. The diverse environments multi-channel acoustic 
    noise database (demand): A database of multichannel environmental noise 
    recordings. In Proceedings of Meetings on Acoustics, 2013

    The dataset looks like:

        voicebank-demand (2.5 GB)
        ├── clean_trainset_wav
        │    └── ... (11572 wavs)
        ├── noisy_trainset_wav
        │    └── ... (11572 wavs)
        ├── clean_testset_wav
        │    └── ... (824 wavs)
        └── noisy_testset_wav
             └── ... (824 wavs)
    """

    url = "https://datashare.ed.ac.uk/handle/10283/2791"

    duration = 71761.32  # Dataset duration (s), 20 hours, including training 
    # and testing.

    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        crop: Optional[callable] = RandomCrop(clip_duration=2., end_pad=0.),
        stem_transform: Optional[callable] = Mono(),
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.stem_transform = stem_transform

        self.meta_dict = self.load_metadata()
        
    def __getitem__(self, index: int) -> dict:

        # Audio paths
        clean_audio_path = str(self.meta_dict["clean_audio_path"][index])
        mixture_audio_path = str(self.meta_dict["mixture_audio_path"][index])

        # Start time of a clip
        audio_duration = librosa.get_duration(path=clean_audio_path)
        start_time, clip_duration = self.crop(audio_duration=audio_duration)

        data = {
            "dataset_name": "VoicebankDemand",
            "clean_audio_path": clean_audio_path,
            "mixture_audio_path": mixture_audio_path
        }

        # Load target
        data["target"] = load(
            path=clean_audio_path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )
        # shape: (channels, audio_samples)

        # Load mixture
        mixture = load(
            path=mixture_audio_path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )
        # shape: (channels, audio_samples)

        data["background"] = mixture - data["target"]

        if self.stem_transform is not None:
            data["target"] = call(self.stem_transform, data["target"])
            data["background"] = call(self.stem_transform, data["background"])
            
        data["mixture"] = data["target"] + data["background"]
        
        return data

    def __len__(self):

        audios_num = len(self.meta_dict["clean_audio_path"])

        return audios_num

    def load_metadata(self) -> dict:
        r"""Load meta dict.
        """

        clean_audios_dir = Path(self.root, "clean_{}set_wav".format(self.split))
        clean_audio_paths = sorted(list(clean_audios_dir.glob("*.wav")))
        
        mixture_audios_dir = Path(self.root, "noisy_{}set_wav".format(self.split))
        mixture_audio_paths = [Path(mixture_audios_dir, path.name) for path in clean_audio_paths]

        meta_dict = {
            "clean_audio_path": clean_audio_paths,
            "mixture_audio_path": mixture_audio_paths
        }

        return meta_dict