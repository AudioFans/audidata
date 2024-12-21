from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import ToMono
from audidata.transforms.midi import PianoRoll
from audidata.io.midi import read_single_track_midi
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
        transform: Optional[callable] = ToMono(),
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform

        from IPython import embed; embed(using=False); os._exit(0)
        
    def __getitem__(self, index: int) -> dict:

        audio_path = Path(self.root, self.meta_dict["audio_name"][index])
        midi_path = Path(self.root, self.meta_dict["midi_name"][index]) 
        duration = self.meta_dict["duration"][index]

        full_data = {
            "dataset_name": "MAESTRO-V3.0.0",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)
        
        # Load target
        if self.target_types:
            target_data = self.load_target(
                midi_path=midi_path, 
                start_time=audio_data["start_time"],
                clip_duration=audio_data["duration"]
            )
            full_data.update(target_data)
        
        return full_data

    def __len__(self):

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str) -> dict:
        r"""Load meta dict.
        """

        df = pd.read_csv(meta_csv, sep=',')

        indexes = df["split"].values == self.split

        meta_dict = {
            "midi_name": df["midi_filename"].values[indexes],
            "audio_name": df["audio_filename"].values[indexes],
            "duration": df["duration"].values[indexes]
        }

        return meta_dict

    def load_audio(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)
        
        if self.crop:
            # Load a clip
            start_time, duration = self.crop(audio_duration=audio_duration)
        else:
            # Load full song
            start_time = 0.
            duration = audio_duration

        audio = load(
            path=path, 
            sr=self.sr, 
            offset=start_time, 
            duration=duration
        )
        # shape: (channels, audio_samples)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": duration
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_target(
        self, 
        midi_path: str, 
        start_time: float, 
        clip_duration: float
    ) -> dict:
        
        notes, pedals = read_single_track_midi(
            midi_path=midi_path, 
            extend_pedal=self.extend_pedal
        )
        
        data = {
            "note": notes,
            "pedal": pedals,
            "start_time": start_time,
            "clip_duration": clip_duration
        }
        
        if self.target_transform:
            data = self.target_transform(data)

        return data