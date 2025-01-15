from __future__ import annotations
from pathlib import Path
from typing import Optional
import difflib
import os

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll
from audidata.io.midi import read_single_track_midi
from audidata.collate.base import collate_list_fn


default_collate_fn_map.update({list: collate_list_fn})


class Isophonics(Dataset):
    r"""Isophonics [1] is a dataset contains a collection of albums maintained 
    by QMUL. The version until Dec. 2024 contains 283 songs from 18 CDs, with a 
    total duration of 15 hours. Audios are sampled at 44,100 Hz. After 
    decompression, the dataset is 687 MB. Some songs contains annotations of 
    beat, chord, structure annotated by QMUL researchers.

    Audios are from commercial CDs and are not publicly available. Users need
    to purchase the CDs (or ask for those who has the audio files). Annotations 
    can be downloaded from [1]. Annotations may contain errors. Please carefully
    read http://isophonics.net/content/reference-annotations for details.

    Confidence:

    | Dataset         | CDs | Songs | Beat confidence | Chord con.  | Key con. | Structure con. |
    |-----------------|-----|-------|-----------------|-------------|----------|----------------|
    | The Beatles     | 12  | 180   | Moderate        | High        | Low      | Good           |
    | Carole King     |  1  |  14   | -               | Not checked | Moderate | Good           |
    | Michael Jackson |  3  |  51   | -               | -           | -        | Good           |
    | Queen           |  2  |  38   | -               | Moderate    | Moderate | Good           |

    [1] http://isophonics.net/

    Users should prepare the dataset looks like:

        dataset_root (687 MB)
        ├── audio (283 mp3s)
        │   ├── 01_-_Please_Please_Me_01_-_I_Saw_Her_Standing_There.mp3
        │   └── ...
        └── annotations
            ├── beat
            │   └── The Beatles (12 CDs)
            ├── chordlab
            │   ├── Carole King (1 CD)
            │   ├── Queen (2 CDs)
            │   └── The Beatles (12 CDs)
            ├── keylab
            │   ├── Carole King (1 CD)
            │   ├── Queen (2 CDs)
            │   └── The Beatles (12 CDs)
            └── seglab
                ├── Carole King (1 CD)
                ├── Michael Jackson (2 CDs)
                ├── Queen (2 CDs)
                └── The Beatles (12 CDs)
    """

    url = "http://isophonics.net/"

    duration = 54184.58  # Dataset duration (s), 15 hours.

    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        crop: Optional[callable] = RandomCrop(clip_duration=10., end_pad=9.9),
        transform: Optional[callable] = Mono(),
        target_types: bool = True,
        target_transform: Optional[callable] = PianoRoll(fps=100, pitches_num=128),
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target_types = target_types
        self.transform = transform
        self.target_transform = target_transform

        labels_dir = Path(self.root, "annotations", "chordlab")
        label_paths = sorted(list(Path(labels_dir).rglob('*.lab')))

        audios_dir = Path(self.root, "audios")
        audio_names = sorted(os.listdir(audios_dir))
        
        audio_paths = []

        from IPython import embed; embed(using=False); os._exit(0)
        for label_path in label_paths:

            name = str(Path(label_path.stem))
            print(name)
            audio_name = difflib.get_close_matches(name, audio_names)[0]
            audio_path = Path(audios_dir, audio_name)
            audio_paths.append(audio_path)

        meta_dict = {
            "audio_path": audio_paths,
            "label_path": label_paths
        }

        from IPython import embed; embed(using=False); os._exit(0)
            

        # meta_dict = {}

    def __getitem__(self, index: int) -> dict:

        from IPython import embed; embed(using=False); os._exit(0)

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