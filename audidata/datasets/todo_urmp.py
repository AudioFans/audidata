from __future__ import annotations
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono
from audidata.transforms.midi import MultiTrackPianoRoll
from audidata.io.midi import read_multi_track_midi
from audidata.collate.base import collate_list_fn


default_collate_fn_map.update({list: collate_list_fn})


class URMP(Dataset):
    r"""URMP [1] is a dataset comprises 44 simple multi-instrument classical music pieces assembled from coordinated but separately recorded performances of individual tracks. For each piece, the musical score is provided in MIDI format, the audio recordings of the individual tracks, the audio and video recording of the assembled mixture, and ground-truth annotation files including frame-level and note-level transcriptions are provided.

    [1] Bochen Li *, Xinzhao Liu *, Karthik Dinesh, Zhiyao Duan, Gaurav Sharma, "Creating a multi-track classical music performance dataset for multi-modal music analysis: Challenges, insights, and applications", IEEE Transactions on Multimedia, 2018. (* equal contribution)

    The dataset looks like:

        dataset_root (12 GB)
        ├── 01_Jupiter_vn_vc
        │   ├── AuMix_01_Jupiter_vn_vc.wav
        │   ├── AuSep_1_vn_01_Jupiter.wav
        │   ├── AuSep_2_vc_01_Jupiter.wav
        │   ├── F0s_1_vn_01_Jupiter.txt
        │   ├── F0s_2_vc_01_Jupiter.txt
        │   ├── Notes_1_vn_01_Jupiter.txt
        │   ├── Notes_2_vc_01_Jupiter.txt
        │   ├── Sco_01_Jupiter_vn_vc.mid
        ├── 02_Sonata_vn_vn
        ...
        ├── 43_Chorale_tpt_tpt_hn_tbn_tba
        ├── Supplementary_Files
    """

    url = "https://labsites.rochester.edu/air/projects/URMP.html"

    duration = 21355.7  # Dataset duration (s) - 5 hours 55 minutes 55.7 seconds

    def __init__(
        self,
        root: str,
        split: str = "train",
        sr: int = 44100,
        crop: Optional[callable] = RandomCrop(clip_duration=10.),
        transform: Optional[callable] = Mono(),
        target: bool = True,
        target_transform: Optional[callable] = MultiTrackPianoRoll(fps=100, pitches_num=128),
    ):
        self.root = Path(root)
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target = target
        self.target_transform = target_transform

        self.pieces = sorted([d for d in os.listdir(self.root) if d.startswith(("0", "1", "2", "3", "4"))]) # 01-44

    def __getitem__(self, index: int) -> dict:
        piece_dir = self.root / self.pieces[index]
        
        mix_file = next(piece_dir.glob("AuMix_*.wav"))
        stem_files = sorted(piece_dir.glob("AuSep_*.wav"))
        
        midi_file = next(piece_dir.glob("Sco_*.mid"))

        mix_audio = self.load_audio(mix_file)
        
        data = {
            "dataset_name": "URMP",
            "piece_name": self.pieces[index],
            "mix": mix_audio["audio"],
            "start_time": mix_audio["start_time"],
            "duration": mix_audio["duration"],
        }

        data["stems"] = []
        data["instruments"] = []
        for stem_file in stem_files:
            stem_audio = self.load_audio(stem_file, start_time=mix_audio["start_time"], duration=mix_audio["duration"])
            instrument = stem_file.stem.split("_")[2]
            data["stems"].append(stem_audio["audio"])
            data["instruments"].append(instrument)

        if self.target:
            midi_data = self.load_midi(midi_file, start_time=mix_audio["start_time"], clip_duration=mix_audio["duration"])
            data.update(midi_data)

        return data

    def __len__(self):
        return len(self.pieces)

    def load_audio(self, path: str, start_time: float = None, duration: float = None) -> dict:
        audio_duration = librosa.get_duration(path=path)
        
        if self.crop and start_time is None:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            clip_duration = duration

        audio = load(
            path=path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration if clip_duration else audio_duration
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_midi(self, midi_path: str, start_time: float, clip_duration: float) -> dict:
        midi_data = read_multi_track_midi(midi_path=midi_path)

        data = {
            "tracks": [],
            "start_time": start_time,
            "clip_duration": clip_duration
        }

        for track in midi_data:
            track_data = {
                "program": track["program"],
                "is_drum": track["is_drum"],
                "notes": track["notes"],
                "start_time": start_time,
                "clip_duration": clip_duration
            }
            data["tracks"].append(track_data)

        return data