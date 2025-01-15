from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from typing_extensions import Literal
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset

from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono


class ESC50(Dataset):
    r"""The ESC-50 [1] dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
    It comprises 2000 5s-clips of 50 different classes across natural, human and domestic sounds, again, drawn from Freesound.org.

    [1] Piczak, Karol J. "ESC: Dataset for environmental sound classification." Proceedings of the 23rd ACM international conference on Multimedia. 2015.
    """

    url = "https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download"

    duration = 10008  # Dataset duration (s), 21,050 hours, including

    def __init__(
            self,
            root: str = None,
            sr: int = 44100,
            fold: Optional[int] = None,
            split: Optional[Literal["train", "test"]] = None,
            crop: Optional[callable] = StartCrop(clip_duration=29.),
            transform: Optional[callable] = Mono(),
            target_transform: Optional[callable] = None
    ):

        self.root = root
        self.sr = sr
        assert (fold is None) == (split is None)
        self.fold = fold  # the current fold for the 5-fold cross-validation in the ESC-50 dataset.
        self.split = split
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        label_id = self.meta_dict["label_id"][index]
        label = self.meta_dict["label"][index]

        target = np.zeros(self.classes_num, dtype="float32")
        target[label_id] = 1

        full_data = {
            "dataset_name": "ESC50",
            "audio_path": str(audio_path),
            "label": label,
            "target": target
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self) -> dict:
        r"""Load metadata of the ESC-50 dataset."""
        meta_csv = os.path.join(self.root, 'meta', 'esc50.csv')
        metadata = pd.read_csv(meta_csv)

        self.classes_num = len(metadata['target'].unique())

        meta_dict = {
            "audio_path": [],
            "audio_name": [],
            "label_id": [],
            "label": []
        }

        for _, row in metadata.iterrows():
            fold = row['fold'] - 1
            if self.fold is not None and self.split == 'train' and self.fold == fold:
                continue
            if self.fold is not None and self.split == 'test' and self.fold != fold:
                continue
            audio_path = os.path.join(self.root, 'audio', row['filename'])
            audio_name = row['filename']
            label_id = row['target']
            label = row['category']

            meta_dict["audio_name"].append(audio_name)
            meta_dict["audio_path"].append(audio_path)
            meta_dict["label_id"].append(label_id)
            meta_dict["label"].append(label)

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