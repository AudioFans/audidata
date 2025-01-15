from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono


class MagnaTagATune(Dataset):
    r"""MagnaTagATune[1] dataset contains 25,863 music clips.
    Each clip is a 29-seconds-long excerpt belonging to one of the 5223 songs, 445 albums and 230 artists.
    The clips span a broad range of genres like Classical, New Age, Electronica, Rock, Pop, World, Jazz, Blues, Metal, Punk, and more.
    Each audio clip is supplied with a vector of binary annotations of 188 tags.
    There are 16 parts, and researchers comonnly use parts 1-12 for training, part 13 for validation and parts 14-16 for testing.

    [1] Law, E., West, K., Mandel, M. I., Bay, M., and Downie, J. S. (2009). Evaluation of algorithms
    using games: The case of music tagging. In ISMIR, pages 387–392. Citeseer.

    """

    url = "https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset"

    duration = 75708000  # Dataset duration (s), 21,050 hours, including 
    # training, valdiation, and testing

    top_50 = [
        "guitar", "classical", "slow", "techno", "strings", "drums", "electronic", "rock", "fast", "piano",
        "ambient", "beat", "violin", "vocal", "synth", "female", "indian", "opera", "male", "singing", "vocals",
        "no vocals", "harpsichord", "loud", "quiet", "flute", "woman", "male vocal", "no vocal", "pop", "soft",
        "sitar", "solo", "man", "classic", "choir", "voice", "new age", "dance", "male voice", "female vocal",
        "beats", "harp", "cello", "no voice", "weird", "country", "metal", "female voice", "choral"
    ]

    classes_num = len(top_50)
    lb_to_ix = {lb: ix for ix, lb in enumerate(top_50)}
    ix_to_lb = {ix: lb for ix, lb in enumerate(top_50)}

    def __init__(
            self,
            root: str = None,
            split: ["train", "test", "val"] = "train",
            sr: int = 16000,
            crop: Optional[callable] = StartCrop(clip_duration=29.),
            transform: Optional[callable] = Mono(),
            target_transform: Optional[callable] = None
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        meta_csv = Path(self.root, "annotations_final.csv")
        self.meta_dict = self.load_meta(meta_csv)

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        labels = self.meta_dict["labels"][index]

        full_data = {
            "dataset_name": "MagnaTagATune",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load target
        target_data = self.load_target(labels=labels)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv) -> dict:
        r"""Load metadata of the MagnaTagATune dataset."""
        top_50 = MagnaTagATune.top_50
        meta_dict = {
            "labels": [],
            "audio_name": [],
            "audio_path": []
        }

        df = pd.read_csv(meta_csv, sep="\t")
        df = df[df[top_50].sum(axis=1) > 0]
        # remove the columns that are not in top_50 + mp3_path and clip_id
        df = df[top_50 + ["mp3_path", "clip_id"]]

        # Load train, validation, and test IDs
        if self.split == "train":
            ids_df = pd.read_csv(os.path.join(self.root, "train_gt_mtt.tsv"), sep="\t", header=None)
        elif self.split == "val":
            ids_df = pd.read_csv(os.path.join(self.root, "val_gt_mtt.tsv"), sep="\t", header=None)
        elif self.split == "test":
            ids_df = pd.read_csv(os.path.join(self.root, "test_gt_mtt.tsv"), sep="\t", header=None)

        ids = ids_df[0].tolist()

        split_df = df[df["clip_id"].isin(ids)]

        label_names = df.columns.drop(["mp3_path", "clip_id"])

        for index, row in split_df.iterrows():
            # Get the mp3_path
            audio_name = row["mp3_path"]
            audio_path = os.path.join(self.root, audio_name)

            # Generate list of labels based on the row values
            labels = [label for label in label_names if row[label] == 1]

            # If the song has no tags, skip it
            if not labels:
                continue

            meta_dict["labels"].append(labels)
            meta_dict["audio_name"].append(audio_name)
            meta_dict["audio_path"].append(audio_path)

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

    def load_target(self, labels: list[str]) -> dict:
        classes_num = MagnaTagATune.classes_num
        lb_to_ix = MagnaTagATune.lb_to_ix

        target = np.zeros(classes_num, dtype="float32")

        target[[lb_to_ix[label] for label in labels]] = 1

        data = {
            "target": target,
            "labels": labels
        }

        return data