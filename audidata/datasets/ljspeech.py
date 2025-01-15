from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class LJSpeech(Dataset):
    r"""LJSpeech [1] is a speech dataset containing 13,100 audio clips from a 
    single speaker. The total duration is 24 hours of speech. The audio clips 
    range in duration from 1.11 seconds to 10.10 seconds. All audio are mono 
    sampled at 22,050 Hz. The captions range from 1 to 59 words. After 
    decompression, the dataset size is 3.6 GB.

    [1] https://keithito.com/LJ-Speech-Dataset/

    The dataset looks like:

        LJSpeech-1.1 (3.6 GB)
        ├── wavs (13,100 .wavs)
        │   ├── LJ001-0001.wav
        │   └── ...
        ├── metadata.csv
        ├── README
        ├── train.txt
        ├── valid.txt
        └── test.txt

    """

    URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

    # Train, valid, test splits files are downloaded from:
    # https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/train.txt?download=true
    # https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/valid.txt?download=true
    # https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/test.txt?download=true

    DURATION = 86117.08  # Dataset duration (s), 24 hours

    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "valid" "test"] = "train",
        sr: float = 22050,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=10.),
        transform: None | callable = Mono(),
        target_transform: None | callable = None,
    ) -> None:
    
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {LJSpeech.URL}")

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        caption = self.meta_dict["caption"][index]

        full_data = {
            "dataset_name": "LJSpeech",
            "audio_path": audio_path,
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

    def load_meta(self) -> dict:
        r"""Load metadata of the GTZAN dataset.
        """

        # Load split file
        split_path = Path(self.root, "{}.txt".format(self.split))
        df = pd.read_csv(split_path, header=None)
        split_names = df[0].values

        # Load csv file
        csv_path = Path(self.root, "metadata.csv")
        df = pd.read_csv(csv_path, sep="|", header=None)
        audio_names = df[0].values
        captions = df[1].values

        # Get split indexes
        idxes = []
        for i in range(len(audio_names)):
            if audio_names[i] in split_names:
                idxes.append(i)

        audio_names = audio_names[idxes]
        captions = captions[idxes]
        audio_paths = [str(Path(self.root, "wavs", "{}.wav".format(name))) for name in audio_names]

        meta_dict = {
            "audio_name": audio_names,
            "audio_path": audio_paths,
            "caption": captions
        }

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
        )  # shape: (channels, audio_samples)

        # Transform audio
        if self.transform is not None:
            audio = call(transform=self.transform, x=audio)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration
        }

        return data

    def load_target_data(self, caption: str) -> np.ndarray:

        target = caption

        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)
            # target: (classes_num,)

        data = {
            "caption": caption,
            "target": target
        }

        return data