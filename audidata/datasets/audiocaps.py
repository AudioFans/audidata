import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from torch.utils.data import Dataset

from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import ToMono
from audidata.transforms.text import TextNormalization


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

    url = "https://github.com/cdjkim/audiocaps"

    duration = 501667.11  # Dataset duration (s), 139 hours, including training, 
    # validation, and testing

    def __init__(
        self, 
        root: str = None, 
        split: Union["train", "val" "test"] = "train",
        sr: float = 16000,  # Sampling rate
        crop: Optional[callable] = StartCrop(clip_duration=10.),
        transform: Optional[callable] = ToMono(),
        target_transform: Optional[callable] = TextNormalization()
    ) -> None:
    
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

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load target
        target_data = self.load_target(caption=caption)
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

    def load_target(self, caption: str) -> dict:

        data = {"caption": caption}

        if self.target_transform:
            data = self.target_transform(data)

        return data


if __name__ == "__main__":
    r"""Example.
    """
    
    from torch.utils.data import DataLoader
    
    root = "/datasets/audiocaps"

    sr = 16000
    clip_duration = 10.

    dataset = AudioCaps(
        root=root,
        split="train",
        sr=sr,
        crop=StartCrop(clip_duration=clip_duration)
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        
        n = 0
        audio_path = data["audio_path"][n]
        start_time = data["start_time"][n].cpu().numpy()
        audio = data["audio"][n].cpu().numpy()
        caption = data["caption"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("start_time:", start_time)
    print("audio:", audio.shape)
    print("caption:", caption)