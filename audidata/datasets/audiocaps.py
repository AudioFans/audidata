from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import librosa
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class AudioCaps(Dataset):
    r"""AudioCaps [1] is an audio caption dataset containing 51,308 audio clips, 
    each accompanied by one caption. The audio samples are approximately 10 
    seconds long and are mono sampled at 32,000 Hz. After decompression, the 
    dataset size is 131 GB.

    [1] C. D. Kim, et al. AudioCaps: Generating Captions for Audios in The Wild, NAACL-HLT 2019

    The dataset looks like:

        audiocaps (131 GB)
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

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {AudioCaps.URL}")

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        caption = self.meta_dict["caption"][index]
        
        full_data = {
            "dataset_name": "AudioCaps",
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

    def load_meta(self, meta_csv: str) -> dict:

        df = pd.read_csv(meta_csv, sep=',')

        meta_dict = {
            "audio_name": [], 
            "audio_path": [],
            "audiocap_id": [], 
            "caption": []
        }

        for n in range(len(df)):
            
            audio_name = "Y{}.wav".format(df["youtube_id"][n])
            audio_path = str(Path(self.audios_dir, audio_name))

            meta_dict["audio_name"].append(audio_name)
            meta_dict["audio_path"].append(audio_path)
            meta_dict["audiocap_id"].append(df["audiocap_id"][n])
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
        )  # shape: (channels_num, audio_samples)

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