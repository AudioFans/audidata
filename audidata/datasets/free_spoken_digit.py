from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class FreeSpokenDigit(Dataset):
    r"""FreeSpokenDigit [1] is a speech dataset containing 3,000 recordings of 
    10 digits spoken by 6 speakers. The total duration of the dataset is 22 
    minutes. The durations of the audio clips range from 0.14 seconds to 2.28 
    seconds. All audio are mono sampled at 8 kHz. After decompression, the 
    dataset size is 58 MB.

    [1] https://github.com/Jakobovski/free-spoken-digit-dataset

    The dataset looks like:

        free-spoken-digit-dataset (22 MB)
        └── recordings (3,000 wavs)
            ├── 0_george_0.wav
            ├── 1_george_0.wav
            └── ...
    """

    URL = "https://github.com/Jakobovski/free-spoken-digit-dataset"

    DURATION = 1312.30  # Dataset duration (s), 22 minutes

    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "test"] = "train",
        sr: float = 8000,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=3.),
        transform: None | callable = Mono(),
    ) -> None:
    
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {FreeSpokenDigit.URL}")

        self.speakers = self.get_speakers(split)

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        label = self.meta_dict["label"][index]
        speaker = self.meta_dict["speaker"][index]

        full_data = {
            "dataset_name": "FreeSpokenDigit",
            "audio_path": audio_path,
        }

        # Load audio data
        audio_data = self.load_audio_data(path=audio_path)
        full_data.update(audio_data)

        # Load target data
        target_data = self.load_target_data(label=label, speaker=speaker)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def get_speakers(self, split: str) -> list[str]:

        if split == "train":
            speakers = ["jackson", "nicolas", "theo", "yweweler", "george"]

        elif split == "test":
            speakers = ["lucas"]

        else:
            raise ValueError(split)

        return speakers

    def load_meta(self) -> dict:
        r"""Load metadata of the dataset.
        """

        meta_dict = {
            "audio_name": [],
            "audio_path": [],
            "label": [],
            "speaker": []
        }
        
        audios_dir = Path(self.root, "recordings")

        audio_paths = sorted(list(Path(audios_dir).glob("*.wav")))

        for audio_path in audio_paths:

            audio_name = Path(audio_path).name
            label, speaker = audio_name.split("_")[0 : 2]

            if speaker in self.speakers:

                meta_dict["audio_name"].append(audio_name)
                meta_dict["audio_path"].append(str(audio_path))
                meta_dict["label"].append(label)
                meta_dict["speaker"].append(speaker)

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

    def load_target_data(self, label: str, speaker: str) -> np.ndarray:

        caption = "speaker: {}, label: {}".format(speaker, label)

        data = {
            "label": label,
            "speaker": speaker,
            "caption": caption
        }

        return data