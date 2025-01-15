from __future__ import annotations

from pathlib import Path

import librosa
from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono, Normalize
from audidata.utils import call
from torch.utils.data import Dataset


class VoicebankDemand(Dataset):
    r"""Voicebank-Demand [1] contains 24,793 paired noisy and clean speech for 
    training speech enhancement algorithms. The total duration is 20 hours. 
    Audio files are mono are sampled at 16,000 Hz. After decompression, the 
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

    URL = "https://datashare.ed.ac.uk/handle/10283/2791"

    DURATION = 71761.32  # Dataset duration (s), 20 hours, including training 
    # and testing.

    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        crop: None | callable = RandomCrop(clip_duration=2., end_pad=0.),
        stem_transform: None | callable = [Mono(), Normalize()],
    ) -> None:

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.stem_transform = stem_transform

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {VoicebankDemand.URL}")

        self.meta_dict = self.load_metadata()
        
    def __getitem__(self, index: int) -> dict:

        # Audio paths
        mixture_audio_path = self.meta_dict["mixture_audio_path"][index]
        clean_audio_path = self.meta_dict["clean_audio_path"][index]

        full_data = {
            "dataset_name": "VoicebankDemand",
            "clean_audio_path": clean_audio_path,
            "mixture_audio_path": mixture_audio_path
        }

        # Load audio data
        audio_data = self.load_audio_data(mixture_audio_path, clean_audio_path)
        
        if self.stem_transform is not None:
            audio_data["target"] = call(self.stem_transform, audio_data["target"])
            audio_data["background"] = call(self.stem_transform, audio_data["background"])
            
        audio_data["mixture"] = audio_data["target"] + audio_data["background"]
        full_data.update(audio_data)
        
        return full_data

    def __len__(self):

        audios_num = len(self.meta_dict["clean_audio_path"])

        return audios_num

    def load_metadata(self) -> dict:
        r"""Load meta dict.
        """

        clean_audios_dir = Path(self.root, "clean_{}set_wav".format(self.split))
        clean_audio_paths = [str(path) for path in sorted(list(clean_audios_dir.glob("*.wav")))]
        
        mixture_audios_dir = Path(self.root, "noisy_{}set_wav".format(self.split))
        mixture_audio_paths = [str(Path(mixture_audios_dir, Path(path).name)) for path in clean_audio_paths]

        meta_dict = {
            "clean_audio_path": clean_audio_paths,
            "mixture_audio_path": mixture_audio_paths
        }

        return meta_dict

    def load_audio_data(self, mixture_audio_path: str, clean_audio_path: str) -> dict:

        audio_duration = librosa.get_duration(path=mixture_audio_path)

        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.
            clip_duration = audio_duration

        # Load mixture
        mixture = load(
            path=mixture_audio_path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )  # shape: (channels_num, audio_samples)

        # Load target
        target = load(
            path=clean_audio_path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )  # shape: (channels_num, audio_samples)

        background = mixture - target

        data = {
            "target": target, 
            "background": background,
            "start_time": start_time,
            "duration": clip_duration
        }

        return data