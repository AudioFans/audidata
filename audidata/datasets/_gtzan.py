import os
import re
from pathlib import Path
from typing import Optional, Union
import librosa
import numpy as np
from torch.utils.data import Dataset
import time

from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import ToMono


class GTZAN(Dataset):
    r"""GTZAN [1] is a music dataset containing 1000 30-second music. 
    GTZAN contains 10 genres. Audios are sampled at 22,050 Hz. Dataset size is 1.3 GB. 

    [1] Tzanetakis, G., et al., Musical genre classification of audio signals. 2002

    The dataset looks like:

        dataset_root (1.3 GB)
        └── genres
            ├── blues (100 files)
            ├── classical (100 files)
            ├── country (100 files)
            ├── disco (100 files)
            ├── hiphop (100 files)
            ├── jazz (100 files)
            ├── metal (100 files)
            ├── pop (100 files)
            ├── reggae (100 files)
            └── rock (100 files)
    """

    url = "http://marsyas.info/index.html"

    duration = 30024.07  # Dataset duration (s), including training, validation, and testing.

    labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", 
        "metal", "pop", "reggae", "rock"]

    classes_num = len(labels)
    lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
    ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}

    def __init__(
        self, 
        root: str = None, 
        split: Union["train", "test"] = "train",
        test_fold: int = 0,  # E.g., fold 0 is used for testing. Fold 1 - 9 are used for training.
        sr: float = 16000,  # Sampling rate
        crop: Optional[callable] = StartCrop(clip_duration=30.),
        transform: Optional[callable] = ToMono(),
        target_transform: Optional[callable] = None
    ) -> None:
    
        self.root = root
        self.split = split
        self.test_fold = test_fold
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        if not Path(root).exists():
            raise "Please download the GTZAN dataset from {} (Invalid anymore. Please search a source)".format(GTZAN.url)

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        label = self.meta_dict["label"][index]

        full_data = {
            "dataset_name": "MUSDB18HQ",
            "audio_path": str(audio_path),
        }

        t1 = time.time()
        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load target
        target_data = self.load_target(label=label)
        full_data.update(target_data)
        print("a1", time.time() - t1)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self) -> dict:
        r"""Load metadata of the GTZAN dataset.
        """

        labels = GTZAN.labels

        meta_dict = {
            "label": [],
            "audio_name": [],
            "audio_path": []
        }

        audios_dir = Path(self.root, "genres")

        for genre in labels:

            audio_names = sorted(os.listdir(Path(audios_dir, genre)))
            # len(audio_names) = 1000

            train_audio_names, test_audio_names = self.split_train_test(audio_names)
            # len(train_audio_names) = 900
            # len(test_audio_names) = 100

            if self.split == "train":
                filtered_audio_names = train_audio_names

            elif self.split == "test":
                filtered_audio_names = test_audio_names

            for audio_name in filtered_audio_names:

                audio_path = Path(audios_dir, genre, audio_name)

                meta_dict["label"].append(genre)
                meta_dict["audio_name"].append(audio_name)
                meta_dict["audio_path"].append(audio_path)

        return meta_dict

    def split_train_test(self, audio_names: list) -> tuple[list, list]:

        train_audio_names = []
        test_audio_names = []

        test_ids = range(self.test_fold * 10, (self.test_fold + 1) * 10)
        # E.g., if test_fold = 3, then test_ids = [30, 31, 32, ..., 39]

        for audio_name in audio_names:

            audio_id = int(re.search(r'\d+', audio_name).group())
            # E.g., if audio_name is "blues.00037.au", then audio_id = 37

            if audio_id in test_ids:
                test_audio_names.append(audio_name)

            else:
                train_audio_names.append(audio_name)

        return train_audio_names, test_audio_names

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

    def load_target(self, label: str) -> np.ndarray:

        classes_num = GTZAN.classes_num
        lb_to_ix = GTZAN.lb_to_ix

        target = np.zeros(classes_num, dtype="float32")
        class_ix = lb_to_ix[label]
        target[class_ix] = 1
        # target: (classes_num,)

        data = {
            "target": target,
            "label": label
        }

        if self.target_transform:
            data = self.target_transform(data)

        return data


if __name__ == "__main__":
    r"""Example.
    """

    from torch.utils.data import DataLoader

    root = "/datasets/gtzan"

    sr = 16000

    dataset = GTZAN(
        root=root,
        split="train",
        test_fold=0,
        sr=sr
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        
        n = 0
        audio_path = data["audio_path"][n]
        start_time = data["start_time"][n].cpu().numpy()
        audio = data["audio"][n].cpu().numpy()
        target = data["target"][n].cpu().numpy()
        label = data["label"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("start_time:", start_time)
    print("audio:", audio.shape)
    print("target:", target.shape)
    print("label:", label)