from __future__ import annotations

from pathlib import Path

import librosa
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class ClothoAQA(Dataset):
    r"""Clotho-AQA [1] is an audio query-answering dataset consisting of 1,991
    audio samples from the Clotho dataset. Each audio sample has 6 associated 
    questions. For each question, the answers are provided by three different 
    annotators making a total of 35,838 question-answer pairs. The total 
    duration is 12.5 hours. The audio samples range from 15 to 30 seconds in 
    length. After decompression, the dataset size is 3.7 GB.

    [1] S. Lipping, et al., Clotho-AQA: A Crowdsourced Dataset for Audio 
    Question Answering, ArXiv, 2020

    The dataset looks like:

        clotho-aqa (3.7 GB)
        ├── audio_files (2894 wavs)
        ├── clotho_aqa_metadata.csv
        ├── clotho_aqa_test.csv
        ├── clotho_aqa_train.csv
        ├── clotho_aqa_val.csv
        └── LICENSE.txt
    """

    URL = "https://zenodo.org/records/6473207"

    DURATION = 44847.45  # Dataset duration (s), 12.5 hours, including 
    # train, val, and test sets.

    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "val", "test"] = "train",
        sr: float = 44100,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=10.),
        transform: None | callable = Mono(),
        question_transform: None | callable = None,
        target_transform: None | callable = None
    ) -> None:
    
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.question_transform = question_transform
        self.target_transform = target_transform

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {Clotho.URL}")

        self.audios_dir = Path(root, "audio_files")
        self.meta_csv = Path(root, "clotho_aqa_{}.csv".format(split))

        self.meta_dict = self.load_meta(self.meta_csv)

    def __getitem__(self, index: int) -> dict:

        
        audio_path = self.meta_dict["audio_path"][index]
        answering = self.meta_dict["answering"][index]
        question = self.meta_dict["question"][index]

        full_data = {
            "dataset_name": "Clotho-AQA",
            "audio_path": audio_path,
        }

        # Load audio data
        audio_data = self.load_audio_data(path=audio_path)
        full_data.update(audio_data)

        # Load question data
        question_data = self.load_question_data(question=question)
        full_data.update(question_data)

        # Load target data
        target_data = self.load_target_data(answering=answering)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str) -> dict:

        meta_dict = {
            "audio_name": [], 
            "audio_path": [], 
            "question": [], 
            "answering": []
        }

        df = pd.read_csv(meta_csv, sep=',')

        for n in range(len(df)):

            audio_name = df["file_name"][n]
            audio_path = str(Path(self.audios_dir, audio_name))

            meta_dict["audio_name"].append(audio_name)
            meta_dict["audio_path"].append(audio_path)
            meta_dict["question"].append(df["QuestionText"][n])
            meta_dict["answering"].append(df["answer"][n])

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

    def load_question_data(self, question: str) -> dict:

        # Transform target
        if self.target_transform:
            question = call(transform=self.target_transform, x=question)

        data = {
            "question": question
        }

        return data

    def load_target_data(self, answering: str) -> dict:

        target = answering

        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)

        data = {
            "answering": answering,
            "target": target,
        }

        return data