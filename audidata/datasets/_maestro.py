from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import ToMono
from audidata.transforms.midi import PianoRoll
from audidata.io._midi import read_single_track_midi 
from audidata.collate.base import collate_list_fn
import time

default_collate_fn_map.update({list: collate_list_fn})


class MAESTRO(Dataset):
    r"""MAESTRO [1] is a dataset containing 199 hours of 1,276 audio files and 
    aligned MIDI files captured by Yamaha Disklaiver. Audios are sampled at 44,100 Hz. 
    After decompression, the dataset is 131 GB.

    [1] C. Hawthorne, et al., Enabling Factorized Piano Music Modeling and 
    Generation with the MAESTRO Dataset, ICLR, 2019

    The dataset looks like:

        dataset_root (131 GB)
        ├── 2004 (132 songs, wav + flac + midi + tsv)
        ├── 2006 (115 songs, wav + flac + midi + tsv)
        ├── 2008 (147 songs, wav + flac + midi + tsv)
        ├── 2009 (125 songs, wav + flac + midi + tsv)
        ├── 2011 (163 songs, wav + flac + midi + tsv)
        ├── 2013 (127 songs, wav + flac + midi + tsv)
        ├── 2014 (105 songs, wav + flac + midi + tsv)
        ├── 2015 (129 songs, wav + flac + midi + tsv)
        ├── 2017 (140 songs, wav + flac + midi + tsv)
        ├── 2018 (93 songs, wav + flac + midi + tsv)
        ├── LICENSE
        ├── maestro-v3.0.0.csv
        ├── maestro-v3.0.0.json
        └── README
    """

    url = "https://magenta.tensorflow.org/datasets/maestro"

    duration = 717232.49  # Dataset duration (s), 199 hours, including training, 
    # validation, and testing.

    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        crop: Optional[callable] = RandomCrop(clip_duration=10., end_pad=9.9),
        transform: Optional[callable] = ToMono(),
        target: bool = True,
        extend_pedal: bool = True,
        target_transform: Optional[callable] = PianoRoll(fps=100, pitches_num=128),
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target = target
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform

        meta_csv = Path(self.root, "maestro-v3.0.0.csv")

        self.meta_dict = self.load_meta(meta_csv)
        
    def __getitem__(self, index: int) -> dict:

        audio_path = Path(self.root, self.meta_dict["audio_name"][index])
        midi_path = Path(self.root, self.meta_dict["midi_name"][index]) 
        duration = self.meta_dict["duration"][index]

        t1 = time.time()

        full_data = {
            "dataset_name": "MAESTRO",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)
        print("a1", time.time() - t1)
        
        t1 = time.time()
        # Load target
        if self.target:
            target_data = self.load_target(
                midi_path=midi_path, 
                start_time=audio_data["start_time"],
                clip_duration=audio_data["duration"]
            )
            full_data.update(target_data)
        print("b1", time.time() - t1)

        return full_data

    def __len__(self):

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str) -> dict:
        r"""Load meta dict.
        """

        df = pd.read_csv(meta_csv, sep=',')

        indexes = df["split"].values == self.split

        meta_dict = {
            "midi_name": df["midi_filename"].values[indexes],
            "audio_name": df["audio_filename"].values[indexes],
            "duration": df["duration"].values[indexes]
        }

        return meta_dict

    def load_audio(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)
        
        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.
            clip_duration = None

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

    def load_target(
        self, 
        midi_path: str, 
        start_time: float, 
        clip_duration: float
    ) -> dict:
        
        notes, pedals = read_single_track_midi(
            midi_path=midi_path, 
            extend_pedal=self.extend_pedal
        )

        data = {
            "note": notes,
            "pedal": pedals,
            "start_time": start_time,
            "clip_duration": clip_duration
        } 

        if self.target_transform:
            data = self.target_transform(data)

        return data


if __name__ == '__main__':
    r"""Example.
    """

    import matplotlib.pyplot as plt
    import soundfile
    from torch.utils.data import DataLoader

    root = "/datasets/maestro-v3.0.0"

    sr = 16000

    # Dataset
    dataset = MAESTRO(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=10., end_pad=9.9),
        target_transform=PianoRoll(fps=100, pitches_num=128),
    )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=4, 
        num_workers=0, 
    )

    for data in dataloader:

        n = 0
        audio = data["audio"][n].cpu().numpy()
        frame_roll = data["frame_roll"][n].cpu().numpy()
        onset_roll = data["onset_roll"][n].cpu().numpy()
        offset_roll = data["offset_roll"][n].cpu().numpy()
        velocity_roll = data["velocity_roll"][n].cpu().numpy()
        break

    # ------ Visualize ------
    print("audio:", audio.shape)
    print("frame_roll:", frame_roll.shape)
    print("onset_roll:", frame_roll.shape)
    print("offset_roll:", frame_roll.shape)
    print("velocity_roll:", frame_roll.shape)

    # Write audio
    out_path = "out.wav"
    soundfile.write(file=out_path, data=audio.T, samplerate=sr)
    print("Write out audio to {}".format(out_path))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio[0], sr=sr, n_fft=2048, 
        hop_length=160, n_mels=229, fmin=0, fmax=8000)

    # Plot
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(20, 15))
    axs[0].matshow(np.log(mel), origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[2].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[3].matshow(offset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[4].matshow(velocity_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    fig_path = "out.pdf"
    plt.savefig(fig_path)
    print("Write out fig to {}".format(fig_path))
