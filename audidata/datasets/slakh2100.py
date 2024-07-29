from pathlib import Path
from typing import Optional

import yaml
import os
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import ToMono
from audidata.transforms.midi import PianoRoll
from audidata.io.midi import read_single_track_midi
from audidata.collate.base import collate_list_fn


default_collate_fn_map.update({list: collate_list_fn})


class Slakh2100(Dataset):
    r"""Slakh2100 [1] is a multiple track MIDI-audio paired dataset containing
    145 hours of 2,100 audio audio files rendered by MIDI files. Audios are 
    sampled at 44,100 Hz. After decompression, the dataset is 101 GB.

    [1] E. Manilow, Cutting music source separation some Slakh: A dataset to 
    study the impact of training data quality and quantity, WASPAA, 2019

    After decompression, dataset looks like:

        dataset_root (131 GB)
        ├── train (1500 songs)
        │   ├── Track00001
        │   │   ├── all_src.mid
        │   │   ├── metadata.yaml
        │   │   ├── MIDI
        │   │   │   ├── S00.mid
        │   │   │   ├── S01.mid
        │   │   │   └── ...
        │   │   ├── mix.flac
        │   │   └── stems
        │   │       ├── S00.flac
        │   │       ├── S01.flac
        │   │       └── ...
        │   ├── Track00002
        │   └── ...
        ├── validation (375 songs)
        └── test (225 songs) 
    """

    url = "https://zenodo.org/records/4599666"

    duration = 521806.45  # Dataset duration (s), 145 hours, including training, 
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

        audios_dir = Path(self.root, self.split)
        self.meta_dict = {"audio_name": sorted(os.listdir(audios_dir))}

    def __getitem__(self, index: int) -> dict:

        prefix = Path(self.root, self.split, self.meta_dict["audio_name"][index])
        audio_path = Path(prefix, "mix.flac")
        meta_csv = Path(prefix, "metadata.yaml")
        midis_dir = Path(prefix, "MIDI")

        full_data = {
            "dataset_name": "Slakh2100",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load target
        if self.target:
            target_data = self.load_target(
                meta_csv=meta_csv,
                midis_dir=midis_dir,
                start_time=audio_data["start_time"],
                clip_duration=audio_data["duration"]
            )
            full_data.update(target_data)

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
        meta_csv: str,
        midis_dir: str,
        start_time: float, 
        clip_duration: float
    ) -> dict:
        
        with open(meta_csv, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        data = {
            # "mix": []
            "tracks": []
        }

        for stem_name, stem_data in meta["stems"].items():

            if not stem_data["midi_saved"]:
                continue

            # We do not load "midi_program_name" and "program_num" because those
            # information are not used to render audio.

            inst_class = stem_data["inst_class"]
            is_drum = stem_data["is_drum"]
            plugin_name = stem_data["plugin_name"]

            midi_path = Path(midis_dir, "{}.mid".format(stem_name))

            notes, pedals = read_single_track_midi(
                midi_path=midi_path, 
                extend_pedal=self.extend_pedal
            )

            tmp = {
                "track_name": stem_name,
                "inst_class": inst_class,
                "is_drum": is_drum,
                "plugin_name": plugin_name,
                "note": notes,
                "pedal": pedals,
                "start_time": start_time,
                "clip_duration": clip_duration
            }

            data["tracks"].append(tmp)
        data["start_time"] = start_time
        data["clip_duration"] = clip_duration

        if self.target_transform:
            data = self.target_transform(data)

        return data


class AA:
    def __init__(self):
        self.piano_roll_extractor = PianoRoll(fps=100, pitches_num=128)

    def __call__(self, data: dict) -> dict:

        for track in data["tracks"]: 
            track = self.piano_roll_extractor(track)
            
        

        from IPython import embed; embed(using=False); os._exit(0)



if __name__ == '__main__':
    r"""Example.
    """

    import matplotlib.pyplot as plt
    import soundfile
    from torch.utils.data import DataLoader

    from audidata.utils import Compose

    root = "/datasets/slakh2100_flac"

    sr = 16000

    target_transform = Compose([
        AA()
    ])

    # Dataset
    dataset = Slakh2100(
        root=root,
        split="test",
        sr=sr,
        crop=RandomCrop(clip_duration=10., end_pad=9.9),
        # target_transform=PianoRoll(fps=100, pitches_num=128),
        target_transform=target_transform,
    )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=4, 
        num_workers=0, 
    )

    for data in dataloader:

        n = 0
        from IPython import embed; embed(using=False); os._exit(0)
        audio = data["audio"][n].cpu().numpy()
        # frame_roll = data["frame_roll"][n].cpu().numpy()
        # onset_roll = data["onset_roll"][n].cpu().numpy()
        # offset_roll = data["offset_roll"][n].cpu().numpy()
        # velocity_roll = data["velocity_roll"][n].cpu().numpy()
        # break

    # ------ Visualize ------
    # print("audio:", audio.shape)
    # print("frame_roll:", frame_roll.shape)
    # print("onset_roll:", frame_roll.shape)
    # print("offset_roll:", frame_roll.shape)
    # print("velocity_roll:", frame_roll.shape)

    # Write audio
    out_path = "out.wav"
    soundfile.write(file=out_path, data=audio.T, samplerate=sr)
    print("Write out audio to {}".format(out_path))

    # # Mel spectrogram
    # mel = librosa.feature.melspectrogram(y=audio[0], sr=sr, n_fft=2048, 
    #     hop_length=160, n_mels=229, fmin=0, fmax=8000)

    # # Plot
    # fig, axs = plt.subplots(5, 1, sharex=True, figsize=(20, 15))
    # axs[0].matshow(np.log(mel), origin='lower', aspect='auto', cmap='jet')
    # axs[1].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[1].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[2].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[3].matshow(offset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[4].matshow(velocity_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # fig_path = "out.pdf"
    # plt.savefig(fig_path)
    # print("Write out fig to {}".format(fig_path))
