from pathlib import Path
from typing import Callable, Dict, Optional

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load, random_start_time
from audidata.io.midi import notes_to_data, read_single_track_midi
from audidata.utils.collate import collate_list_fn


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

    pitches_num = 128

    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        mono: bool = True,
        clip_duration: float = 10.,
        fps: int = 100,
        extend_pedal: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.mono = mono
        self.clip_duration = clip_duration
        self.fps = fps
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform

        self.clip_frames = round(self.clip_duration * self.fps) + 1
        self.clip_samples = round(self.clip_duration * self.sr)

        meta_csv = Path(self.root, "maestro-v3.0.0.csv")

        self.meta_dict = self.load_meta(meta_csv)
        
    def __getitem__(self, index: int) -> Dict:

        audio_path = Path(self.root, self.meta_dict["audio_name"][index])
        midi_path = Path(self.root, self.meta_dict["midi_name"][index]) 
        duration = self.meta_dict["duration"][index]

        clip_start_time = random_start_time(audio_path)

        # Load audio
        audio = self.load_audio(path=audio_path, offset=clip_start_time)
        # shape: (channels, audio_samples)
        
        # Load target
        target_data = self.load_target(midi_path=midi_path, clip_start_time=clip_start_time)
        # shape: (classes_num,)

        full_data = {
            "audio_path": str(audio_path),
            "clip_start_time": clip_start_time,
            "audio": audio,
        }

        # Merge dict
        full_data.update(target_data)

        return full_data

    def __len__(self):

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str) -> Dict:
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

    def load_audio(self, path: str, offset: float) -> np.ndarray:

        audio = load(
            path,
            sr=self.sr,
            mono=self.mono,
            offset=offset,
            duration=self.clip_duration
        )
        # shape: (channels, audio_samples)

        audio = librosa.util.fix_length(data=audio, size=self.clip_samples, axis=-1)
        # shape: (channels, audio_samples)

        if self.transform is not None:
            audio = self.transform(audio)

        return audio

    def load_target(self, midi_path: str, clip_start_time: float) -> Dict:

        pitches_num = MAESTRO.pitches_num

        notes, pedals = read_single_track_midi(midi_path=midi_path, extend_pedal=self.extend_pedal)

        target_data = notes_to_data(
            notes=notes, 
            clip_frames=self.clip_frames, 
            classes_num=self.pitches_num, 
            clip_start_time=clip_start_time, 
            clip_duration=self.clip_duration, 
            fps=self.fps
        )

        if self.target_transform:
            target_data = self.target_transform(target_data)

        return target_data


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
        clip_duration=10.,
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
