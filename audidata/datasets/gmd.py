from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll
from audidata.io.midi import read_single_track_midi
from audidata.collate.base import collate_list_fn

default_collate_fn_map.update({list: collate_list_fn})

class GMD(Dataset):
    """The Groove MIDI Dataset (GMD) [1] is composed of 13.6 hours of aligned MIDI and (synthesized) audio of human-performed, tempo-aligned expressive drumming. The dataset contains 1,150 MIDI files and over 22,000 measures of drumming.
    
    [1] Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman. "Learning to Groove with Inverse Sequence Transformations." International Conference on Machine Learning (ICML), 2019.
    """
    
    url = "https://magenta.tensorflow.org/datasets/groove"

    duration = 3944.09 # 10 hours, 51 minutes, 38.09 seconds

    def __init__(
        self,
        root: str,
        split: str = "train",
        sr: float = 16000,
        crop: Optional[callable] = RandomCrop(clip_duration=10., end_pad=9.9),
        transform: Optional[callable] = Mono(),
        target: bool = True,
        target_transform: Optional[callable] = PianoRoll(fps=100, pitches_num=128),
    ):
        self.root = Path(root)
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:
        audio_path = self.root / self.meta_dict["audio_filename"][index]
        midi_path = self.root / self.meta_dict["midi_filename"][index]

        full_data = {
            "dataset_name": "Groove MIDI Dataset",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load target
        if self.target:
            target_data = self.load_target(
                midi_path=midi_path,
                start_time=audio_data["start_time"],
                clip_duration=audio_data["duration"]
            )
            full_data.update(target_data)

        return full_data

    def __len__(self):
        return len(self.meta_dict["audio_filename"])

    def load_meta(self) -> dict:
        """Load metadata for the dataset."""
        csv_path = self.root / "info.csv"
        df = pd.read_csv(csv_path)
        df = df[df['split'] == self.split]
        
        meta_dict = {
            "midi_filename": df["midi_filename"].values,
            "audio_filename": df["audio_filename"].values,
            "duration": df["duration"].values,
            "style": df["style"].values,
            "bpm": df["bpm"].values,
            "beat_type": df["beat_type"].values,
            "time_signature": df["time_signature"].values
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

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration if clip_duration else audio_duration
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_target(self, midi_path: str, start_time: float, clip_duration: float) -> dict:
        notes, pedals = read_single_track_midi(
            midi_path=midi_path, 
            extend_pedal=False
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
    import matplotlib.pyplot as plt
    import soundfile
    from torch.utils.data import DataLoader

    root = "/root/groove"

    sr = 16000

    dataset = GMD(
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
    axs[2].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[3].matshow(offset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[4].matshow(velocity_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    fig_path = "out.pdf"
    plt.savefig(fig_path)
    print("Write out fig to {}".format(fig_path))