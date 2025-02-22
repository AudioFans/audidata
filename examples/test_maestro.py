import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from pathlib import Path
from torch.utils.data import DataLoader

from audidata.datasets import MAESTRO
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll


if __name__ == '__main__':
    r"""Example.

    The MAESTRO V3.0.0 dataset: https://magenta.tensorflow.org/datasets/maestro

    The dataset looks like:

        maestro-v3.0.0 (131 GB)
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

    root = "/datasets/maestro-v3.0.0"

    # Dataset
    dataset = MAESTRO(
        root=root,
        split="train",
        sr=44100,
        crop=RandomCrop(clip_duration=10., end_pad=9.9),
        transform=Mono(),
        load_target=True,
        extend_pedal=True,
        target_transform=PianoRoll(fps=100, pitches_num=128),
    )

    # Example of fetch a data
    print(dataset[0].keys())

    # Example of dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=4)

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
    soundfile.write(file=out_path, data=audio.T, samplerate=dataset.sr)
    print("Write out audio to {}".format(out_path))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio[0], sr=dataset.sr, n_fft=2048, 
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
