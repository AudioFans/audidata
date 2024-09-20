"""Generate visualizations for a random sample from the Slakh2100 dataset. Use this script to inspect the audio and multi-track piano roll data.
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from pathlib import Path
from torch.utils.data import DataLoader
import colorsys

from audidata.datasets import Slakh2100
from audidata.utils import Compose
from audidata.io.crops import RandomCrop
from audidata.transforms.midi import MultiTrackPianoRoll

def generate_distinct_colors(n):
    # generate evenly spaced colors in HSV space
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    return list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

if __name__ == '__main__':
    root = "/root/slakh2100_flac_redux"
    # root = "/datasets/slakh2100_flac"
    sr = 16000
    fps = 100
    pitches_num = 128

    target_transforms = Compose(callables=[
        MultiTrackPianoRoll(
            fps=fps,
            pitches_num=pitches_num,
        )
    ])

    dataset = Slakh2100(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=10., end_pad=9.9),
        target_transform=target_transforms,
    )

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=1, 
        num_workers=0,
    )

    for data in dataloader:
        audio = data["audio"][0].cpu().numpy()
        track_rolls = data["track_rolls"][0]
        break

    print("Audio shape:", audio.shape)

    Path("results").mkdir(parents=True, exist_ok=True)
    out_path = "results/slakh_example.wav"
    soundfile.write(file=out_path, data=audio.T, samplerate=sr)
    print(f"Audio out: {out_path}")

    mel = librosa.feature.melspectrogram(y=audio[0], sr=sr, n_fft=2048, 
        hop_length=160, n_mels=229, fmin=0, fmax=8000)

    num_tracks = len(track_rolls)
    colors = generate_distinct_colors(num_tracks)

    fig, axs = plt.subplots(2, 1, figsize=(20, 15), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    axs[0].imshow(librosa.power_to_db(mel, ref=np.max), aspect='auto', origin='lower', cmap='jet')
    axs[0].set_title("Mel Spectrogram")
    axs[0].set_ylabel("Mel Bins")

    for i, (track_roll, color) in enumerate(zip(track_rolls, colors)):
        frame_roll = track_roll["frame_roll"].T
        axs[1].imshow(frame_roll, aspect='auto', origin='lower', cmap=plt.cm.colors.ListedColormap([color, 'white']), alpha=0.5)
        axs[1].set_title("Multi-track Piano Roll")
        axs[1].set_ylabel("Pitch")

    axs[1].set_xlabel("Time (frames)")

    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.5, label=f"{track_roll['instrument']} ({'Drum' if track_roll['is_drum'] else 'Pitched'})") 
                       for track_roll, color in zip(track_rolls, colors)]
    axs[1].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.tight_layout()
    fig_path = "results/slakh_visualization.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved overall visualization to {fig_path}")
    plt.close()

    for i, (track_roll, color) in enumerate(zip(track_rolls, colors)):
        fig, axs = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
        
        for j, (roll_type, roll) in enumerate([("Onset", track_roll["onset_roll"]), 
                                               ("Offset", track_roll["offset_roll"]), 
                                               ("Velocity", track_roll["velocity_roll"])]):
            im = axs[j].imshow(roll.T, aspect='auto', origin='lower', cmap=plt.cm.colors.ListedColormap([color, 'white']))
            axs[j].set_title(f"Track {i+1}: {roll_type} Roll - {track_roll['instrument']} ({'Drum' if track_roll['is_drum'] else 'Pitched'})")
            axs[j].set_ylabel("Pitch")
            fig.colorbar(im, ax=axs[j])
        
        plt.xlabel("Time (frames)")
        plt.tight_layout()
        
        track_fig_path = f"results/slakh_track_{i+1}_rolls_colored.png"
        plt.savefig(track_fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved track {i+1}'s visualization to {track_fig_path}")
        
        plt.close()