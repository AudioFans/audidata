from pathlib import Path
from torch.utils.data import DataLoader

from audidata.datasets import MUSDB18HQ


if __name__ == "__main__":
    r"""Example.

    The MUSDB18HQ dataset: https://zenodo.org/records/3338373

    The dataset looks like:

        dataset_root (30 GB)
        ├── train (100 files)
        │   ├── A Classic Education - NightOwl
        │   │   ├── bass.wav
        │   │   ├── drums.wav
        │   │   ├── mixture.wav
        │   │   ├── other.wav
        │   │   └── vocals.wav
        │   ... 
        │   └── ...
        └── test (50 files)
            ├── Al James - Schoolboy Facination
            │   ├── bass.wav
            │   ├── drums.wav
            │   ├── mixture.wav
            │   ├── other.wav
            │   └── vocals.wav
            ... 
            └── ...
    """
    
    root = "/datasets/musdb18hq"

    sr = 44100

    dataset = MUSDB18HQ(
        root=root,
        split="train",
        sr=sr,
        mono=False,
        clip_duration=2.,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:

        n = 0
        audio_path = data["audio_path"][n]
        bass = data["bass"][n].cpu().numpy()
        drums = data["drums"][n].cpu().numpy()
        other = data["other"][n].cpu().numpy()
        vocals = data["vocals"][n].cpu().numpy()
        accompaniment = data["accompaniment"][n].cpu().numpy()
        mixture = data["mixture"][n].cpu().numpy()
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("mixture:", mixture.shape)

    import soundfile
    Path("results").mkdir(parents=True, exist_ok=True)
    soundfile.write(file="results/out_bass.wav", data=bass.T, samplerate=sr)
    soundfile.write(file="results/out_drums.wav", data=drums.T, samplerate=sr)
    soundfile.write(file="results/out_other.wav", data=other.T, samplerate=sr)
    soundfile.write(file="results/out_vocals.wav", data=vocals.T, samplerate=sr)
    soundfile.write(file="results/out_acc.wav", data=accompaniment.T, samplerate=sr)
    soundfile.write(file="results/out_mixture.wav", data=mixture.T, samplerate=sr)