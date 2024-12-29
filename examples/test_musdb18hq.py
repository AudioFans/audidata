from pathlib import Path
from torch.utils.data import DataLoader

from audidata.io.crops import RandomCrop
from audidata.datasets import MUSDB18HQ
from audidata.samplers import InfiniteSampler, MUSDB18HQ_RandomSongSampler


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
        crop=RandomCrop(clip_duration=2., end_pad=0.),
        target_stems=["vocals", "drums"],
        time_align="group",
        mixture_transform=None,
        group_transform=None,
        stem_transform=None
    )

    print(dataset[3])
    print(dataset[{"vocals": 3, "bass": 11, "drums": 3, "other": 11}])

    sampler1 = InfiniteSampler(dataset)  # Mix stems from a same song.
    sampler2 = MUSDB18HQ_RandomSongSampler(dataset)  # Mix stems from different songs. Better performance

    dataloader = DataLoader(dataset=dataset, batch_size=4, sampler=sampler2)

    for data in dataloader:
        print(data)
        target = data["target"][0].cpu().numpy()
        bg = data["background"][0].cpu().numpy()
        mixture = data["mixture"][0].cpu().numpy()
        break

    import soundfile
    Path("results").mkdir(parents=True, exist_ok=True)
    soundfile.write(file="results/musdb18hq_target.wav", data=target.T, samplerate=sr)
    soundfile.write(file="results/musdb18hq_bg.wav", data=bg.T, samplerate=sr)
    soundfile.write(file="results/musdb18hq_mixture.wav", data=mixture.T, samplerate=sr)


