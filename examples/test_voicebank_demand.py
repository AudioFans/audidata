from pathlib import Path
from torch.utils.data import DataLoader

from audidata.io.crops import RandomCrop
from audidata.datasets import VoicebankDemand
from audidata.samplers import InfiniteSampler
from audidata.transforms import Mono, Normalize


if __name__ == "__main__":
    r"""Example.

    The VoicebankDemand dataset: https://datashare.ed.ac.uk/handle/10283/2791

    The dataset looks like:

        voicebank-demand (2.5 GB)
        ├── clean_trainset_wav
        │    └── ... (11572 wavs)
        ├── noisy_trainset_wav
        │    └── ... (11572 wavs)
        ├── clean_testset_wav
        │    └── ... (824 wavs)
        └── noisy_testset_wav
             └── ... (824 wavs)
    """
    
    root = "/datasets/voicebank-demand"

    sr = 44100

    dataset = VoicebankDemand(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=2., end_pad=0.),
        stem_transform=[Mono(), Normalize()]
    )

    # Example of get one data
    print(dataset[3])
    
    # Example of sampler and dataloader
    sampler = InfiniteSampler(dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=4, sampler=sampler)

    for data in dataloader:
        print(data)
        target = data["target"][0].cpu().numpy()
        bg = data["background"][0].cpu().numpy()
        mixture = data["mixture"][0].cpu().numpy()
        break

    import soundfile
    Path("results").mkdir(parents=True, exist_ok=True)
    soundfile.write(file="results/voicebank_demand_target.wav", data=target.T, samplerate=sr)
    soundfile.write(file="results/voicebank_demand_bg.wav", data=bg.T, samplerate=sr)
    soundfile.write(file="results/voicebank_demand_mixture.wav", data=mixture.T, samplerate=sr)


