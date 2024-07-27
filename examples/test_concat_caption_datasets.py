from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from audidata.datasets import Clotho, AudioCaps, WavCaps
from audidata.samplers import MultiDatasetsBatchSampler
from audidata.io.crops import StartCrop, RandomCrop
from audidata.utils import RandomChoice


if __name__ == "__main__":
    r"""Example of concatenating multiple datasets. 

    For each batch, first, sample a dataset. Then, sample all batch indexes 
    from the sampled dataset.
    """

    clotho_root = "/datasets/clotho"
    audiocaps_root = "/datasets/audiocaps"
    wavcaps_root = "/datasets/wavcaps"

    sr = 16000
    clip_duration = 10.

    crop1 = StartCrop(clip_duration=clip_duration)
    crop2 = RandomCrop(clip_duration=clip_duration, end_pad=0.)
    crop = RandomChoice(callables=[crop1, crop2], weights=[0.5, 0.5])

    clotho_dataset = Clotho(
        root=clotho_root,
        split="train",
        sr=sr,
        crop=crop,
    )

    audiocaps_dataset = AudioCaps(
        root=audiocaps_root,
        split="train",
        sr=sr,
        crop=crop1
    )

    wavcaps_dataset = WavCaps(
        root=wavcaps_root,
        sr=sr,
        crop=crop1
    )
    
    dataset_list = [clotho_dataset, audiocaps_dataset, wavcaps_dataset]

    # Concat dataset
    dataset = ConcatDataset(datasets=dataset_list)

    # Multiple dataset sampler
    batch_sampler = MultiDatasetsBatchSampler(
        datasets=dataset_list, 
        batch_size=4,
        weights=None
    )

    # Dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
    )

    for batch_idx, data in enumerate(dataloader):
        
        
        n = 0
        dataset_name = data["dataset_name"][n]
        audio_path = data["audio_path"][n]
        start_time = data["start_time"][n].cpu().numpy()
        audio = data["audio"][n].cpu().numpy()
        caption = data["caption"][n]

        print(batch_idx)
        print("dataset_name:", dataset_name)
        print("audio_path:", audio_path)
        print("start_time:", start_time)
        print("audio:", audio.shape)
        print("caption:", caption)

        if batch_idx == 200:
            break