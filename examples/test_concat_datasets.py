from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from audidata.datasets import GTZAN, MUSDB18HQ
from audidata.samplers import MultiDatasetsBatchSampler


if __name__ == "__main__":
    r"""Example of concatenating multiple datasets. 

    For each batch, first, sample a dataset. Then, sample all batch indexes 
    from the sampled dataset.
    """
    
    gtzan_root = "/datasets/gtzan"
    musdb18hq_root = "/datasets/musdb18hq"

    sr = 16000

    gtzan_dataset = GTZAN(
        root=gtzan_root,
        split="train",
        test_fold=0,
        sr=sr
    )
    
    musdb18hq_dataset = MUSDB18HQ(
        root=musdb18hq_root,
        split="train",
        sr=sr,
    )
    
    dataset_list = [gtzan_dataset, musdb18hq_dataset]

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
        
        print(batch_idx, data.keys())

        if batch_idx == 20:
            break