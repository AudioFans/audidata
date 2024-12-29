from __future__ import annotations
import random
import numpy as np
from typing import Optional

from torch.utils.data import Dataset


class MultiDatasetsBatchSampler:
    r"""Multiple datasets sampler. First, sample a dataset. Then, sample all 
    batch indexes from the sampled dataset.
    """

    def __init__(
        self, 
        datasets: list[Dataset], 
        batch_size: int,
        weights: Optional[np.ndarray] = None  # Probability of datasets to be sampled
    ):

        self.batch_size = batch_size
        self.datasets_num = len(datasets)
        
        # Probability of datasets to be sampled
        if weights:
            self.weights = weights
        else:
            self.weights = [dataset.duration for dataset in datasets]

        # Indexes of audio files of all datasets
        self.indexes_list = [list(range(len(dataset))) for dataset in datasets]

        for indexes in self.indexes_list:
            random.shuffle(indexes)

        self.datasets_lens = [len(indexes) for indexes in self.indexes_list]

    def __iter__(self) -> list[int]:
        r"""Sample a batch of indexes. First, sample a dataset. Then, sample all 
        batch indexes from the sampled dataset."""

        pointer_list = [0 for _ in range(self.datasets_num)]

        while True:

            # Sample a dataset
            dataset_index = random.choices(
                population=range(self.datasets_num), 
                weights=self.weights
            )[0]

            batch_indexes = []

            for _ in range(self.batch_size):

                indexes = self.indexes_list[dataset_index]
                pointer = pointer_list[dataset_index]

                # Reset pointer and shuffle
                if pointer == len(indexes):
                    random.shuffle(indexes)
                    pointer = 0

                # Get index
                index = indexes[pointer]
                pointer += 1
                pointer_list[dataset_index] = pointer

                # Get absolute index
                abs_index = int(np.sum(self.datasets_lens[0 : dataset_index])) + index
                batch_indexes.append(abs_index)

            yield batch_indexes