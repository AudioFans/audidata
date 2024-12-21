import random

from torch.utils.data import Dataset


class InfiniteSampler:
    r"""Randomly yield index infinitely.
    """

    def __init__(self, dataset: Dataset):

        self.indexes = list(range(len(dataset)))
        random.shuffle(self.indexes)
        
    def __iter__(self) -> int:

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer]
            pointer += 1

            yield index