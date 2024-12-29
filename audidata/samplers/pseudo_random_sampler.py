from typing import Sized

import numpy as np


class PseudoRandomSampler:
    def __init__(self, dataset: Sized) -> None:
        r"""Pseudo random sampler. This sampler is usually used for evaluation.
        """

        rs = np.random.RandomState(1234)  # Random state
        self.indexes = rs.permutation(len(dataset))
        
    def __iter__(self) -> int:
        r"""Yield an index."""

        for index in self.indexes:
            yield index