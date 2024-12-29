from __future__ import annotations
import random
from typing import Callable
import numpy as np


def call(
    transform: callable | list[callable], 
    x: Any
) -> Any:
    r"""Transform input/output data."""

    if isinstance(transform, Callable):
        return transform(x)

    elif isinstance(transform, list):
        for trans in transform:
            x = trans(x)
        return x

    else:
        raise TypeError(transform)


class RandomChoice:
    def __init__(self, callables: callable, weights: list[float]):
        
        self.callables = callables
        self.weights = weights

    def __call__(self, **kwargs) -> Any:

        call = random.choices(
            population=self.callables, 
            weights=self.weights
        )[0]

        return call(**kwargs)
