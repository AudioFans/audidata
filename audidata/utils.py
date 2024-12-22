import random
from typing import Any, Union, Callable


def call(
    transform: Union[callable, list[callable]], 
    x: Any
) -> Any:

    if isinstance(transform, Callable):
        return transform(x)

    elif isinstance(transform, list):
        for trans in transform:
            x = trans(x)
        return x

    else:
        raise TypeError(transform)


class RandomChoice:
    def __init__(self, callables: object, weights: list[float]):
        
        self.callables = callables
        self.weights = weights

    def __call__(self, **kwargs) -> Any:

        call = random.choices(
            population=self.callables, 
            weights=self.weights
        )[0]

        return call(**kwargs)


# class Compose:
#     def __init__(self, callables: object):
        
#         self.callables = callables

#     def __call__(self, data: dict) -> dict:

#         for call in self.callables:
#             data = call(data)

#         return data


