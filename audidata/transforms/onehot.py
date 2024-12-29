import numpy as np


class OneHot:
    r"""Convert label to one-hot representation.
    """

    def __init__(self, classes_num: int):
        self.classes_num = classes_num

    def __call__(self, index: int) -> np.ndarray:
        
        target = np.zeros(self.classes_num, dtype="float32")
        target[index] = 1

        return target