# from audidata.transforms.audio import StartCrop, RandomCrop
from audidata.transforms.audio import ToMono
from audidata.transforms.midi import Note2Token

'''
import random
from typing import Any

class RandomChoice:
	def __init__(self, transforms: object, weights: list[float]):
		
		self.transforms = transforms
		self.weights = weights

	def __call__(self, data: Any) -> Any:

		transform = random.choices(
			population=self.transforms, 
			weights=self.weights
		)[0]

		return transform(data)
'''