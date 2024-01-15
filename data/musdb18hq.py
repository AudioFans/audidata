import torch
from pathlib import Path
import pandas as pd
import random
import torchaudio
from pathlib import Path
import librosa


class Musdb18HQ:
	
	def __init__(self,
		root: str = None, 
		split: str = "train",
		segment_seconds: float = 4,
		tokenizer=None,
	):

		self.root = root
		self.split = split
		self.segment_seconds = segment_seconds
		
		self.fps = 100
		self.segment_frames = int(self.segment_seconds * self.fps) + 1

		self.audios_dir = Path(self.root, self.split)
		self.audio_names = sorted(list(Path(self.audios_dir).glob("*")))
		self.audios_num = len(self.audio_names)
		
		self.source_types = ["mixture", "vocals"]
		
	def __getitem__(self, index):

		audio_index = random.randint(0, self.audios_num - 1)
		audio_name = self.audio_names[audio_index]
		# audio_name = self.audio_names[index]

		data_dict = {}

		audio_path = Path(self.audios_dir, audio_name, "mixture.wav")
		duration = librosa.get_duration(path=audio_path)
		orig_sr = librosa.get_samplerate(path=audio_path)

		segment_start_time = random.uniform(0, duration - self.segment_seconds)
		segment_start_sample = int(segment_start_time * orig_sr)
		segment_samples = int(self.segment_seconds * orig_sr)

		for source_type in self.source_types:

			audio_path = Path(self.audios_dir, audio_name, "{}.wav".format(source_type))

			segment, _ = torchaudio.load(
				audio_path, 
				frame_offset=segment_start_sample, 
				num_frames=segment_samples
			)
			# (channels, audio_samples)

			data_dict[source_type] = segment

		return data_dict

	def __len__(self):

		# return self.audios_num
		return 1000


'''
class Sampler(Sampler):
    
    def __init__(self):
        
        
        self.audios_num = hf['target'].shape[0]

    def __iter__(self):
        
        indexes = np.arange(self.audios_num)
        
        while(True):

        	meta_dict = {
        		"source_types": ["mixture", "vocals"]
        	}
            
    def __len__(self):
        return -1
'''