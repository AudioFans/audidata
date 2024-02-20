import torch
from pathlib import Path
import pandas as pd
import random
import torchaudio
import pretty_midi
import os
import numpy as np
from constants import *

class MaestroDataset:
	# reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
	def __init__(
		self, 
		root: str = None, 
		split: str = "train",
		segment_seconds: float = SEGMENT_SECONDS,
		tokenizer=None,
	):
		# self.random = np.random.RandomState(seed=42)
		self.root = root
		self.split = split
		self.segment_seconds = segment_seconds
		self.tokenizer = tokenizer

		self.sample_rate = SAMPLE_RATE
		self.fps = TARGET_FRAMES_PER_SECONDS  # frames per second
		self.pitches_num = 128
		self.segment_frames = int(self.segment_seconds * self.fps) + 1
		self.meta_csv = Path(self.root, "maestro-v2.0.0.csv")

		self.load_meta()

		self.data = []

		for input_files in self.files():
			self.data.append(self.load_all(*input_files))

	def load_meta(self):
		df = pd.read_csv(self.meta_csv, sep=',')

		indexes = df["split"].values == self.split

		self.midi_path = [self.root + "/" +filename for filename in df["midi_filename"].values[indexes]]
		self.audio_path = [self.root + "/" +filename for filename in df["audio_filename"].values[indexes]]
		self.durations = df["duration"].values[indexes] #midi最后一个音符的结束时间
		self.audios_num = len(self.midi_path)

	def files(self):
		files = list(zip(self.audio_path, self.midi_path))
		result = []
		for audio_path, midi_path in files:
			result.append((audio_path, midi_path))
		return result

	def __getitem__(self, index):
		data = self.data[index]
		if self.split != "test":
			segment_start_time = float(int(random.uniform(0, self.durations[index] - self.segment_seconds) * self.sample_rate) / self.sample_rate)
			audio_begin = int(segment_start_time * self.sample_rate)
			audio_end = int(audio_begin + self.segment_seconds * self.sample_rate)

			# Load audio.
			audio = data['audio'][audio_begin:audio_end] # shape: (audio_samples)

			# Load tokens.
			targets_dict = self.load_targets(data['notes'], segment_start_time, self.segment_frames) # shape: (tokens_num,)
		else:
			segment_start_time = 0
			audio = data['audio']
			targets_dict = self.load_targets(data['notes'], segment_start_time, int(len(audio) * self.fps/self.sample_rate) + 1)
		result = {
			"audio": audio,
			"tokens": targets_dict["tokens"],
			"frames_roll": targets_dict["frames_roll"],
			"onsets_roll": targets_dict["onsets_roll"],
			"offsets_roll": targets_dict["offsets_roll"],
			"velocity_roll": targets_dict["velocity_roll"],
			"audio_path": self.audio_path[index],
			"segment_start_time": segment_start_time,
		}

		return result

	def __len__(self):

		return self.audios_num

	def load_all(self, audio_path, midi_path):
		# address to save the .pt file
		saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt').replace("/lan/ifc", ".")
		if os.path.exists(saved_data_path):
			return torch.load(saved_data_path)

		# load audio
		audio, fs = torchaudio.load(audio_path) # (channels, audio_samples)
		audio = torch.mean(audio, dim=0) # shape: (audio_samples,)
		audio = torchaudio.functional.resample(waveform=audio, orig_freq=fs, new_freq=self.sample_rate)# shape: (audio_samples,)

		# load midi
		midi_data = pretty_midi.PrettyMIDI(str(midi_path))
		assert len(midi_data.instruments) == 1
		notes = midi_data.instruments[0].notes

		targets_dict = dict(audio=audio, notes = notes)
		if not os.path.exists(os.path.dirname(saved_data_path)):
			os.makedirs(os.path.dirname(saved_data_path))
		torch.save(targets_dict, saved_data_path)

		return targets_dict

	def load_targets(self, notes, segment_start_time, segment_frames):
		# Load active notes inside the segment.
		if self.split != "test":
			active_notes = []
			segment_end_time = segment_start_time + self.segment_seconds

			for i in range(len(notes)):
				if segment_start_time <= notes[i].start < segment_end_time:
					active_notes.append(notes[i])
		else:
			active_notes = notes

		# Covert notes information to words.
		frames_roll = np.zeros((segment_frames, self.pitches_num))
		onsets_roll = np.zeros((segment_frames, self.pitches_num))
		offsets_roll = np.zeros((segment_frames, self.pitches_num))
		velocity_roll = np.zeros((segment_frames, self.pitches_num))

		#Process the start and end times of notes
		for i in range(len(active_notes)):
			active_notes[i].start = int(np.round((active_notes[i].start - segment_start_time) * self.fps))
			active_notes[i].end = int(np.round((active_notes[i].end - segment_start_time) * self.fps))
			active_notes[i].end = int(min(self.segment_frames, active_notes[i].end))

		# sorted according to the start time, and if the start time is the same, they are sorted according to the pitch
		active_notes = sorted(active_notes, key=lambda note: (note.start, note.pitch))

		words = ["<sos>"]

		for note in active_notes:
			onset_time = note.start
			offset_time = note.end
			pitch = int(note.pitch)
			velocity = note.velocity

			words.append("<time>={}".format(onset_time))
			words.append("<pitch>={}".format(pitch))
			words.append("<velocity>={}".format(velocity))

			onset_index = onset_time
			onset_right = int(min(self.segment_frames, onset_index + 1))
			offset_index = offset_time
			offset_right = int(min(self.segment_frames, offset_index + 1))

			frames_roll[onset_index:offset_index, pitch] = 1
			onsets_roll[onset_index, pitch] = 1
			offsets_roll[offset_index:offset_right, pitch] = 1
			velocity_roll[onset_index:offset_index, pitch] = velocity

		words.append("<eos>")

		# Convert words to tokens.
		tokens = []
		for word in words:
			token = self.tokenizer.stoi(word)
			tokens.append(token)

		targets_dict = {
			"tokens": tokens,
			"frames_roll": frames_roll,
			"onsets_roll": onsets_roll,
			"offsets_roll": offsets_roll,
			"velocity_roll": velocity_roll,
		}

		return targets_dict