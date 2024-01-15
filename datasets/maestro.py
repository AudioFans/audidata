import torch
from pathlib import Path
import pandas as pd
import random
import torchaudio
import pretty_midi
import numpy as np


class MaestroDataset:
	# reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
	def __init__(
		self, 
		root: str = None, 
		split: str = "train",
		segment_seconds: float = 10.,
	):
	
		self.root = root
		self.split = split
		self.segment_seconds = segment_seconds
		self.tokenizer = tokenizer

		self.maestro_sr = 44100
		self.sample_rate = 16000
		self.fps = 100
		self.pitches_num = 128
		self.segment_frames = int(self.segment_seconds * self.fps) + 1

		self.meta_csv = Path(self.root, "maestro-v2.0.0.csv")

		self.load_meta()
				
	def load_meta(self):

		df = pd.read_csv(self.meta_csv, sep=',')

		indexes = df["split"].values == self.split

		self.midi_filenames = df["midi_filename"].values[indexes]
		self.audio_filenames = df["audio_filename"].values[indexes]
		self.durations = df["duration"].values[indexes]
		self.audios_num = len(self.midi_filenames)
		

	def __getitem__(self, index):

		audio_index = random.randint(0, self.audios_num - 1)

		audio_path = Path(self.root, self.audio_filenames[audio_index])
		midi_path = Path(self.root, self.midi_filenames[audio_index]) 
		duration = self.durations[audio_index]

		segment_start_time = random.uniform(0, duration - self.segment_seconds)

		if True:
			audio_path = Path(self.root, self.audio_filenames[0])
			midi_path = Path(self.root, self.midi_filenames[0]) 
			segment_start_time = 100

		# Load audio.
		audio = self.load_audio(audio_path, segment_start_time)
		# shape: (audio_samples)

		# Load tokens.
		targets_dict = self.load_targets(midi_path, segment_start_time)
		# shape: (tokens_num,)

		data = {
			"audio": audio,
			"tokens": targets_dict["tokens"],
			"frames_roll": targets_dict["frames_roll"],
			"onsets_roll": targets_dict["onsets_roll"],
			"audio_path": audio_path,
			"segment_start_time": segment_start_time,
		}

		return data

	def __len__(self):

		return self.audios_num

	def load_audio(self, audio_path, segment_start_time):

		segment_start_sample = int(segment_start_time * self.maestro_sr)
		segment_samples = int(self.segment_seconds * self.maestro_sr)

		audio, fs = torchaudio.load(
			audio_path, 
			frame_offset=segment_start_sample, 
			num_frames=segment_samples
		)
		# (channels, audio_samples)

		audio = torch.mean(audio, dim=0)
		# shape: (audio_samples,)

		audio = torchaudio.functional.resample(
			waveform=audio, 
			orig_freq=self.maestro_sr, 
			new_freq=self.sample_rate
		)
		# shape: (audio_samples,)

		return audio

	'''
	def load_targets(self, midi_path, segment_start_time):

		midi_data = pretty_midi.PrettyMIDI(str(midi_path))

		assert len(midi_data.instruments) == 1

		notes = midi_data.instruments[0].notes

		# Load active notes inside the segment.
		active_notes = []
		segment_end_time = segment_start_time + self.segment_seconds

		for i in range(len(notes)):
			if segment_start_time <= notes[i].start < segment_end_time:
				active_notes.append(notes[i])

		# Covert notes information to words.
		frames_roll = np.zeros((self.segment_frames, self.pitches_num))
		onsets_roll = np.zeros((self.segment_frames, self.pitches_num))

		words = ["<sos>"]

		for note in active_notes:

			onset_time = note.start - segment_start_time
			offset_time = note.end - segment_start_time
			pitch = note.pitch
			velocity = note.velocity

			words.append("<time>={}".format(onset_time))
			words.append("<pitch>={}".format(pitch))
			words.append("<velocity>={}".format(velocity))

			onset_index = int(np.round(onset_time * self.fps))
			offset_index = int(np.round(offset_time * self.fps))
			frames_roll[onset_index : min(self.segment_frames, offset_index + 1), pitch] = 1
			onsets_roll[onset_index, pitch] = 1

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
		}

		return targets_dict
	'''

	def load_targets(self, midi_path, segment_start_time):

		midi_data = pretty_midi.PrettyMIDI(str(midi_path))

		assert len(midi_data.instruments) == 1

		notes = midi_data.instruments[0].notes

		# Load active notes inside the segment.
		active_notes = []
		segment_end_time = segment_start_time + self.segment_seconds

		for i in range(len(notes)):
			if segment_start_time <= notes[i].start < segment_end_time:
				active_notes.append(notes[i])

		####
		control_changes = midi_data.instruments[0].control_changes

		active_pedals = []

		for i in range(len(control_changes)):

			if control_changes[i].number == 64:

				if segment_start_time <= control_changes[i].time < segment_end_time:

					if len(active_pedals) == 0:
						if control_changes[i].value >= 64:
							j = 1
							while i - j >= 0 and control_changes[i - j].value >= 64:
								active_pedals.insert(0, control_changes[i])

					active_pedals.append(control_changes[i])

				if i >= 1 and control_changes[i - 1].time < segment_end_time and control_changes[i].time > segment_end_time:

					if control_changes[i - 1] >= 64:

						j = 0
						while i + j < len(control_changes) and control_changes[i + j].value >= 64:
							active_pedals.append(control_changes[i])

		from IPython import embed; embed(using=False); os._exit(0)

		onset_time = None
		offset_time = None

		for i in range(len(active_pedals_events)):

			if active_pedals_events[i].value >= 64 and onset_time is None:
				onset_time = active_pedals_events[i].time

			if active_pedals_events[i].value < 64 and onset_time is not None and offset_time is None:
				offset_time = active_pedals_events[i].time
				pair = (onset_time, offset_time)
				onset_time = None
				offset_time = None
				pairs.append(pair)

		from IPython import embed; embed(using=False); os._exit(0)

		# Covert notes information to words.
		frames_roll = np.zeros((self.segment_frames, self.pitches_num))
		onsets_roll = np.zeros((self.segment_frames, self.pitches_num))

		words = ["<sos>"]

		for note in active_notes:

			onset_time = note.start - segment_start_time
			offset_time = note.end - segment_start_time
			pitch = note.pitch
			velocity = note.velocity

			words.append("<time>={}".format(onset_time))
			words.append("<pitch>={}".format(pitch))
			words.append("<velocity>={}".format(velocity))

			onset_frame_index = int(np.round(onset_time * self.fps))
			offset_frame_index = int(np.round(offset_time * self.fps))
			frames_roll[onset_frame_index : min(self.segment_frames, offset_frame_index + 1), pitch] = 1
			onsets_roll[onset_frame_index, pitch] = 1

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
		}

		return targets_dict