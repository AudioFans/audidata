import torch
from pathlib import Path
import pandas as pd
import random
import librosa
import torchaudio
import pretty_midi
import numpy as np
import os

from .tokenizers import Tokenizer3
from .constants import *

class MyNote(pretty_midi.Note):
    def __init__(self, velocity, pitch, start, end, instrument):
        super().__init__(velocity, pitch, start, end)
        self.instrument = instrument

class MusicNetEMDataset:
	# reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
	def __init__(
		self, 
		root: str = None, 
		split: str = "train",
		segment_seconds: float = SEGMENT_SECONDS,
		tokenizer=Tokenizer3(),
	):
	
		self.root = root
		self.split = split
		if self.split == "train":
			self.wav_dir = Path(self.root, "train_data")
			self.midi_dir = Path(self.root, "train_labels")
		elif self.split == "test":
			self.wav_dir = Path(self.root, "test_data")
			self.midi_dir = Path(self.root, "test_labels")
		self.segment_seconds = segment_seconds
		self.tokenizer = tokenizer

		self.sample_rate = SAMPLE_RATE
		self.fps = TARGET_FRAMES_PER_SECONDS
		self.pitches_num = 128
		self.instrument_num = 128
		self.segment_frames = int(self.segment_seconds * self.fps) + 1

		self.meta_csv = Path(self.root, "musicnet_metadata.csv")

		self.load_meta()
				
	def load_meta(self):
		self.audio_filenames = sorted(os.listdir(self.wav_dir))
		self.midi_filenames = sorted(os.listdir(self.midi_dir))
		self.audios_num = len(self.audio_filenames)
		df = pd.read_csv(self.meta_csv, sep=',')
		file_ids = [int(filename.split('.')[0]) for filename in self.audio_filenames]
		# filter rows based on ids and then extract corresponding seconds values
		self.durations = df[df['id'].isin(file_ids)]['seconds'].tolist()


	def __getitem__(self, index):

		audio_index = random.randint(0, self.audios_num - 1)

		audio_path = Path(self.wav_dir, self.audio_filenames[audio_index])
		midi_path = Path(self.midi_dir, self.midi_filenames[audio_index])
		duration = self.durations[audio_index]

		segment_start_time = random.uniform(0, duration - self.segment_seconds)

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

		musicnet_sr = librosa.get_samplerate(audio_path)

		segment_start_sample = int(segment_start_time * musicnet_sr)
		segment_samples = int(self.segment_seconds * musicnet_sr)

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
			orig_freq=musicnet_sr,
			new_freq=self.sample_rate
		)
		# shape: (audio_samples,)

		return audio

	def load_targets(self, midi_path, segment_start_time):

		midi_data = pretty_midi.PrettyMIDI(str(midi_path))

		# Load active notes inside the segment.
		active_notes = []
		segment_end_time = segment_start_time + self.segment_seconds
		for j in range(len(midi_data.instruments)):
			notes = midi_data.instruments[j].notes
			for i in range(len(notes)):
				if segment_start_time <= notes[i].start < segment_end_time:
					active_notes.append(MyNote(notes[i].velocity, notes[i].pitch, notes[i].start, notes[i].end, instrument=midi_data.instruments[j].program))

		# Process the start and end times of notes
		for i in range(len(active_notes)):
			active_notes[i].start = int(np.round((active_notes[i].start - segment_start_time) * self.fps))
			active_notes[i].end = int(np.round((active_notes[i].end - segment_start_time) * self.fps))
			active_notes[i].end = int(min(self.segment_frames, active_notes[i].end))

		# sorted according to the start time, and if the start time is the same, they are sorted according to the pitch
		active_notes = sorted(active_notes, key=lambda note: (note.start, note.pitch, note.instrument))

		frames_roll = np.zeros((self.segment_frames, self.pitches_num, self.instrument_num))
		onsets_roll = np.zeros((self.segment_frames, self.pitches_num, self.instrument_num))

		# Covert notes information to words.
		words = ["<sos>"]

		for note in active_notes:
			onset_time = note.start
			offset_time = note.end
			pitch = note.pitch
			velocity = note.velocity
			instrument = note.instrument

			words.append("<time>={}".format(onset_time))
			words.append("<pitch>={}".format(pitch))
			words.append("<velocity>={}".format(velocity))
			words.append("<inst>={}".format(instrument))

			frames_roll[onset_time:offset_time, pitch, instrument] = 1
			onsets_roll[onset_time, pitch, instrument] = 1

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