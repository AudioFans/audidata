from __future__ import annotations
import math
import librosa
import numpy as np
from pretty_midi import Note

from audidata.tokenizers.base import BaseTokenizer, DictTokenizer


class PianoRoll:
    r"""Convert the MIDI note and pedal events a full song into piano rolls of 
    a short clip. The rolls include frame roll, onset roll, offset roll, and
    velocity roll.
    """

    def __init__(
        self, 
        fps: int = 100, 
        pitches_num: int = 128, 
        soft_target: bool = False
    ):
        self.fps = fps
        self.pitches_num = pitches_num
        self.soft_target = soft_target

    def __call__(self, data: dict) -> dict:
        r"""Convert data dict to piano rolls."""

        notes = data["note"]
        pedals = data["pedal"]
        start_time = data["start_time"]
        clip_duration = data["clip_duration"]

        clip_frames = round(self.fps * clip_duration) + 1

        # Rolls
        frame_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")
        onset_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")
        offset_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")
        velocity_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")

        if self.soft_target:
            soft_onset_roll = np.zeros((clip_frames, self.classes_num), dtype="float32")
            soft_offset_roll = np.zeros((clip_frames, self.classes_num), dtype="float32")

        clip_notes = []

        # Go through all notes
        for note in notes:

            onset_time = note.start - start_time
            offset_time = note.end - start_time
            pitch = note.pitch
            velocity = note.velocity

            if offset_time < 0:
                continue

            elif clip_duration < onset_time < math.inf:
                continue

            if offset_time == onset_time:
                offset_time = onset_time + (1. / fps)

            clip_note = Note(
                pitch=pitch, 
                start=onset_time, 
                end=offset_time, 
                velocity=velocity
            )
            clip_notes.append(clip_note)

            if offset_time < onset_time:
                raise "offset should not be smaller than onset!"

            # Update rolls
            elif onset_time < 0 and 0 <= offset_time <= clip_duration:

                offset_idx = round(offset_time * self.fps)
                offset_roll[offset_idx, pitch] = 1
                frame_roll[0 : offset_idx + 1, pitch] = 1

                if self.soft_target:
                    pass
                    # TODO but not necessary
                    # tmp = np.zeros(clip_frames)
                    # tmp[offset_idx] = 1
                    # delayed_frames = (offset_time * fps) % 1
                    # tmp = fractional_delay(tmp, delayed_frames)
                    # soft_offset_roll[:, pitch] += tmp
                    # from IPython import embed; embed(using=False); os._exit(0)

            elif onset_time < 0 and clip_duration < offset_time < math.inf:

                frame_roll[:, pitch] = 1

            elif 0 <= onset_time <= clip_duration and 0 <= offset_time <= clip_duration:

                onset_idx = round(onset_time * self.fps)
                offset_idx = round(offset_time * self.fps)
                onset_roll[onset_idx, pitch] = 1
                velocity_roll[onset_idx, pitch] = velocity / 128.0
                offset_roll[offset_idx, pitch] = 1
                frame_roll[onset_idx : offset_idx + 1, pitch] = 1

            elif 0 <= onset_time <= clip_duration and clip_duration < offset_time < math.inf:

                onset_idx = round(onset_time * self.fps)
                onset_roll[onset_idx, pitch] = 1
                velocity_roll[onset_idx, pitch] = velocity / 128.0
                frame_roll[onset_idx : , pitch] = 1

            else:
                raise NotImplementedError

        # Sort notes
        clip_notes.sort(key=lambda note: (note.start, note.pitch, note.end, note.velocity))

        data.update({
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "frame_roll": frame_roll,
            "velocity_roll": velocity_roll,
            "clip_note": clip_notes
        })

        return data

class MultiTrackPianoRoll:
    r""" Will return a list of piano rolls for each track in the midi file.
    """
    def __init__(
        self,
        fps: int = 100,
        pitches_num: int = 128,
        soft_target: bool = False
    ):
        self.fps = fps
        self.pitches_num = pitches_num
        self.soft_target = soft_target

    def __call__(self, data: dict) -> dict:
        tracks = data["tracks"]
        start_time = data["start_time"]
        clip_duration = data["clip_duration"]

        clip_frames = round(self.fps * clip_duration) + 1

        # Generate rolls for each track
        track_rolls = []
        for track in tracks:
            track_roll = self.create_track_rolls(track, start_time, clip_duration, clip_frames)
            track_rolls.append(track_roll)

        data["track_rolls"] = track_rolls
        return data

    def create_track_rolls(self, track, start_time, clip_duration, clip_frames):
        notes = track["note"]
        
        frame_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")
        onset_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")
        offset_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")
        velocity_roll = np.zeros((clip_frames, self.pitches_num), dtype="float32")

        clip_notes = []

        for note in notes:
            onset_time = note.start - start_time
            offset_time = note.end - start_time
            pitch = note.pitch
            velocity = note.velocity

            if offset_time < 0 or onset_time >= clip_duration:
                continue

            if offset_time == onset_time:
                offset_time = onset_time + (1. / self.fps)

            clip_note = Note(
                pitch=pitch,
                start=max(0, onset_time),
                end=min(clip_duration, offset_time),
                velocity=velocity
            )
            clip_notes.append(clip_note)

            onset_idx = max(0, round(onset_time * self.fps))
            offset_idx = min(clip_frames - 1, round(offset_time * self.fps))

            if 0 <= onset_time < clip_duration:
                onset_roll[onset_idx, pitch] = 1
                velocity_roll[onset_idx, pitch] = velocity / 128.0

            if 0 <= offset_time <= clip_duration:
                offset_roll[offset_idx, pitch] = 1

            frame_roll[max(0, onset_idx):offset_idx + 1, pitch] = 1

        return {
            "frame_roll": frame_roll,
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "velocity_roll": velocity_roll,
            "clip_notes": clip_notes,
            "instrument": track.get("inst_class", "Unknown"),
            "is_drum": track.get("is_drum", False)
        }


class Note2Token:
    r"""Target transform. Transform midi notes to tokens. Users may define their
    own target transforms.
    """

    def __init__(self, 
        tokenizer: BaseTokenizer, 
        max_tokens: int,
    ):
        
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __call__(self, data: dict) -> list[int]:
        
        notes = data["clip_note"]
        clip_duration = data["clip_duration"]

        words = ["<sos>"]

        for note in notes:

            onset_time = note.start
            offset_time = note.end
            pitch = note.pitch
            velocity = note.velocity
            is_drum = note.is_drum

            if 0 <= onset_time <= clip_duration:

                words.append("name=note_on")
                words.append("time={}".format(onset_time))
                if not is_drum:
                    words.append("pitch={}".format(pitch))
                else:
                    words.append("drum_pitch={}".format(pitch))
                words.append("velocity={}".format(velocity))
                
            if 0 <= offset_time <= clip_duration:
                if not is_drum: # no note_off for drums
                    words.append("name=note_off")
                    words.append("time={}".format(offset_time))
                    words.append("pitch={}".format(pitch))

        words.append("<eos>")

        # Words to tokens
        tokens = np.array([self.tokenizer.stoi(w) for w in words])
        tokens_num = len(tokens)

        # Masks
        masks = np.ones_like(tokens)

        tokens = librosa.util.fix_length(data=tokens, size=self.max_tokens)
        masks = librosa.util.fix_length(data=masks, size=self.max_tokens)

        data["word"] = words
        data["token"] = tokens
        data["mask"] = masks
        data["tokens_num"] = tokens_num

        return data
    
class MultiTrackNote2Token:
    def __init__(self, 
        tokenizer: BaseTokenizer, 
        max_tokens: int
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.class_to_midi_program_mapping = { #TODO: move this out?
            "Acoustic Piano": 0,
            "Electric Piano": 4,
            "Chromatic Percussion": 8,
            "Organ": 16,
            "Acoustic Guitar": 24,
            "Clean Electric Guitar": 26,
            "Distorted Electric Guitar": 29,
            "Acoustic Bass": 32,
            "Electric Bass": 33,
            "Violin": 40,
            "Viola": 41,
            "Cello": 42,
            "Contrabass": 43,
            "Orchestral Harp": 46,
            "Timpani": 47,
            "String Ensemble": 48,
            "Synth Strings": 50,
            "Choir and Voice": 52,
            "Orchestral Hit": 55,
            "Trumpet": 56,
            "Trombone": 57,
            "Tuba": 58,
            "French Horn": 60,
            "Brass Section": 61,
            "Soprano/Alto Sax": 64,
            "Tenor Sax": 66,
            "Baritone Sax": 67,
            "Oboe": 68,
            "English Horn": 69,
            "Bassoon": 70,
            "Clarinet": 71,
            "Pipe": 73,
            "Synth Lead": 80,
            "Synth Pad": 88
        }
        
    def get_inst_program_token(self, inst_class: str) -> int:
        return self.class_to_midi_program_mapping.get(inst_class, 0)

    def __call__(self, data: dict) -> dict:
        tracks = data["tracks"]
        clip_start_time = data["start_time"]
        clip_duration = data["clip_duration"]

        all_note_activities = []
        for track_idx, track in enumerate(tracks):
            track_data = {
                "clip_note": track["note"],
                "clip_duration": clip_duration + clip_start_time
            }
            
            is_drum = track.get("is_drum", False)

            for note in track_data["clip_note"]:
                onset = note.start
                offset = note.end
                pitch = note.pitch
                velocity = note.velocity
                
                if 0 <= onset <= clip_duration:
                    all_note_activities.append({
                        "time": onset,
                        "pitch": pitch,
                        "velocity": velocity,
                        "program": self.get_inst_program_token(track["inst_class"]),
                        "activity": "note_on",
                        "is_drum": is_drum
                    })
                
                if 0 <= offset <= clip_duration:
                    if not is_drum: # no note_off for drums
                        all_note_activities.append({
                            "time": offset,
                            "pitch": pitch,
                            "velocity": velocity,
                            "activity": "note_off"
                        })
                    
        all_note_activities.sort(key=lambda x: (x["time"], x["pitch"]))
        # TODO: need to discuss. Time should be sorted of course, but pitch? This assumes lower pitches should be predicted first. Make sense because bass notes hint the chord progression...?
        
        all_words = ["<sos>"]
        for note_activity in all_note_activities:
            time = note_activity["time"]
            pitch = note_activity["pitch"]
            velocity = note_activity.get("velocity", 0)
            program = note_activity.get("program", 0)
            activity = note_activity["activity"]
            
            all_words.append(f"name={activity}")
            if activity == "note_on":
                all_words.append(f"program={program}")
                if not note_activity["is_drum"]:
                    all_words.append(f"pitch={pitch}")
                else:
                    all_words.append(f"drum_pitch={pitch}")
            all_words.append(f"time={time}")
            all_words.append(f"velocity={velocity}")

        all_words.append("<eos>")
        
        # For debugging
        # with open("slakh2100_tokenized.txt", "w") as f:
        #     for w in all_words:
        #         f.write(w + "\n")
        
        # print("Here")
        
        tokens = np.array([self.tokenizer.stoi(w) for w in all_words])
        tokens_num = len(tokens)

        masks = np.ones_like(tokens)

        tokens = librosa.util.fix_length(data=tokens, size=self.max_tokens)
        masks = librosa.util.fix_length(data=masks, size=self.max_tokens)

        data["word"] = all_words
        data["token"] = tokens
        data["mask"] = masks
        data["tokens_num"] = tokens_num

        return data
    
class Note2DictToken:
    def __init__(
        self, 
        tokenizer: DictTokenizer, 
        max_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        
    def __call__(self, data: dict) -> dict:
        notes = data["clip_note"]
        clip_duration = data["clip_duration"]

        sequence = ["<bot>", "<special:<sos>>", "<eot>"]
        
        for note in notes:
            onset_time = note.start
            offset_time = note.end
            
            if onset_time < 0 and 0 <= offset_time <= clip_duration:
                sequence.extend([
                    "<bot>",
                    "<onset:name=note_sustain>",
                    f"<pitch:pitch={note.pitch}>",
                    f"<offset:time={offset_time}>",
                    f"<velocity:velocity={note.velocity}>",
                    "<eot>"
                ])
            elif 0 <= onset_time <= offset_time <= clip_duration:
                sequence.extend([
                    "<bot>",
                    f"<onset:time={onset_time}>",
                    f"<pitch:pitch={note.pitch}>",
                    f"<offset:time={offset_time}>",
                    f"<velocity:velocity={note.velocity}>",
                    "<eot>"
                ])
            elif 0 <= onset_time <= clip_duration < offset_time:
                sequence.extend([
                    "<bot>",
                    f"<onset:time={onset_time}>",
                    f"<pitch:pitch={note.pitch}>",
                    "<offset:name=note_sustain>",
                    f"<velocity:velocity={note.velocity}>",
                    "<eot>"
                ])
            elif onset_time < 0 and clip_duration < offset_time:
                sequence.extend([
                    "<bot>",
                    "<onset:name=note_sustain>",
                    f"<pitch:pitch={note.pitch}>",
                    "<offset:name=note_sustain>",
                    f"<velocity:velocity={note.velocity}>",
                    "<eot>"
                ])
        
        sequence.extend(["<bot>", "<special:<eos>>", "<eot>"])

        # Tokenize the sequence
        tokens = self.tokenizer.tokenize(sequence)

        # Ensure the token list doesn't exceed max_tokens
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        else:
            # pad with <pad> tokens
            pad_token = self.tokenizer.tokenize(["<bot>", "<special:<pad>>", "<eot>"])
            tokens.extend(pad_token * (self.max_tokens - len(tokens)))

        # Create mask
        masks = [1] * len(tokens)
        masks = masks[:self.max_tokens]
        masks.extend([0] * (self.max_tokens - len(masks)))

        data["word"] = sequence
        data["token"] = tokens
        data["mask"] = np.array(masks)
        data["tokens_num"] = len(sequence)

        return data