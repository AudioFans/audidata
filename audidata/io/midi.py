import copy
import math
import time
from typing import Dict, List, Tuple

import numpy as np
from pretty_midi import ControlChange, Note, PrettyMIDI


class Pedal:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def __repr__(self):
        return "Pedal(start={:.4f}, end={:.4f})".format(self.start, self.end)


def read_single_track_midi(
    midi_path: str, 
    extend_pedal: bool
) -> Tuple[List[Note], List[Pedal]]:
    r"""Read note and pedals from a single-track MIDI file.

    Returns:
        notes: List[Note]
        pedals: List[Pedal]
    """

    midi_data = PrettyMIDI(str(midi_path))
    
    assert len(midi_data.instruments) == 1

    notes = midi_data.instruments[0].notes
    control_changes = midi_data.instruments[0].control_changes

    # Get pedals
    pedals = get_pedals(control_changes)
    
    # Extend note offsets by pedal information
    if extend_pedal:
        notes = extend_offset_by_pedal(notes, pedals)

    return notes, pedals


def get_pedals(control_changes: List[ControlChange]) -> List[Pedal]:
    r"""Get list of pedal events."""

    onset = None
    offset = None
    pedals = []

    control_changes.sort(key=lambda cc: cc.time)

    for cc in control_changes:

        if cc.number == 64:  # Sustain pedal

            if cc.value >= 64 and onset is None:
                onset = cc.time

            elif cc.value < 64 and onset is not None:
                offset = cc.time
                pedal = Pedal(start=onset, end=offset)
                pedals.append(pedal)
                onset = None
                offset = None

    if onset is not None and offset is None:
        offset = control_changes[-1].time
        pedal = Pedal(start=onset, end=offset)
        pedals.append(pedal)

    return pedals


def extend_offset_by_pedal(
    notes: List[Note], 
    pedals: List[Pedal]
) -> List[Note]:
    r"""Extend the note offset times to pedal offset times.

    Returns:
        new_notes: List[Note]
    """

    notes = copy.deepcopy(notes)
    pedals = copy.deepcopy(pedals)
    pitches_num = 128

    notes.sort(key=lambda note: note.end)

    notes_dict = {pitch: [] for pitch in range(pitches_num)}
    
    while len(pedals) > 0 and len(notes) > 0:

        pedal = pedals[0]  # Get the first pedal.

        while notes:

            note = notes[0]  # Get the first note.

            pitch = note.pitch

            if 0 <= note.end < pedal.start:
                notes_dict[pitch].append(note)
                notes.pop(0)

            elif pedal.start <= note.end < pedal.end:

                new_note = Note(
                    pitch=pitch, 
                    start=note.start, 
                    end=pedal.end, 
                    velocity=note.velocity
                )

                notes_dict[pitch].append(new_note)
                notes.pop(0)

            elif pedal.end <= note.end < math.inf:
                pedals.pop(0)
                break

            else:
                raise NotImplementedError 

    # Append remaining notes to dict
    for note in notes:
        notes_dict[note.pitch].append(note)

    # Note dict to note list
    new_notes = []
    for pitch in notes_dict.keys():
        new_notes.extend(notes_dict[pitch])
    
    new_notes.sort(key=lambda note: note.start)

    return new_notes


def notes_to_data(
    notes: List[Note], 
    clip_frames: int, 
    classes_num: int, 
    clip_start_time: float, 
    clip_duration: float, 
    fps: int
) -> Dict:
    r"""Transform the whole piece notes to onset roll, offset roll, frame roll, 
    velocity roll, and notes within a short cilp.

    Returns:
        data: Dict
    """

    # Rolls
    frame_roll = np.zeros((clip_frames, classes_num), dtype="float32")
    onset_roll = np.zeros((clip_frames, classes_num), dtype="float32")
    offset_roll = np.zeros((clip_frames, classes_num), dtype="float32")
    velocity_roll = np.zeros((clip_frames, classes_num), dtype="float32")

    soft_target = True

    if soft_target:
        soft_onset_roll = np.zeros((clip_frames, classes_num), dtype="float32")
        soft_offset_roll = np.zeros((clip_frames, classes_num), dtype="float32")

    clip_notes = []

    # Go through all notes
    for note in notes:

        onset_time = note.start - clip_start_time
        offset_time = note.end - clip_start_time
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

            offset_idx = round(offset_time * fps)
            offset_roll[offset_idx, pitch] = 1
            frame_roll[0 : offset_idx + 1, pitch] = 1

            if soft_target:
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

            onset_idx = round(onset_time * fps)
            offset_idx = round(offset_time * fps)
            onset_roll[onset_idx, pitch] = 1
            velocity_roll[onset_idx, pitch] = velocity / 128.0
            offset_roll[offset_idx, pitch] = 1
            frame_roll[onset_idx : offset_idx + 1, pitch] = 1

        elif 0 <= onset_time <= clip_duration and clip_duration < offset_time < math.inf:

            onset_idx = round(onset_time * fps)
            onset_roll[onset_idx, pitch] = 1
            velocity_roll[onset_idx, pitch] = velocity / 128.0
            frame_roll[onset_idx : , pitch] = 1

        else:
            raise NotImplementedError

    # Sort notes
    clip_notes.sort(key=lambda note: (note.start, note.pitch, note.end, note.velocity))

    data = {
        "onset_roll": onset_roll,
        "offset_roll": offset_roll,
        "frame_roll": frame_roll,
        "velocity_roll": velocity_roll,
        "note": clip_notes
    }

    return data
