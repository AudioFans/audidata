from __future__ import annotations
import copy
import math
import time

import numpy as np
from pretty_midi import ControlChange, Note, PrettyMIDI


class Pedal:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def __repr__(self):
        return "Pedal(start={:.4f}, end={:.4f})".format(self.start, self.end)


def read_midi_beat(midi_path: str):

    midi_data = PrettyMIDI(str(midi_path))

    beats = midi_data.get_beats()
    downbeats = midi_data.get_downbeats()

    return beats, downbeats


def read_single_track_midi(
    midi_path: str, 
    extend_pedal: bool
) -> tuple[list[Note], list[Pedal]]:
    r"""Read note and pedals from a single-track MIDI file.

    Returns:
        notes: list[Note]
        pedals: list[Pedal]
    """

    # Load MIDI is the 10x slower than other IOs
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

def read_multi_track_midi(midi_path: str) -> list[dict]:
    r"""Read notes from a multi-track MIDI file.
    Did not implement extend_pedal here. TODO: future work

    Returns:
        data: list[dict]
    """

    midi_data = PrettyMIDI(str(midi_path))
    
    data = []

    for instrument in midi_data.instruments:

        notes = instrument.notes
        control_changes = instrument.control_changes

        # Get pedals
        pedals = get_pedals(control_changes)

        data.append({
            "notes": notes,
            "pedals": pedals,
            "program": instrument.program,
            "is_drum": instrument.is_drum
        })

    return data


def get_pedals(control_changes: list[ControlChange]) -> list[Pedal]:
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
    notes: list[Note], 
    pedals: list[Pedal]
) -> list[Note]:
    r"""Extend the note offset times to pedal offset times.

    Returns:
        new_notes: list[Note]
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