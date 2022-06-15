#!/usr/bin/python3
from dataclasses import dataclass
from typing import List
from librosa import note_to_hz
import numpy as np

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def convert_hz_to_note(notes_arr):
    split_notes_arr = np.array([n.split(',') for n in notes_arr.ravel()])
    return note_to_hz(split_notes_arr)

@dataclass
class Scale:

    root: str = 'C'
    name: str = 'major'    
    
    # denotes which indices of the scale are true
    modes = {
        'major':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'dorian':     [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'phrygian':   [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'lydian':     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        'mixolydian': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'minor':      [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'pentatonic': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0], # min pent
        'locrian':    [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        'whole':      [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'chromatic':  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }

    def _scale(self) -> np.array:
        '''
        Returns the True indices for a given mode, considering 12 half-steps.

        e.g. 'major' returns:
        [ 0  2  4  5  7  9 11]
        '''
        return np.nonzero(self.modes[self.name])[0]

    def _rearrange_notes(self, note: str) -> List:
        '''
        Rearranges the above NOTES list so that the 0th index is the supplied `note`.
        '''
        root_index = np.nonzero(np.isin(NOTES, note))[0][0]
        rearranged_notes = NOTES[root_index:] + NOTES[:root_index]
        return rearranged_notes

    @property
    def notes(self) -> List:
        '''
        Returns the note names of a given Scale object.
        '''
        return [self._rearrange_notes(self.root)[i] for i in self._scale()]

    @property
    def hz(self) -> np.array:
        '''
        Returns the values in hz for a Scale object.
        '''
        hz_list = note_to_hz([f'{n}{i}' for n in self.notes for i in range(1,10)])
        return np.sort(hz_list)