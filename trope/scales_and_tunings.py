#!/usr/bin/python3
from dataclasses import dataclass
from typing import List
from librosa import note_to_hz
import numpy as np

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# TODO: incorporate `NOTES_SHARP` and `NOTES_FLAT` to ensure the correct `notes` property is returned. This is largely cosmetic, so not terribly high priority. 
NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

CHORD_ORDER = ['major', 'minor', 'minor', 'major', 'major', 'minor', 'diminished'] # "default"

def convert_hz_to_note(notes_arr):
    split_notes_arr = np.asarray([n.split(', ') for n in notes_arr.ravel()])
    return note_to_hz(split_notes_arr)

def rotate(lst, idx):
    # a list argument and the index by which to rotate 
    rotated_lst = lst[idx:] + lst[:idx]
    return rotated_lst


@dataclass
class Scale:

    root: str = 'C'
    name: str = 'major'    
    
    # denotes which indices of the scale are true
    names = {
        'major':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'dorian':     [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'phrygian':   [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'lydian':     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        'mixolydian': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'minor':      [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'locrian':    [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        'pentatonic': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0], # min pent
        'whole':      [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'chromatic':  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }

    def _scale(self):
        '''
        Returns the True indices for a given mode, considering 12 half-steps.

        e.g. 'major' returns:
        [0  2  4  5  7  9 11]
        '''
        return np.nonzero(self.names[self.name])[0]

    def _rearrange_notes(self, note):
        '''
        Rearranges the above NOTES list so that the 0th index is the supplied `note`.
        '''
        root_idx = NOTES.index(note)
        return rotate(NOTES, root_idx)

    @property
    def notes(self):
        '''
        Returns the note names of a given Scale object.
        '''
        return [self._rearrange_notes(self.root)[i] for i in self._scale()]

    @property
    def hz(self):
        '''
        Returns the values in hz for a Scale object.
        '''
        # TODO: calling note_to_hz() twice in this method is redundant; quick and dirty way to ensure sorting is correct, but should be fixed at some point
        notes_list = sorted([f'{n}{i}' for n in self.notes for i in range(1,9)], key=lambda x:note_to_hz(x))
        hz_arr = note_to_hz(
            notes_list[notes_list.index(f'{self.root}1'):]
            )
        return hz_arr

    def _get_chord_order(self):
        # returns the order of chord types for the current scale / mode 
        mode_idx = list(self.names).index(self.name)
        return rotate(CHORD_ORDER, mode_idx)

    @property
    def chords(self):
        chord_dict = {
            'notes': {},
            'hz': {},
            'triads': {}
        }

        for note, chord_type in zip(self.notes, self._get_chord_order()):
            # TODO: need to accommodate for cases where this goes "past" locrian in the `names` dict, e.g. pentatonic, whole, and chromatic
            C = Chord(note, chord_type)
            freqs = C.hz

            chord_abbr = note + chord_type[:3]
            chord_dict['notes'][chord_abbr] = C.notes
            chord_dict['hz'][chord_abbr] = freqs

            # TODO: add mapping for equal-temperament values
            third_value = 1.259921 if chord_type == 'major' else 1.189207
            fifth_value = 1.587401 if chord_type == 'augmented' else 1.498307

            root = freqs[
                np.isclose(
                    freqs % freqs[0], 
                    0., 
                    atol=1e-2
                    )
                    ]
            third = freqs[
                np.isclose(
                    (freqs / third_value) % freqs[0], 
                    0., 
                    atol=1e-2
                    )
                    ]

            fifth = freqs[
                np.isclose(
                    (freqs / fifth_value) % freqs[0], 
                    0., 
                    atol=1e-2
                    )
                    ]
            chord_dict['triads'][chord_abbr] = {
                'root': root,
                'third': third,
                'fifth': fifth
                }

        return chord_dict


@dataclass
class Chord(Scale):

    names = {
        'major':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'minor':      [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'diminished': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        'augmented':  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    }