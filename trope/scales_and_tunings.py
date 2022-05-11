#!/usr/bin/python3
from typing import Union, List
from librosa import hz_to_note, note_to_hz
import librosa
import numpy as np

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # rename NOTES_SHARP
# NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

class Scale:

    mode_names = {
        # denotes which indices of the scale are true
        'major':         [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'dorian':        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'phrygian':      [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'lydian':        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        'mixolydian':    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'minor':         [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'pentatonic':    [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0], # minor pent
        'locrian':       [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        'whole':         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'chromatic':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }

    def __init__(self, name: str, root: str ='C') -> None:
        self.name = name
        self.root = root

    def scale(self) -> np.array:
        '''
        Returns the True indices for a given mode, considering 12 half-steps.

        e.g. mode_names['major'] returns:
        [ 0  2  4  5  7  9 11]
        '''
        return np.nonzero(self.mode_names[self.name])[0]

    def rearrange_notes(self, note: str) -> List:
        '''
        Rearranges the above NOTES list so that the 0th index is the supplied `note`.
        '''
        root_index = np.nonzero(np.isin(NOTES, note))[0][0]
        rearranged_notes = NOTES[root_index:] + NOTES[:root_index]
        return rearranged_notes

    def notes(self, root_note: str = None) -> List:
        '''
        Returns the note names of a given Scale object.
        '''
        root_note = self.root
        return [self.rearrange_notes(root_note)[i] for i in self.scale()]

    def hz(self) -> np.array:
        '''
        Returns the values in hz for a Scale object.
        '''
        hz_list = note_to_hz([n + str(i) for n in self.notes() for i in range(1,10)])
        return np.sort(hz_list)


class Chord(Scale):

    def __init__(self, name: str, root: str, chord_name: str = None, chord_root: str = None) -> None:
        # add an argument for the chord type + the scale / mode name
        # that way I can define the key and the root of the chord.
        # I should be able to ideally specify a chord in these two ways:
        #  - G major key, D chord (can override, but a D by itself will play the D in the key, whether its major, minor, etc)
        #    - I want to also make arguments where I can give the index of the chord within the given Scale object, e.g. "G major, V" where V is a D chord, of course
        #  - D major chord by itself
        super().__init__(name, root)
        self.chord_name = chord_name
        self.chord_root = chord_root
        if chord_root is None:
            self.chord_root = self.root

    def get_chord_root_index(self):
        # get the chord root index and use this to shift scale(); see note there
        pass

    def scale(self) -> np.array:
        '''
        Returns the True indices for a triad, considering 12 half-steps.
        '''
        # use rearrange_notes() to move to the correct root note for the given chord; basically get, say, Gmaj's scale but shift it so that the chord root is the 0th index
        return np.nonzero(self.mode_names[self.name])[0][:6:2]

    def chord_progression(self, progression: List[int] = None) -> List:
        pass
