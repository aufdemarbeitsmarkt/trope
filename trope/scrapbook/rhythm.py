#!/usr/bin/python3
import librosa
import numpy as np
import random
import scipy

### old old old drafts of rhythm-centric functions

bpm = 120 # global bpm - reconsider this, but this is fine for now

def duration_to_time(x=1, bpm=bpm):
    # TODO: change this to an array so I can broadcast-multiple by beat_seconds rather than loop 
    '''
    Returns the number of seconds for the desired

    1 = one beat, e.g. quarter note; 0.5 = half a beat, e.g. eighth note; 0.25 a quarter of a beat, e.g. sixteenth note... and so on.
    '''
    beat_seconds = 60 / bpm
    durations = []
    if isinstance(x, (float, int)):
        return beat_seconds * x
    if isinstance(x, (list, np.ndarray)): # solution _might_ not be the most comprehensive, see: https://stackoverflow.com/a/12569453
        for n in x:
            durations.append(beat_seconds * n)
        return durations

def get_onsets():
    '''
    Returns a list of onset times, scaled to the global bpm. For further manipulation or straight-up usage.
    '''
    y,sr = inputs.load_audio()
    tempo = librosa.beat.tempo(y,sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onsets_diff = np.diff(onsets)
    onsets_diff_adj = onsets_diff * (tempo / bpm)
    return np.around(onsets_diff_adj, decimals=3) # human hearing threshold is about 0.02 - 0.03 seconds (or 20 - 30 milliseconds); I think this gives appropriate resolution for now

#### Draft rhythm functions
def tile(x, n=4):
    '''
    Returns an array, of tiles of the same length x by n times
    '''
    # return ([x for i in range(n)])
    return np.tile(x, n)

def repeat(x, n=4):
    '''
    Returns an array, of tiles of the same length x by n times
    '''
    return np.repeat(x, n)

def dur_repeat(n, length=4):
    '''
    Returns an array that sums to the designated length with a number n beats.
    '''
    return np.tile(length / n, n)
