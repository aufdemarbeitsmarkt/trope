import librosa
import numpy as np

def duration_to_time(durations, bpm):
    # TODO: change this to an array so I can broadcast-multiple by beat_seconds rather than loop 
    # TODO: change the name of this function / method; this name is awkward and doesn't speecifically convey its intent
    '''
    Returns the number of seconds for the desired

    1 = one beat, e.g. quarter note; 0.5 = half a beat, e.g. eighth note; 0.25 a quarter of a beat, e.g. sixteenth note... and so on.
    '''
    beat_seconds = 60 / bpm
    return np.asarray(durations) * beat_seconds


class Rhythm:
    
    default_bpm = 120

    def __init__(self, bpm=None):
        if bpm is None:
            bpm = self.default_bpm
        self.bpm = bpm