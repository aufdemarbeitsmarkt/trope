#!/usr/bin/python3
from typing import List, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks_cwt

from performing import Audio


class Timbre:

    '''
    Heavily influenced by: https://mapio.github.io/sinuous-violin/

    _get_timbre() is the primary method for this class.

    Will be adding save() and load() methods.
    '''

    def __init__(self, input_audio, timbre = None):
        if timbre is None:
            self.timbre = self._get_timbre(input_audio)

    def _get_timbre(audio: Audio): # this doesn't need to be an Audio object explictly
        '''
        Returns tuple of tuples: the first number provides the harmonic overtone, the second number provides the corresponding amplitude of a the give overtone.

        >> ((1, 0.75), (2, 0.5), (4, 0.5), (8, 0.1))

        The first index is the fundamental and its corresponding amplitude as scaled by _scale_amplitude_linear(). Additional entries in this output provide the factor to multiply the fundamental by for each harmonic overtone and, of course, the corresponding amplitude_list as well.
        '''
        y, sr = audio.audio, audio.sample_rate # add self.audio to __init__()

        # get the spectrum for the real fft, dft sample frequencies, and the absolute values of the spectrum
        N = y.shape[0]
        spectrum = np.fft.rfft(y)
        frequencies = np.fft.rfftfreq(N, 1 / sr)
        amplitudes = np.abs(spectrum)

        # get the maxima of the amplitudes and the frequencies
        peak_indices = find_peaks_cwt(amplitudes, widths = (60, np.max(amplitudes)))  # may want to finetune the width values at a later time, but this is sufficient for a reasonable reproduction of many sounds

        amplitudes_maxima = list(map(lambda idx: np.max(amplitudes[idx - 10:idx + 10]), peak_indices))
        frequencies_maxima = frequencies[np.isin(amplitudes, amplitudes_maxima)]

        def _scale_amplitude_linear(amplitude_list: List, a: float = 0, b: float = 0.5) -> List[float]:
            '''
            Returns a scaled list of amplitudes based on values for a and b as well as the min and max of the input.
            '''
            low, high = np.min(amplitude_list), np.max(amplitude_list)
            scaled_amplitude = [((b - a) * (x - low)) / (high - low) + a for x in amplitude_list] # this gives a linear scale from a to b
            return scaled_amplitude

        def _get_overtone_factors(frequencies_list: List) -> List[float]:
            '''
            Returns a list, as described in the docstring for _get_timbre(), where the first index is the fundamental (1.0) and the remaining entries are the factor by which one would multiply the fundamental to get the Hz value of the overtones.
            '''
            fundamental = frequencies_list[0]
            overtone_factors = [f / fundamental for f in frequencies_list]
            return overtone_factors

        scaled_amplitudes_maxima = _scale_amplitude_linear(amplitudes_maxima)
        overtones = _get_overtone_factors(frequencies_maxima)

        return zip(overtones, scaled_amplitudes_maxima)



y, sr = librosa.load('/Users/timnilsen/Documents/trope_new/181425__serylis__guitar-chord.wav')

T = Timbre(y)
