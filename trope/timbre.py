#!/usr/bin/python3
from typing import List, Tuple

import librosa
import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.stats import mode

from performing import Audio


# TODO: for uniformity's sake, I should probably make this functionally similar to Envelope; the user could use a similar `from_audio()`-type method
# behind the scenes, I might be able to make this a bit cleaner for Synthesis to use the Timbre class; right now, it's a bit messy
# also, the scoping is all outta whack; I should remove the nested functions 

class Timbre:
    '''
    Heavily influenced by: https://mapio.github.io/sinuous-violin/
    '''
    def __init__(
        self, 
        audio
        ):
        self.audio = audio

        self.audio_to_process = self.audio.audio
        self.sample_rate = self.audio.sample_rate 

        self.timbre = self._get_timbre()

    # TODO: need to implement this downsampling
    def _downsample_audio(self):
        # downsample audio for speed; has the added benefit of excluding relatively high frequencies
        return librosa.resample(
            self.audio.audio,
            orig_sr=self.audio.sample_rate,
            target_sr=5512
        )

    def _get_timbre(self): 
        '''
        Returns tuple of tuples: the first number provides the harmonic overtone, the second number provides the corresponding amplitude of a the give overtone.

        >> [(1, 0.75), (2, 0.5), (4, 0.5), (8, 0.1)]

        The first index is the fundamental and its corresponding amplitude as scaled by scale_linearly(). Additional entries in this output provide the factor to multiply the fundamental by for each harmonic overtone and, of course, the corresponding amplitude_list as well.
        '''

        # get the spectrum for the real fft, dft sample frequencies, and the absolute values of the spectrum
        N = self.audio_to_process.shape[0]
        spectrum = np.fft.rfft(self.audio_to_process)
        frequencies = np.fft.rfftfreq(N, 1 / self.sample_rate)
        amplitudes = np.abs(spectrum)

        # get the maxima of the amplitudes and the frequencies
        peak_indices = find_peaks_cwt(amplitudes, widths = (60,)) 

        amplitudes_maxima = [np.max(amplitudes[idx - 10:idx+10]) for idx in peak_indices]
        frequencies_maxima = frequencies[np.isin(amplitudes, amplitudes_maxima)]

        def _get_fundamental(audio, frequencies):
            # TODO: need to handle when there is no fundamental found -- this'll return a nan
            pyin_fundamental = mode(librosa.pyin(audio, fmin=20, fmax=3000)[0], nan_policy='omit').mode
            fundamental_index = np.isclose(pyin_fundamental, frequencies, atol=4)
            return frequencies[fundamental_index]

        def _get_overtone_factors(frequencies_list: List) -> List[float]:
            '''
            Returns a list, as described in the docstring for _get_timbre(), where the first index is the fundamental (1.0) and the remaining entries are the factor by which one would multiply the fundamental to get the Hz value of the overtones.
            '''
            fundamental = _get_fundamental(self.audio_to_process, frequencies_maxima)
            if fundamental in frequencies_list:
                overtone_factors = [f / fundamental for f in frequencies_list]
            else:
                overtone_factors = [f / frequencies_list[0] for f in frequencies_list]
            return overtone_factors

        scaled_amplitudes_maxima = amplitudes_maxima
        overtones = _get_overtone_factors(frequencies_maxima)

        return zip(overtones, scaled_amplitudes_maxima)
