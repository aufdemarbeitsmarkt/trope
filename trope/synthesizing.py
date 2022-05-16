#!/usr/bin/python3
from itertools import product
import librosa
from math import ceil
import numpy as np


class Tone:

    default_amplitude = 0.5

    def __init__(
        self,
        frequency,
        duration,
        sample_rate,
        amplitude=None
        ):
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate
        self.duration_in_samples = self._get_duration_in_samples()

        if amplitude is None:
            amplitude = self.default_amplitude
        self.amplitude = amplitude

        self.tone = self._generate_tone()

    def _get_duration_in_samples(self):
        '''
        This method sets a number of instance variables and returns the duration in samples quantized to a zero crossing rate.
        '''
        self.cycle_time_samples = self.sample_rate / self.frequency
        self.zero_crossing_time_samples = self.cycle_time_samples / 2
        self.num_samples = self.sample_rate * self.duration
        self.range_cycles = np.ceil(np.arange(0, int(self.num_samples + self.zero_crossing_time_samples), self.zero_crossing_time_samples))

        diff_second_to_last, diff_last = abs(self.num_samples - self.range_cycles[-2]), abs(self.num_samples - self.range_cycles[-1])

        which_cycle = np.argmin([diff_second_to_last, diff_last])

        return self.range_cycles[-2] if which_cycle == 0 else self.range_cycles[-1]

    def _generate_tone(self):
        each_sample = np.arange(self.duration_in_samples)
        sine = np.sin(2 * np.pi * each_sample * self.frequency / self.sample_rate) * self.amplitude
        return sine


class Synthesis:

    def __init__(
        self,
        input_refrain,
        durations,
        sample_rate,
        timbre=None
        ):
        self.input_refrain = np.asarray(input_refrain)
        self.durations = np.asarray(durations)
        self.sample_rate = sample_rate
        self.timbre = timbre

        self._setup_metadata()

        self.synthesized_output = self.synthesize()

    def _setup_metadata(self):
        self.chord_present = any([isinstance(i, tuple) for i in self.input_refrain]) # True or False whether a chord is present
        self.longest_chord_len = max(len(r) if isinstance(r, tuple) else 1 for r in self.input_refrain)
        self.timbre_present = self.timbre is not None # same as chord_present, re: timbre

        if len(self.input_refrain) == len(self.durations):
            self.setup_refrain = self.input_refrain, self.durations
        else:
            self.setup_refrain =  np.tile(self.input_refrain, len(self.durations)), np.tile(self.durations, len(self.input_refrain))

        self.total_duration = sum(self.setup_refrain[-1]) # duration in seconds; must be created after self.setup_refrain

        self.sample_boundaries = np.cumsum(self.setup_refrain[-1] * self.sample_rate, dtype='int') # the sample boundaries where each note or chord will be set
        self.sample_boundaries = np.insert(self.sample_boundaries[:-1], 0, 0)

        return None

    def _initialize_matrix(self):
        axis0 = self.longest_chord_len * len(self.timbre) if self.timbre_present else self.longest_chord_len
        axis1 = ceil(self.total_duration) * self.sample_rate
        return np.zeros((axis0, axis1))

    # def _generate_tone(self, frequency, duration, amplitude=1):
    #     each_sample = np.arange(ceil(duration * self.sample_rate))
    #     tone = np.sin(2 * np.pi * each_sample * frequency / self.sample_rate) * amplitude
    #     return tone

    def _create_nxn_array(self, frequencies, duration):
        chord = isinstance(frequencies, tuple)
        if chord and self.timbre_present:
            axis0 = len(frequencies) * len(self.timbre)
        elif self.timbre_present:
            axis0 = len(self.timbre)
        else:
            axis0 = len(frequencies)

        nxn_array = np.empty((axis0, ceil(duration * self.sample_rate)))

        if chord and self.timbre_present:
            chord_timbre_prod = product(frequencies, self.timbre)
            for i, (freq, (overtone, amplitude)) in enumerate(chord_timbre_prod):
                nxn_array[i] = Tone(frequency=freq*overtone, duration=duration, amplitude=amplitude).tone

                # self._generate_tone(freq*overtone, duration=duration, amplitude=amplitude)

        elif self.timbre_present:
            for i, (overtone, amplitude) in enumerate(self.timbre):
                tone = Tone(frequency=frequencies*overtone, duration=duration, sample_rate=self.sample_rate, amplitude=amplitude).tone

                # self._generate_tone(frequencies*overtone, duration=duration, amplitude=amplitude)

                nxn_array[i] = tone

        else:
            for i, freq in enumerate(frequencies):
                nxn_array[i] = Tone(frequency=freq, duration=duration, sample_rate=self.sample_rate).tone
                # self._generate_tone(freq, duration=duration)

        return nxn_array

    def synthesize(self, sum_output=True):
        matrix = self._initialize_matrix()
        freq, dur = self.setup_refrain

        for i, (f,d) in enumerate(zip(freq, dur)):
            if f is None: # a rest
                pass

            elif isinstance(f, tuple) or self.timbre_present:
                nxn = self._create_nxn_array(f, d)
                matrix[0:nxn.shape[0], self.sample_boundaries[i]:self.sample_boundaries[i] + nxn.shape[-1]] = nxn

            elif isinstance(f, (float, np.floating, int, np.integer)):
                tone = Tone(frequency=f, duration=d, sample_rate=self.sample_rate).tone

                # self._generate_tone(f, d)

                ## THERE'S SOME WEIRDNESS HERE - NEED TO GET THIS TO WORK WITH THE NEW TONE CLASS

                matrix[0, self.sample_boundaries[i]:self.sample_boundaries[i] + tone.shape[-1]] = tone

            elif isinstance(f, list):
                pass # not supporting arpeggios just yet

        normalized_matrix = librosa.util.normalize(matrix, norm=0, axis=0, threshold=None, fill=None)

        if sum_output:
            return np.sum(normalized_matrix, axis=0)
        else:
            return normalized_matrix
