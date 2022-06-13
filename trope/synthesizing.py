#!/usr/bin/python3
from itertools import product
import librosa
from math import ceil
import numpy as np

from envelope import Envelope

class Synthesis:

    def __init__(
        self,
        input_refrain,
        input_durations,
        sample_rate,
        envelope=None,
        timbre=None
        ):
        self.input_refrain = np.asmatrix(input_refrain) # this will enforce 2-dimensionality via exception - ValueError
        self.input_durations = np.asarray(input_durations)
        self.sample_rate = sample_rate
        self.envelope = envelope
        self.timbre = timbre

        if self.input_refrain.shape[1] != self.input_durations.shape[0]:
            # ensure they have the same size by tiling, create 'refrain' and 'durations' attributes
            self.refrain = np.tile(self.input_refrain, len(self.input_durations))
            self.durations = np.tile(self.input_durations, self.input_refrain.shape[1])

        else:
            self.refrain = self.input_refrain
            self.durations = self.input_durations

        self._vectorized_get_duration_in_samples = np.vectorize(self._get_duration_in_samples)

        self.durations_in_samples = self._vectorized_get_duration_in_samples(self.refrain, self.durations)

        self.max_durations_samples = np.max(self.durations_in_samples, axis=0).astype('int')

        _cumsum_max_durations = np.ravel(np.cumsum(self.max_durations_samples))
        self.sample_boundaries = np.insert(_cumsum_max_durations[:-1], 0, 0)

        self.synthesized_output = self._synthesize()

    def _generate_tone(self, frequency, duration_in_samples, amplitude=0.5, pad_amount=0):
        each_sample = np.arange(duration_in_samples)
        sine = np.sin(2 * np.pi * each_sample * frequency / self.sample_rate) * amplitude

        if pad_amount - duration_in_samples > 0:
            pad_for_each_side = (pad_amount - duration_in_samples) / 2
            beginning = int(pad_for_each_side)
            end = ceil(pad_for_each_side)
            return np.pad(sine, (beginning, end)) # note: also works for rests

        return sine

    def _get_duration_in_samples(self, frequency, duration_in_seconds):
        if frequency == 0:
            return 0
        cycle_time_samples = self.sample_rate / frequency
        num_samples = self.sample_rate * duration_in_seconds
        range_cycles = np.ceil(np.arange(0, int(num_samples + cycle_time_samples), cycle_time_samples))

        diff_second_to_last, diff_last = abs(num_samples - range_cycles[-2]), abs(num_samples - range_cycles[-1])

        which_cycle = np.argmin([diff_second_to_last, diff_last])

        return range_cycles[-2] if which_cycle == 0 else range_cycles[-1]

    def _initialize_matrix(self):
        '''
        Sets self.total_duration_in_samples.
        Returns a matrix for the entire Synthesis object, i.e. allocates space in memory in advance.
        '''
        self.total_duration_in_samples = np.sum(self.max_durations_samples, dtype='int')
        return np.empty((self.refrain.shape[0], self.total_duration_in_samples))

    def _synthesize(self, normalize_output=True, sum_output=True):
        output = self._initialize_matrix()

        for i,r in np.ndenumerate(self.refrain):

            tone = self._generate_tone(
                frequency=r,
                duration_in_samples=self.durations_in_samples[i], pad_amount=self.max_durations_samples[0,i[1]]
                )

            # set envelope
            if self.envelope is None: 
                E = Envelope.base(sample_rate=self.sample_rate)
            else:
                E = Envelope(*self.envelope, sample_rate=self.sample_rate)
            
            env = E.generate_envelope_signal(tone)
            tone *= env

            output[i[0], self.sample_boundaries[i[1]] : self.sample_boundaries[i[1]]+tone.size] = tone

        if normalize_output:
            output = librosa.util.normalize(output, norm=0, axis=0, threshold=None, fill=None)
        if sum_output:
            output = np.sum(output, axis=0)

        return output
