#!/usr/bin/python3
from itertools import product
import librosa
import numpy as np


class Tone:

    default_amplitude = 0.5

    def __init__(
        self,
        frequency,
        duration_in_samples,
        sample_rate,
        amplitude=None
        ):
        self.frequency = frequency
        self.duration_in_samples = self.duration_in_samples
        self.sample_rate = sample_rate

        if amplitude is None:
            amplitude = self.default_amplitude
        self.amplitude = amplitude

        self.each_sample = np.arange(self.duration_in_samples)

        self.tone = self._generate_tone()

    def _generate_tone(self):
        sine = np.sin(2 * np.pi * self.each_sample * self.frequency / self.sample_rate) * self.amplitude
        return sine


class Synthesis:

    def __init__(
        self,
        input_refrain,
        input_durations,
        sample_rate,
        timbre=None
        ):
        self.input_refrain = np.asmatrix(input_refrain) # this will enforce 2-dimensionality via exception - ValueError
        self.input_durations = np.asarray(input_durations)
        self.sample_rate = sample_rate
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

    def _get_duration_in_samples(self, frequency, duration_in_seconds):
        if frequency == 0:
            return 0
        cycle_time_samples = self.sample_rate / frequency
        num_samples = self.sample_rate * duration_in_seconds
        range_cycles = np.ceil(np.arange(0, int(num_samples + cycle_time_samples), cycle_time_samples))

        diff_second_to_last, diff_last = abs(num_samples - range_cycles[-2]), abs(num_samples - range_cycles[-1])

        which_cycle = np.argmin([diff_second_to_last, diff_last])

        return range_cycles[-2] if which_cycle == 0 else range_cycles[-1]
