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
        durations,
        sample_rate,
        timbre=None
        ):
        self.input_refrain = np.asmatrix(input_refrain) # this will enforce 2-dimensionality even if it's with an exception (ValueError)
        self.durations = np.asarray(durations)
        self.sample_rate = sample_rate
        self.timbre = timbre

        if self.input_refrain.shape[1] == self.durations.shape[0]:
            # if the shape of the input_refrain's first axis and the durations' 0th axis are the same, give us the setup_refrain as a tuple
            self.setup_refrain = self.input_refrain, self.durations
        else:
            # else, ensure they have the same size by tiling
            self.setup_refrain = np.tile(self.input_refrain, len(self.durations)), np.tile(self.durations, len(self.input_refrain))

        self.duration_in_samples = self._duration_in_samples(self.input_refrain) # returns a matrix of the input_refrain's durations, but in samples
        

        @staticmethod
        def get_durations_samples(frequency, duration_in_seconds, sample_rate):
            if frequency == 0:
                return 0
            cycle_time_samples = sample_rate / frequency
            num_samples = sample_rate * duration
            range_cycles = np.ceil(np.arange(0, int(num_samples + cycle_time_samples), cycle_time_samples))

            diff_second_to_last, diff_last = abs(num_samples - range_cycles[-2]), abs(num_samples - range_cycles[-1])

            which_cycle = np.argmin([diff_second_to_last, diff_last])

            return range_cycles[-2] if which_cycle == 0 else range_cycles[-1]

        def _duration_in_samples(self):
            return np.vectorize(get_duration_in_samples)
