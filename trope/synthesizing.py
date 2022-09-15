#!/usr/bin/python3
import librosa
from math import ceil
import numpy as np

from envelope import Envelope
from scales_and_tunings import convert_hz_to_note


class Synthesis:

    def __init__(
        self,
        input_refrain,
        input_durations,
        sample_rate,
        envelope=None,
        timbre=None
        ):
        # TODO
        # enforce 2-dimensionality of refrain and 1-dimensionality of durations
        self.input_refrain = np.asarray(input_refrain) 
        self.input_durations = np.asarray(input_durations)
        self.sample_rate = sample_rate
        self.envelope = envelope
        self.timbre = timbre

        # check whether the performer is passing note values directly
        if all([notes.dtype.type is np.str_ for r in self.input_refrain for notes in r]):
            self.input_refrain = convert_hz_to_note(self.input_refrain)

        if self.input_refrain.shape[1] != self.input_durations.shape[0]:
            # ensure they have the same size by tiling, create 'refrain' and 'durations' attributes
            self.refrain = np.tile(self.input_refrain, len(self.input_durations))
            self.durations = np.tile(self.input_durations, self.input_refrain.shape[1])
        else:
            self.refrain = self.input_refrain
            self.durations = self.input_durations
        
        # if a timbre value is provided, use that to set refrain to include the timbre values
        if self.timbre is not None:
            self.refrain = np.reshape(
                    [self.refrain * factor for (factor,_) in self.timbre], 
                    (self.refrain.shape[0] * len(self.timbre), self.refrain.shape[-1])
                    )

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

        t = iter(self.timbre) if self.timbre is not None else False
        counter = 0

        for i,r in np.ndenumerate(self.refrain):
            # use this counter to grab the next value from t, i.e. self.timbre
            if counter == 0:
                # TODO - this 0.5 default value for amp could be specified elsewhere, especially to allow the end-user to set it themselves
                amp = next(t)[1] if t else 0.5 
            counter += 1
            if counter / (self.refrain.size / len(self.timbre)) == 1:
                counter = 0

            tone = self._generate_tone(
                frequency=r,
                duration_in_samples=self.durations_in_samples[i], 
                amplitude=amp,
                pad_amount=self.max_durations_samples[i[1]]
                )

            # set envelope
            if self.envelope is None: 
                E = Envelope.base(sample_rate=self.sample_rate)

            # if we've an envelope generated directly from audio, the envelope will simply be the user-defined instance of the class
            # TODO: make this a bit smarter; can I self.__dict__.get or something like that? 
            elif hasattr(self.envelope, '_from_audio_envelope'):
                if self.envelope._from_audio_envelope is not None:
                    E = self.envelope
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