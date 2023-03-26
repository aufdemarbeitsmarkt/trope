#!/usr/bin/python3
import librosa
import numpy as np


class Effect:

    def __init__(self, input_audio, sample_rate, **kwargs):
        self.input_audio = input_audio
        self.sample_rate = sample_rate
        self.__dict__.update(kwargs)

        self.output_audio = self._delay()

    def _delay(self):
        delay_params = self.__dict__.get('delay')
        # TODO: "feedback" implies the signal feeds back into itself; should rename this something like "repeats" 
        feedback, delay_time, decay, mode = delay_params
        delay_time_samples = int(self.sample_rate * (delay_time / 1000))

        if mode is None or mode == 'empty': # empty could be bad
            mode = 'constant'

        delayed_signal = librosa.feature.stack_memory(self.input_audio, n_steps=feedback, delay=delay_time_samples, mode=mode)

        if decay is not None:
            decay_range = np.linspace(decay[0], decay[1], feedback)
            for drng, dlyd in zip(decay_range, delayed_signal[1:]):
                dlyd *= drng

        return delayed_signal 
    