#!/usr/bin/python3
import librosa
import numpy as np
from scipy.io.wavfile import write
import time

from effects import Effect
from improvising import Improv
from synthesizing import Synthesis



class Audio:

    default_sample_rate = 22050

    def __init__(self, audio, sample_rate=None):
        self.audio = audio
        if sample_rate is None:
            sample_rate = self.default_sample_rate
        self.sample_rate = sample_rate

    @property
    def playtime(self):
        return librosa.get_duration(y=self.audio, sr=self.sample_rate)

    def save(self, filename=None, filetype='.wav'):
        write(filename=filename + filetype, rate=self.sample_rate, data=self.audio)

    @classmethod
    def load(cls, file, sr=default_sample_rate, mono=True, offset=0.0, duration=None):
        y, _ = librosa.load(file, sr=sr, mono=mono, offset=offset, duration=duration)
        return cls(y, sr)


class Performer(Audio):

    note_type = 'hz'

    def __init__(
        self,
        refrain,
        audio=None,
        sample_rate=None,
        note_type=None,
        **kwargs
        ):
        super().__init__(sample_rate)
        self.__dict__.update(kwargs)
        self.refrain = refrain

        if sample_rate is None:
            sample_rate = self.default_sample_rate
        self.sample_rate = sample_rate

        if note_type is None:
            note_type = self.note_type
        self.note_type = note_type

        self.audio = self._create_audio()

    def _create_audio(self):
        effects_dict = self.__dict__.get('effects')
        durations = self.__dict__.get('durations', [1])
        timbre = self.__dict__.get('timbre')
        envelope = self.__dict__.get('envelope')

        y = Synthesis(
            input_refrain=self.refrain,
            input_durations=durations,
            sample_rate=self.sample_rate,
            timbre=timbre,
            envelope=envelope
        ).synthesized_output

        if effects_dict is None:
            return y
        elif effects_dict is not None:
            effect_y = Effect(
                input_audio=y,
                sample_rate=self.sample_rate,
                **effects_dict
            ).output_audio
            return effect_y


class Performance(Audio):

    def __init__(self, performers=None, audio=None, sample_rate=None,  **kwargs):
        super().__init__(sample_rate)
        if performers is None:
            self.performers = []
        else:
            shortest = min([len(p.audio) for p in performers])
            self.performers = [p.audio[:shortest] for p in performers]

        self.__dict__.update(kwargs)

        self.audio = self._create_performance()
        if sample_rate is None:
            sample_rate = self.default_sample_rate
        self.sample_rate = sample_rate

    def _create_performance(self):

        normalized_performers = librosa.util.normalize(self.performers, norm=0, axis=0, threshold=None, fill=None)

        return np.sum(normalized_performers, axis=0)
