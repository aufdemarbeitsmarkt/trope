#!/usr/bin/python3
import librosa
import numpy as np
from scipy.io.wavfile import write
import time

from effects import Effect
from improvising import Improv
from synthesizing import Synthesis
from scales_and_tunings import convert_hz_to_note


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

    def save(self, filename=None, filetype='wav'):
        file = f'saved/{filename}.{filetype}'
        write(filename=file, rate=self.sample_rate, data=self.audio)

    @classmethod
    def load(cls, file, sr=default_sample_rate, mono=True, offset=0.0, duration=None):
        # TODO: confirm whether self.sample_rate is saved properly if the end-user changes the `sr` argument
        y, _ = librosa.load(file, sr=sr, mono=mono, offset=offset, duration=duration)
        return cls(y, sr)

    def play(self):
        import IPython.display as ipd
        return ipd.Audio(self.audio, rate=self.sample_rate)


class Performer(Audio):

    def __init__(
        self,
        refrain,
        audio=None,
        sample_rate=None,
        **kwargs
        ):
        super().__init__(sample_rate)
        self.__dict__.update(kwargs)
        
        self.refrain = np.asarray(refrain)

        if sample_rate is None:
            sample_rate = self.default_sample_rate
        self.sample_rate = sample_rate

        self.audio = self._create_audio()


    def _create_audio(self):
        effects_dict = self.__dict__.get('effects')
        durations = self.__dict__.get('durations', [1])
        timbre = self.__dict__.get('timbre', ((1,1), (1,1))) # TODO: remove this; temporary fallback til I fix where a user is required to input a timbre arg, but they shouldn't have to
        envelope = self.__dict__.get('envelope')
        # TODO: improvisation could be a kwarg, but I don't have to imply that the user can't create a refrain-type variable and just instantiate an Improv class with that (then call the desired method later)
        # something like:
        # improvisation_type = self.__dict__.get('improv) # in Improv class, map the argument to the corresponding method

        # check whether the performer is passing note values directly
        # if all([notes.dtype.type is np.str_ for r in self.refrain for notes in r]):
        #     self.refrain = convert_hz_to_note(self.refrain)

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