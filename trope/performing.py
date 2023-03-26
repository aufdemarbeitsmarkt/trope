#!/usr/bin/python3
import IPython.display as ipd
import librosa
import numpy as np
from scipy.io.wavfile import write

from effects import Effect
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
    
    # "helper" methods for normalizing and summing audio
    def _normalize_audio(self, audio):
        return librosa.util.normalize(audio, norm=0, axis=0, threshold=None, fill=None)
    
    def _sum_audio(self, audio):
        return np.sum(audio, axis=0) 

    def _sum_and_normalize(self, audio):
        return self._sum_audio(self._normalize_audio(audio))

    def save(self, filename=None, filetype='wav'):
        file = f'{filename}.{filetype}'
        write(filename=file, rate=self.sample_rate, data=self.audio)

    @classmethod
    def load(cls, file, sr=default_sample_rate, mono=True, offset=0.0, duration=None):
        # TODO: confirm whether self.sample_rate is saved properly if the end-user changes the `sr` argument
        y, _ = librosa.load(file, sr=sr, mono=mono, offset=offset, duration=duration)
        return cls(y, sr)

    def play(self):
        #  TODO: there needs to be a check to ensure audio is summed normalized before playing, see: https://github.com/aufdemarbeitsmarkt/trope/issues/10
        return ipd.Audio(self.audio, rate=self.sample_rate)


class Performer(Audio):

    DEFAULT_BPM = 120

    def __init__(
        self,
        refrain,
        audio=None,
        sample_rate=None,
        note_type=None,
        duration_type=None,
        tempo=None,
        **kwargs
        ):
        super().__init__(sample_rate)
        
        self.refrain = np.asarray(refrain)
        
        if sample_rate is None:
            sample_rate = self.default_sample_rate
        self.sample_rate = sample_rate


        if note_type is None:
            note_type = 'name' # TODO: this is a "placeholder name"; come up with list of options, e.g. 'name', 'frequency', ? 
        self.note_type = note_type
        
        if duration_type is None:
            duration_type = 'beat' # TODO: this is a "placeholder name"; come up with list of options, e.g. 'beat', 'second', ? 
        self.duration_type = duration_type 

        if tempo == None:
            tempo = self.DEFAULT_BPM
        self.tempo = tempo

        for k,v in kwargs.items():
            setattr(self, k, v)
        
        # TODO: figure out a clearer way to create this audio attribute
        self.audio = self._create_audio()

    def _create_audio(self):
        durations = getattr(self, 'durations', [1])
        effects = getattr(self, 'effects', None)
        envelope = getattr(self, 'envelope', None)
        loop = getattr(self, 'loop', None)
        timbre = getattr(self, 'timbre', ((1,1), (1,1))) # TODO: evenutally, remove this fallback (right now, a user is required to input a timbre arg, but they shouldn't have to)

        y = Synthesis(
            input_refrain=self.refrain,
            input_durations=durations,
            sample_rate=self.sample_rate,
            note_type=self.note_type,
            duration_type=self.duration_type,
            tempo=self.tempo,
            timbre=timbre,
            envelope=envelope
        ).synthesized_output

        if loop is not None:
            # TODO: allow user to define a loop later, even if they've already instantiated Performer
            y = np.tile(y, loop)

        if effects is None:
            return self._sum_and_normalize(y)
            # return y
        elif effects is not None:
            effect_y = Effect(
                input_audio=y,
                sample_rate=self.sample_rate,
                **effects
            ).output_audio
            return self._sum_and_normalize(effect_y)
            # return effect_y


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

        # TODO: incorporate summing and normalizing here
        normalized_performers = librosa.util.normalize(self.performers, norm=0, axis=0, threshold=None, fill=None)

        return np.sum(normalized_performers, axis=0)