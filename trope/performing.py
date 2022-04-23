#!/usr/bin/python3
import librosa
import numpy as np
from scipy.io.wavfile import write
import sounddevice
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

    def play(self):
        sounddevice.play(self.audio, self.sample_rate)
        time.sleep(self.playtime)
        self.stop()

    def stop(self):
        sounddevice.stop()

    def save(self, filename=None, filetype='.wav'):
        write(filename=filename + filetype, rate=self.sample_rate, data=self.audio)

    @classmethod
    def load(cls, file, sr=default_sample_rate, mono=True, offset=0.0, duration=None):
        y, _ = librosa.load(file, sr=sr, mono=mono, offset=offset, duration=duration)
        return cls(y, sr)

    @classmethod
    def record(cls, duration, sr=default_sample_rate):
        y = sounddevice.rec(int(duration * sr), samplerate=sr, channels=2, blocking='True')
        # sounddevice.wait()
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

        y = Synthesis(
            input_refrain=self.refrain,
            durations=durations,
            sample_rate=self.sample_rate,
            timbre=timbre
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




from librosa import note_to_hz
riff = note_to_hz(['A3', 'C#3', 'C#4', 'E4', 'D4', 'E4', 'A3', 'C#4'])

p1 = Performer(
    Improv(riff).markov(walk_length=len(riff) * 3),
    durations=np.tile([0.5, 0.25],20),
    timbre=((1,1), (3,0.5), (5,0.25), (7,0.25)),
    effects={'delay': (8, 500, (0.9, 0.1), 'reflect')}
)

p2 = Performer(
    Improv(riff).markov(walk_length=8),
    durations=np.tile([0.25, 0.5, 0.25], 20),
timbre=((1,1), (4,0.5), (5,0.25), (8,0.25)),
    effects={'delay': (5, 100, (0.9, 0.1), 'reflect')}
)

perf = Performance(
    [p1, p2]
)

perf.save('performance_example!')
