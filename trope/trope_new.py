#!/usr/bin/python3
import librosa
import numpy as np
from scipy.io.wavfile import write
import sounddevice
import time


class Audio:

    default_sample_rate = 44100

    def __init__(self, audio: np.array, sample_rate: float = default_sample_rate):
        self.audio = audio
        self.sample_rate = sample_rate

    def set_sample_rate(self):
        pass

    def play(self):
        sounddevice.play(self.audio, self.sample_rate)
        time.sleep(len(self.audio) / self.sample_rate)
        self.stop()

    def loop(self):
        # use this to loop audio / could make an argument for play to loop, too
        pass

    def stop(self):
        sounddevice.stop()

    def save(self, filename=None, filetype='wav', sample_rate=default_sample_rate):
        write(filename=filename + '.' + filetype, rate=sample_rate, data=self.audio)

    @classmethod
    def load(cls, file, sample_rate=default_sample_rate, mono=True, offset=0.0, duration=None): # additional add arguments for librosa.load() eventually
        audio, sample_rate = librosa.load(file, sr=sample_rate, mono=mono, offset=offset, duration=duration)
        return cls(audio, sample_rate)

    @classmethod
    def record(cls, duration, sample_rate=default_sample_rate):
        audio = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, blocking='True')
        # sounddevice.wait()
        return cls(audio, sample_rate)

    @classmethod
    def tone(cls, frequency, duration, amplitude=0.5, sample_rate=default_sample_rate):
        each_sample = np.arange(duration * sample_rate)
        t = np.sin(2 * np.pi * each_sample * frequency / sample_rate) * amplitude
        return cls(t, sample_rate)

    def __add__(self, other):
        # will have to consider
        # ValueError: operands could not be broadcast together with shapes (13636896,) (13680996,)
        # i.e. summing of two unequally lengthed pieces of audio will need to be fixed
        # print(len(self.audio), len(other.audio))
        combined_audio = self.audio + other.audio
        return Audio(combined_audio, self.sample_rate)


file_path = '/Users/timnilsen/Documents/Steely Dan - Aja/'
file = '4. Peg (Album Version).flac'
track, sr = librosa.load(file_path + file, duration=2)

A = Audio(track, sr)
A.play()
