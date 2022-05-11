#!/usr/bin/python3
import librosa
import numpy as np
from scipy.signal import filtfilt, butter, ellip
from performing import Audio

sample_rate = 22050

from math import ceil

def generate_tone(frequency, duration, amplitude=1, phase=0, phi=0):
    each_sample = np.arange(ceil(duration * sample_rate))
    tone = amplitude * np.sin(2 * np.pi * each_sample * frequency / sample_rate) * np.cos(2 * np.pi * each_sample * 5 / sample_rate + (np.pi / 180 + phase))
    return tone

freqs = [110, 180, 330]
phases = [90, 270, 630]

phased_tones = []
for f,p in zip(freqs, phases):
    tone = generate_tone(f, duration=4, phase=p)
    phased_tones.append(tone)

summed = np.sum(phased_tones, axis=0)

A = Audio(summed)
A.save('phase_test')
