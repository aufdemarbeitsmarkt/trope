#!/usr/bin/python3
import librosa 
import numpy as np
from scipy.signal import savgol_filter

# TODO: allow end-user to specify linspace() or geomspace() 
class Envelope:

    # insert default arguments for each stage here, e.g. sample_rate * np.linspace(param1, param2, num=param3)

    attack_level = 1.0
    sustain_level = 0.75
    quiet_level = 0.001

    def __init__(
        self,
        attack_setting=None,
        decay_setting=None,
        sustain_setting=None,
        release_setting=None,
        sample_rate=None,
        _from_audio_envelope=None
        ):
        self.attack_setting = attack_setting
        self.decay_setting = decay_setting
        self.sustain_setting = sustain_setting
        self.release_setting = release_setting
        self.sample_rate = sample_rate
        self._from_audio_envelope = _from_audio_envelope


    @classmethod
    def base(cls, sample_rate):
        return cls(
            sample_rate=sample_rate,
            attack_setting=2,
            decay_setting=2,
            sustain_setting=10,
            release_setting=10
            )


    @property
    def attack(self):
        attack_range = self.sample_rate * np.linspace(0.001, 5, num=100)

        complete_attack = np.linspace(
            self.quiet_level,
            self.attack_level,
            num=int(attack_range[self.attack_setting])
            )

        complete_attack[:10] = 0.0
        return complete_attack


    @property
    def decay(self):
        decay_range = self.sample_rate * np.linspace(0.001, 10, num=100)

        complete_decay = np.linspace(
            self.attack_level,
            self.sustain_level,
            num=int(decay_range[self.decay_setting])
        )

        return complete_decay


    @property
    def sustain(self):
        sustain_range = self.sample_rate * np.linspace(0.001, 10, num=100)

        complete_sustain = np.full(int(sustain_range[self.sustain_setting]), self.sustain_level)

        return complete_sustain


    def generate_release(self, input_signal_size=None, min_level=None, max_level=None):
        if min_level is None:
            min_level = self.sustain_level
        if max_level is None:
            max_level = self.quiet_level

        release_size = self.get_release_size(input_signal_size, self.release_setting)

        release_range = self.sample_rate * np.linspace(0.001, 10, num=100)
        complete_release = np.linspace(
            min_level,
            max_level,
            num=release_size
            )
        return complete_release


    @staticmethod
    def get_release_size(input_signal_size, release_point):
        return int(input_signal_size * (release_point/100))


    def generate_envelope_signal(self, input_signal):
                
        input_signal_size = input_signal.size
        
        if self._from_audio_envelope is not None:
            return self._resample_env_from_audio(input_signal_size)

        release_size = self.get_release_size(input_signal_size, self.release_setting)

        envelope = np.zeros(input_signal_size)

        if self.attack.size + release_size >= input_signal_size:
            # set the attack
            attack_amt = input_signal_size - release_size
            envelope[:attack_amt] = self.attack[:attack_amt]
            # set the release
            envelope[-release_size:] = self.generate_release(
                input_signal_size,
                min_level=self.attack[attack_amt - 1]
                )
        elif self.attack.size + self.decay.size + release_size >= input_signal_size:
            # set the attack
            envelope[:self.attack.size] = self.attack
            # set the decay
            decay_amt = input_signal_size - self.attack.size - release_size
            envelope[self.attack.size:-release_size] = self.decay[:decay_amt]
            # set the release
            envelope[-release_size:] = self.generate_release(
                input_signal_size,
                min_level=self.decay[decay_amt - 1]
                )
        elif self.attack.size + self.decay.size + self.sustain.size + release_size >= input_signal_size:
            # set the attack
            envelope[:self.attack.size] = self.attack
            # set the decay
            envelope[self.attack.size:self.attack.size + self.decay.size] = self.decay
            # set the sustain 
            sustain_amt = input_signal_size - self.attack.size - self.decay.size - release_size
            envelope[self.attack.size + self.decay.size:-release_size] = self.sustain[:sustain_amt]
            # set the release
            envelope[-release_size:] = self.generate_release(
                input_signal_size,
                min_level=self.sustain[sustain_amt - 1]
                )
        else:
            # set the attack
            envelope[:self.attack.size] = self.attack
            # set the decay
            envelope[self.attack.size:self.attack.size + self.decay.size] = self.decay
            # set the sustain
            envelope[self.attack.size + self.decay.size:self.attack.size + self.decay.size + self.sustain.size] = self.sustain 
            # set the release 
            envelope[self.attack.size + self.decay.size + self.sustain.size:self.attack.size + self.decay.size + self.sustain.size + release_size] = self.generate_release(
                input_signal_size,
                min_level=self.sustain[-1]
                )

        return envelope

    @classmethod
    def from_audio(cls, input_audio) -> np.array:
        # input is an Audio object
        
        # window_length and polyorder were chose semi-arbitrarily 
        # ran through several values and this seemed to be a sweet spot
        envelope = savgol_filter(
            np.abs(input_audio.audio), 
            window_length=512, 
            polyorder=1, 
            mode='interp'
            )
        normalized_envelope = librosa.util.normalize(envelope)

        return cls(sample_rate=input_audio.sample_rate, _from_audio_envelope=normalized_envelope)
        

    def _resample_env_from_audio(self, input_signal_size):

        target_sample_rate = int((input_signal_size / (self._from_audio_envelope.size)) * self.sample_rate)

        resampled_env = librosa.resample(
            self._from_audio_envelope,
            orig_sr=self.sample_rate,
            target_sr=target_sample_rate
        )

        # ensure the audio and the envelope can be broadcast
        if input_signal_size > resampled_env.size:
            size_diff = input_signal_size - resampled_env.size
            return np.pad(resampled_env, (size_diff,0))
        elif input_signal_size < resampled_env.size:
            return resampled_env[:input_signal_size]
        else:
            return resampled_env