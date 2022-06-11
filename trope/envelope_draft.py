#!/usr/bin/python3
import numpy as np

class Envelope:

    # insert default arguments for each stage here, e.g. sample_rate * np.linspace(param1, param2, num=param3)
    # also attack level, sustain level, and quiet level

    attack_level = 1.0
    sustain_level = 0.75
    quiet_level = 0.0

    def __init__(
        self,
        input_signal,
        sample_rate,
        attack_setting,
        decay_setting,
        sustain_setting,
        release_setting
        ):
        self.input_signal = input_signal
        self.sample_rate = sample_rate
        self.attack_setting = attack_setting
        self.decay_setting = decay_setting
        self.sustain_setting = sustain_setting
        self.release_setting = release_setting

    @property
    def attack(self):
        attack_range = self.sample_rate * np.linspace(0.001, 5, num=100)

        complete_attack = np.linspace(
            self.quiet_level,
            self.attack_level,
            num=int(attack_range[self.attack_setting])
            )

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
        release_size = self.get_release_size(input_signal_size, self.release_setting)

        envelope = np.empty(input_signal_size)

        if self.attack.size + release_size >= input_signal_size:
            # set the attack
            attack_amt = input_signal_size - release_size
            envelope[:attack_amt] = self.attack[:attack_amt]
            # set the release
            envelope[-release_size:] = self.generate_release(
                input_signal_size,
                min_level=self.attack[attack_amt]
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
                min_level=self.decay[decay_amt]
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
                min_level=self.sustain[sustain_amt]
                )

        return envelope
