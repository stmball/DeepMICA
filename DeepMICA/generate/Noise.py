import numpy as np
from colorednoise import powerlaw_psd_gaussian 

class Noise:
    """
        Class for defining a noise pipeline for adding to a continuous time simulation.

        Each noise layer should return a function that has inputs of the ctsHistory numpy array 
        (See above) and a current array outputted from the previous layer. 
        
        Each noise layer needs to output a current of the same size as the input current, 
        with whatever operations added modifying it.
    """

    def __init__(self):
        self.sequence = []

    def make_noisy(self, array, channels_index=1):
        out = array[:, channels_index].astype('float')
        for layer in self.sequence:
            out = layer(array, out)
        return out

    def add(self, noise_layer):
        self.sequence.append(noise_layer)



# Noise layers - might change this to classes later on... not sure.


def simple_f_noise(exponent, mean=0, sd=1):
    def outfunc(array, current):
        return current + powerlaw_psd_gaussian(exponent, current.shape[0]) * sd + mean
    return outfunc


def scaled_f_noise(exponent, scale_factor=2, mean=0, base_sd=1, channels_index=1):
    def outfunc(array, current):
        modifier = (scale_factor - 1) * \
            array[:, channels_index].astype('float') + 1
        return current + (base_sd * modifier) * powerlaw_psd_gaussian(exponent, current.shape[0]) + mean
    return outfunc


def sinusoidal_noise(amplitude, frequency, time_index=2):
    def outfunc(array, current):
        return current + amplitude * np.sin(2 * np.pi * frequency * array[:, time_index].astype('float'))
    return outfunc


def relaxation_noise_opens(decay_speed, decay_factor, channels_index=1):
    def outfunc(array, current):
        counts = 0
        for i in range(len(array)):
            if int(array[i,channels_index]) == 1:
                current[i] += decay_factor * (np.exp(-decay_speed * counts) - 1)
                counts += 1
            else:
                counts = 0
        return current
    return outfunc
