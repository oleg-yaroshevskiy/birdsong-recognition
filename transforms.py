import scipy
import scipy.signal
import librosa
import numpy as np
import tqdm
from multiprocessing import Pool
import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform
import random
import pandas as pd
import glob
from multiprocessing import Manager

from numpy import sqrt, newaxis
from numpy.fft import irfft, rfftfreq
from numpy.random import normal
from numpy import sum as npsum

import cv2

from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=1, fs=30.0, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def powerlaw_psd_gaussian(exponent, size, fmin=0):
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    ix = npsum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w ** 2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    sr = normal(scale=s_scale, size=size)
    si = normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y


class AudioTransform(BasicTransform):
    """Transform for Audio task"""

    @property
    def targets(self):
        return {"data": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params


class NoiseInjection(AudioTransform):
    """It simply add some random value into data by using numpy"""

    def __init__(self, noise_levels=(0, 0.5), always_apply=False, p=0.5):
        super(NoiseInjection, self).__init__(always_apply, p)

        self.noise_levels = noise_levels

    def apply(self, data, **params):
        sound, sr = data
        noise_level = np.random.uniform(*self.noise_levels)
        noise = np.random.randn(len(sound))
        augmented_sound = sound + noise_level * noise
        # Cast back to same data type
        augmented_sound = augmented_sound.astype(type(sound[0]))

        return augmented_sound, sr


class PinksNoiseInjection(AudioTransform):
    """It simply add some random value into data by using numpy"""

    def __init__(self, always_apply=False, p=0.5):
        super(PinksNoiseInjection, self).__init__(always_apply, p)

        self.pink_noise = powerlaw_psd_gaussian(1, 360 * 32000)
        self.brown_noise = powerlaw_psd_gaussian(2, 360 * 32000)

    def apply(self, data, **params):
        sound, sr = data

        idx = np.random.randint(0, 360 * 32000 - len(sound))
        noise_level = np.random.rand()

        if np.random.rand() < 0.5:
            noise = self.pink_noise[idx : idx + len(sound)]
        else:
            noise = self.brown_noise[idx : idx + len(sound)]

        augmented_sound = sound + noise_level * noise
        # Cast back to same data type
        augmented_sound = augmented_sound.astype(type(sound[0]))

        return augmented_sound, sr


class ShiftingTime(AudioTransform):
    """Shifting time axis"""

    def __init__(self, always_apply=False, p=0.5):
        super(ShiftingTime, self).__init__(always_apply, p)

    def apply(self, data, **params):
        sound, sr = data

        shift_max = np.random.randint(1, len(sound))
        shift = np.random.randint(int(sr * shift_max))
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift

        augmented_sound = np.roll(sound, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_sound[:shift] = 0
        else:
            augmented_sound[shift:] = 0

        return augmented_sound, sr


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(PitchShift, self).__init__(always_apply, p)

    def apply(self, data, **params):
        sound, sr = data

        n_steps = np.random.randint(-6, 6)
        augmented_sound = librosa.effects.pitch_shift(sound, sr, n_steps)

        return augmented_sound, sr


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(TimeStretch, self).__init__(always_apply, p)

    def apply(self, data, **params):
        sound, sr = data

        rate = np.random.uniform(0, 2)
        augmented_sound = librosa.effects.time_stretch(sound, rate)

        return augmented_sound, sr


class IntRandomAudio(AudioTransform):
    def __init__(self, seconds=5, always_apply=False, p=0.5):
        super(IntRandomAudio, self).__init__(always_apply, p)

        self.seconds = seconds

    def apply(self, data, **params):
        sound, sr = data
        half = int(sr * self.seconds) // 2

        if len(sound) < half * 2:
            padding = half * 2 - len(sound)
            trim_sound = np.pad(sound, (0, padding), "constant")
            return trim_sound, sr

        step = sr // 10
        frames = len(sound) // step
        probs = np.abs(sound)[: frames * step].reshape(-1, step).max(axis=-1)
        probs /= probs.sum()

        idx = np.random.choice(range(frames), 1, p=probs)[0] * step + np.random.randint(
            -(step // 2), (step // 2)
        )

        if idx < half:
            return sound[: half * 2], sr

        elif idx > len(sound) - half:
            return sound[-half * 2 :], sr

        else:
            return sound[idx - half : idx + half], sr


class RandomAudio(AudioTransform):
    def __init__(self, seconds=5, always_apply=False, p=0.5):
        super(RandomAudio, self).__init__(always_apply, p)

        self.seconds = seconds

    def apply(self, data, **params):
        sound, sr = data

        shift = np.random.randint(len(sound))
        trim_sound = np.roll(sound, shift)

        min_samples = int(sr * self.seconds)

        if len(trim_sound) < min_samples:
            padding = min_samples - len(trim_sound)
            offset = padding // 2
            trim_sound = np.pad(trim_sound, (offset, padding - offset), "constant")
        else:
            trim_sound = trim_sound[:min_samples]

        return trim_sound, sr


class AddBackground(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(AddBackground, self).__init__(always_apply, p)
        self.background_audios = np.load("../input/env/bg2.5.npy")
        self.background_audios_duration = len(self.background_audios)

    def apply(self, data, **params):
        sound, sr = data
        frame = np.random.randint(0, self.background_audios_duration - len(sound))
        bg = self.background_audios[frame : frame + len(sound)]
        noise_level = np.random.rand()

        sound = sound + bg * noise_level

        return sound, sr


class VolumeOff(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(VolumeOff, self).__init__(always_apply, p)

    def apply(self, data, **params):
        sound, sr = data
        alpha = np.random.rand()
        sound = alpha * sound

        return sound, sr


class MelSpectrogram(AudioTransform):
    def __init__(self, parameters, always_apply=False, p=0.5):
        super(MelSpectrogram, self).__init__(always_apply, p)

        self.parameters = parameters

    def apply(self, data, **params):
        sound, sr = data

        melspec = librosa.feature.melspectrogram(sound, sr=sr, **self.parameters)
        melspec = np.log(melspec + 1e-8)
        # melspec = librosa.power_to_db(melspec)
        melspec = melspec.astype(np.float32)

        return melspec, sr


class Stft(AudioTransform):
    def __init__(self, parameters={}, always_apply=False, p=0.5):
        super(Stft, self).__init__(always_apply, p)

        self.parameters = parameters

    def apply(self, data, **params):
        sound, sr = data

        _, _, stft = scipy.signal.stft(sound, nperseg=512, noverlap=192)
        melspec = np.log(np.abs(stft).clip(1e-5, 10))
        melspec = cv2.resize(melspec[1:], (melspec.shape[1], 64))
        melspec = melspec.astype(np.float32)

        # print(melspec.min(), melspec.max())

        return melspec, sr


class SpecAugment(AudioTransform):
    def __init__(
        self,
        num_mask=2,
        freq_masking=0.15,
        time_masking=0.20,
        always_apply=False,
        p=0.5,
    ):
        super(SpecAugment, self).__init__(always_apply, p)

        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def apply(self, data, **params):
        melspec, sr = data

        spec_aug = self.spec_augment(
            melspec, self.num_mask, self.freq_masking, self.time_masking, melspec.min()
        )

        return spec_aug, sr

    # Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
    def spec_augment(
        self,
        spec: np.ndarray,
        num_mask=2,
        freq_masking=0.15,
        time_masking=0.20,
        value=0,
    ):
        spec = spec.copy()
        num_mask = random.randint(1, num_mask)
        for i in range(num_mask):
            all_freqs_num, all_frames_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0 : f0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[:, t0 : t0 + num_frames_to_mask] = value

        return spec


class SpectToImage1c(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(SpectToImage1c, self).__init__(always_apply, p)

    def apply(self, data, **params):
        image, sr = data
        image = image.astype(np.float32) / 100.0

        return np.expand_dims(image, 0)


class SpectToImage3c(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(SpectToImage3c, self).__init__(always_apply, p)

    def apply(self, data, **params):
        image, sr = data
        delta = librosa.feature.delta(image)
        accelerate = librosa.feature.delta(image, order=2)
        image = np.stack([image, delta, accelerate], axis=0)
        image = image.astype(np.float32) / 100.0

        return image


class LowFrequencyMask(AudioTransform):
    def __init__(
        self, p: int = 0.5, always_apply=False, max_cutoff=6000, min_cutoff=800
    ):
        super(LowFrequencyMask, self).__init__(always_apply, p)
        self.max_cutoff = max_cutoff
        self.min_cutoff = min_cutoff

    def apply(self, data, **params):
        audio, sr = data

        cutoff_value = np.random.randint(low=self.min_cutoff, high=self.max_cutoff)
        audio = butter_lowpass_filter(audio, cutoff=cutoff_value, fs=sr)

        return audio, sr
