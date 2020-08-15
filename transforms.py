import scipy
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


def compute_stft(audio, window_size, hop_size, log=True, eps=1e-4):
    f, t, s = scipy.signal.stft(audio, nperseg=window_size, noverlap=hop_size)

    s = np.abs(s)

    if log:
        s = np.log(s + eps)

    return s


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

        n_steps = np.random.randint(-10, 10)
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
        df = pd.read_csv("../input/env/esc50.csv")
        categories = [
            "thunderstorm",
            "airplane",
            "sheep",
            "water_drops",
            "wind",
            "footsteps",
            "frog",
            "brushing_teeth" "drinking_sipping",
            "rain",
            "insects",
            "breathing",
            "coughing",
            "clock_tick",
            "sneezing",
            "sea_waves",
            "crickets",
        ]
        self.background_audios = df[df.category.isin(categories)].filename.values

    def apply(self, data, **params):
        sound, sr = data
        try:
            alpha = np.random.rand()
            bg = random.choice(self.background_audios)
            bg = librosa.load("../input/env/audio/audio/" + bg, 32000)[0]

            sound = alpha * sound + bg * (1 - alpha)
            #print("all good")
        except Exception as e:
            #print("shit happens", e)
            "nothing"

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
        melspec = librosa.power_to_db(melspec)
        melspec = melspec.astype(np.float32)

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


class SpectToImage(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(SpectToImage, self).__init__(always_apply, p)

    def apply(self, data, **params):
        image, sr = data
        delta = librosa.feature.delta(image)
        accelerate = librosa.feature.delta(image, order=2)
        image = np.stack([image, delta, accelerate], axis=0)
        image = image.astype(np.float32) / 100.0

        return image  # np.expand_dims(image, 0)
