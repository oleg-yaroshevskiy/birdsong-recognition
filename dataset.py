import librosa
import os
from pydub import AudioSegment
import albumentations
from transforms import (
    IntRandomAudio,
    RandomAudio,
    NoiseInjection,
    MelSpectrogram,
    Stft,
    SpecAugment,
    SpectToImage1c,
    SpectToImage3c,
    AddBackground,
    VolumeOff,
    PinksNoiseInjection,
    LowFrequencyMask,
)
from args import args
import numpy as np
import torch


def get_train_augmentations(args):
    SpectToImage = SpectToImage1c if "cnn14_att" in args.model else SpectToImage3c
    train_audio_augmentation = [
        IntRandomAudio(seconds=args.max_duration, always_apply=True)
    ]

    if args.augm_vol_prob > 0:
        train_audio_augmentation.append(VolumeOff(p=args.augm_vol_prob))

    if args.augm_noise_or_bg > 0:
        train_audio_augmentation.append(
            albumentations.core.composition.OneOf(
                [
                    AddBackground(p=args.augm_bg_prob),
                    PinksNoiseInjection(p=args.augm_noise_prob),
                ],
                p=args.augm_noise_or_bg,
            )
        )

    if args.augm_low_pass > 0:
        train_audio_augmentation.append(LowFrequencyMask(p=0.75))

    # train_audio_augmentation.extend(
    #     [
    #         MelSpectrogram(
    #             parameters=args.melspectrogram_parameters, always_apply=True
    #         ),
    #         SpecAugment(p=args.augm_spec_prob),
    #         SpectToImage(always_apply=True),
    #     ]
    # )
    return albumentations.Compose(train_audio_augmentation)


def get_valid_augmentations(args):
    SpectToImage = SpectToImage1c if "cnn14_att" in args.model else SpectToImage3c
    return albumentations.Compose(
        [
            IntRandomAudio(seconds=args.max_duration, always_apply=True),
            # MelSpectrogram(
            #     parameters=args.melspectrogram_parameters, always_apply=True
            # ),
            # SpectToImage(always_apply=True),
        ]
    )


class BirdDataset:
    def __init__(self, df, args, valid=False):
        self.args = args
        self.filename = df.filename.values
        self.ebird_label = df.ebird_label.values
        self.ebird_label_secondary = df.ebird_label_secondary.values
        self.ebird_code = df.ebird_code.values
        self.sample_rate = args.sample_rate

        self.folder = df.folder.values

        if valid:
            self.aug = get_valid_augmentations(args)
        else:
            self.aug = get_train_augmentations(args)

    def __len__(self):
        return len(self.filename)

    def load_npy(self, path):
        try:
            return (
                np.load(path.replace(".mp3", ".npy").replace(".wav", ".npy")).astype(
                    np.float32
                ),
                self.sample_rate,
            )
        except:
            print("can't read file", path)
            return (
                np.zeros(self.sample_rate * self.args.max_duration, dtype=np.float32),
                self.sample_rate,
            )

    def __getitem__(self, item):
        filename = self.filename[item]
        ebird_code = self.ebird_code[item]
        ebird_label = self.ebird_label[item]
        ebird_label_secondary = torch.zeros(self.args.num_classes)
        ebird_label_secondary.scatter_(
            0, torch.Tensor(self.ebird_label_secondary[item]).long(), 1
        )
        folder = self.folder[item]

        data = self.load_npy(f"{args.ROOT_PATH}/{folder}/{ebird_code}/{filename}")
        spect, _ = self.aug(data=data)["data"]

        target = ebird_label

        return {
            "spect": torch.tensor(spect, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.long),
            "target_secondary": ebird_label_secondary.long(),
        }
