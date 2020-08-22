import librosa
import numpy as np
import pandas as pd
import albumentations
from transforms import (
    MelSpectrogram,
    SpectToImage1c,
    SpectToImage3c,
)


def get_test_samples(train_le, args):
    SpectToImage = SpectToImage1c if "cnn14_att" in args.model else SpectToImage3c
    summary_df = pd.read_csv("../input/test/merged_summary.csv")
    transforms = albumentations.Compose(
        [
            MelSpectrogram(
                parameters=args.melspectrogram_parameters, always_apply=True
            ),
            SpectToImage(always_apply=True),
        ]
    )

    test_samples_1 = prepare_test(
        [
            "../input/test/BLKFR-10-CPL_20190611_093000.pt540.mp3",
            "../input/test/ORANGE-7-CAP_20190606_093000.pt623.mp3",
        ],
        summary_df,
        train_le,
        args,
        transforms,
    )

    test_samples_2 = prepare_test(
        [
            "../input/test/BLKFR-10-CPL_20190611_093000.pt540.mp3",
            "../input/test/ORANGE-7-CAP_20190606_093000.pt623.mp3",
            # "../input/test/SSW49_20170520.wav",
            # "../input/test/SSW50_20170819.wav",
            "../input/test/SSW51_20170819.wav",
            "../input/test/SSW52_20170429.wav",
            # "../input/test/SSW53_20170513.wav",
            "../input/test/SSW54_20170610.wav",
        ],
        summary_df,
        train_le,
        args,
        transforms,
    )

    return test_samples_1, test_samples_2


def load_test_batch(file_name, sr=32000, duration=5):
    try:
        audio = np.load(file_name.replace(".mp3", ".npy").replace(".wav", ".npy"))
    except:
        audio, sr = librosa.load(file_name, sr=sr)
        np.save(file_name.replace(".mp3", ".npy").replace(".wav", ".npy"), audio)
    chunks = len(audio) // (sr * duration)

    audios = []
    for i in range(int(chunks) - 1):
        audios.append(audio[i * sr * duration : (i + 1) * sr * duration])
        # chunk = audio[i * sr * duration : (i + 3) * sr * duration]
        # if len(chunk) < 3 * sr * duration:
        #     print("padded")
        #     chunk = np.pad(chunk, (0,  3 * sr * duration - len(chunk)), "constant")
        # audios.append(chunk)

    return np.vstack(audios)


def transform_test_batch(audios, transforms, sr=32000):
    """ Stack signals to spectrgrams """
    specs = []

    for sound in audios:
        spect = transforms(data=(sound, sr))["data"]
        specs.append(np.expand_dims(spect, 0))

    return np.vstack(specs)


def prepare_test(files, meta, le_encoder, args, transforms):
    inputs = []
    targets = []
    for file_ in files:
        batch = load_test_batch(file_, sr=args.sample_rate)
        # batch = transform_test_batch(batch, transforms, sr=args.sample_rate)

        test = meta[
            meta.filename_seconds.apply(lambda x: x.rsplit("_", 1)[0])
            == file_.split("/")[-1].split(".")[0]
        ]

        def transform(xx):
            xx = [x for x in xx if x in le_encoder.classes_]
            return le_encoder.transform(xx)

        test["labels"] = test.birds.fillna("").apply(
            lambda x: transform(x.split()) if x is not None else []
        )

        target = np.zeros((len(test), args.num_classes + 1))

        for i, values in enumerate(test["labels"].values):
            if len(values) == 0:
                target[i, -1] = 1
            else:
                target[i, values] = 1

        print("Test file:", file_, target[:, : args.num_classes].sum())

        inputs.append(batch)
        targets.append(target)

    return {"spect": np.vstack(inputs), "targets": np.vstack(targets)}
