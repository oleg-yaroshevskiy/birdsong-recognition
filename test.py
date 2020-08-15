import librosa
import numpy as np
import pandas as pd


def load_test_batch(file_name, sr=32000):
    audio, sr = librosa.load(file_name, sr=sr)
    chunks = len(audio) // (32000 * 5)

    audios = []
    for i in range(int(chunks) - 1):
        audios.append(audio[i * 32000 * 5 : (i + 1) * 32000 * 5])

    return np.vstack(audios)


def transform_test_batch(audios, params):
    specs = []
    sr = 32000

    for sound in audios:
        melspec = librosa.feature.melspectrogram(sound, sr=sr, **params)
        melspec = librosa.power_to_db(melspec)
        image = melspec.astype(np.float32)

        delta = librosa.feature.delta(image)
        accelerate = librosa.feature.delta(image, order=2)
        image = np.stack([image, delta, accelerate], axis=0)
        image = image.astype(np.float32) / 100.0
        specs.append(np.expand_dims(image, 0))

    return np.vstack(specs)


def prepare_test(files, meta, le_encoder, melspectrogram_parameters, num_classes=264):
    inputs = []
    targets = []
    for file_ in files:
        batch = load_test_batch(file_)
        batch = transform_test_batch(batch, melspectrogram_parameters)
        print(batch.shape)
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

        target = np.zeros((len(test), num_classes + 1))

        for i, values in enumerate(test["labels"].values):
            if len(values) == 0:
                target[i, -1] = 1
            else:
                target[i, values] = 1

        print(file_, target[:, :num_classes].sum())

        inputs.append(batch)
        targets.append(target)

    return {"spect": np.vstack(inputs), "targets": np.vstack(targets)}
