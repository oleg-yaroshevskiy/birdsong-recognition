class args:

    ROOT_PATH = "../input/train_audio"

    num_classes = 10
    max_duration = 5  # seconds

    sample_rate = 32000

    img_height = 128
    img_width = 313

    batch_size = 16
    num_workers = 4
    epochs = 2

    lr = 0.0009
    wd = 1e-5
    momentum = 0.9
    eps = 1e-8
    betas = (0.9, 0.999)

    melspectrogram_parameters = {"n_mels": 128, "fmin": 20, "fmax": 16000}
