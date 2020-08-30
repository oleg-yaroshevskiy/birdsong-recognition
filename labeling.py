import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BirdDataset, BirdLabelingDataset
from args import args
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from neural import get_optimizer_scheduler, get_model_loss
from loops import train_fn, valid_fn, test_folds_fn
import wandb
import random
from utils import get_learning_rate, isclose, seed_all
from test import get_test_samples
import os
from tqdm import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
seed_all(args)

model_directory = "{}{}".format(args.model, "_" + args.name if args.name else "")
model_directory = f"../models/{model_directory}/"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

train = pd.read_csv("../input/train.csv")
train_le = LabelEncoder().fit(train.ebird_code.values)
train["folder"] = "train_audio"
train["ebird_label"] = train_le.transform(train.ebird_code.values)
mapping = pd.Series(train.ebird_code.values, index=train.primary_label).to_dict()
train["ebird_label_secondary"] = train.secondary_labels.apply(
    lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
)
train_dataset = BirdLabelingDataset(df=train, args=args)

train_loader = tqdm(DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
    num_workers=args.num_workers,
    collate_fn=lambda x: x[0],
    worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2 ** 32),
))

#for checkpoint_set in ["", "_test_1", "_test_2", "_test_2_05"]:
models = []
for fold in range(args.folds):
    model, loss_fn = get_model_loss(args, pretrained=False)
    model.load_state_dict(torch.load(f"../models/r38_128_10_xeno_light/fold_{fold}.pth"))
    model.eval()
    models.append(model)

with torch.no_grad():
    for batch in train_loader:
        outputs = []
        for model in models:
            o_ = []
            data = batch["spect"].to(device)
            batches = math.ceil(data.shape[0] / 64)

            for i in range(batches):
                o_.append(model(data[i * 64 : (i + 1) * 64]))

            output = torch.cat(o_, dim=0)

            if args.sigmoid:
                output = output.sigmoid()
            outputs.append(output)

        outputs = torch.stack(outputs, 0).mean(axis=0).detach().cpu().numpy()
        
        np.save("../labels/r38_128_10_xeno_light/{}_frame_targets.npy".format(batch["filename"]), outputs[:, batch["target"]])
        if len(batch["target_secondary"]) > 0:
            np.save("../labels/r38_128_10_xeno_light/{}_frame_secondary_targets.npy".format(batch["filename"]), outputs[:, batch["target_secondary"]])