import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BirdDataset
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
seed_all(args)

model_directory = "{}{}".format(args.model, "_" + args.name if args.name else "")
model_directory = f"../models/{model_directory}/"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

train = pd.read_csv("../input/train.csv")
train_le = LabelEncoder().fit(train.ebird_code.values)

test_samples_1, test_samples_2 = get_test_samples(train_le, args)

for checkpoint_set in ["", "_test_1", "_test_2", "_test_2_05"]:
    models = []
    for fold in range(args.folds):
        model, loss_fn = get_model_loss(args, pretrained=False)
        model.load_state_dict(torch.load(f"../models/b4_128_xeno_light_mixup/fold_{fold}{checkpoint_set}.pth"))
        models.append(model)

    print("Checkpoint set:", checkpoint_set)
    test_f1_1 = test_folds_fn(models, loss_fn, device, test_samples_1, "", args)
    test_f1_2 = test_folds_fn(
        models, loss_fn, device, test_samples_2, " extended", args
    )
    print()
    print()