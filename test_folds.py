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


models = []
for fold in range(args.folds):
    model, loss_fn = get_model_loss(args)
    model.load_state_dict(torch.load(f"../models/cnn14_att_128_15s_sec_sm0.2_light_augm_xeno/fold_{fold}_test_2.pth"))
    models.append(model)

test_f1_1 = test_folds_fn(models, loss_fn, device, test_samples_1, "", args)
test_f1_2 = test_folds_fn(
    models, loss_fn, device, test_samples_2, " extended", args
)