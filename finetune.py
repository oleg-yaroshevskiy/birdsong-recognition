import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BirdDataset, DcaseDataset
from args import args
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from neural import get_optimizer_scheduler, get_model_loss, PANNsLoss
from loops import finetune_fn
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

train = pd.concat([
    pd.read_csv("../input/dcase/ff1010bird_metadata_2018.csv"),
    #pd.read_csv("../input/dcase/BirdVoxDCASE20k_csvpublic.csv"),
    #pd.read_csv("../input/dcase/warblrb10k_public_metadata_2018.csv")
], axis=0)


run_id = "finetuning_{}{}".format(args.model, "_" + args.name if args.name else "")
wandb.init(
    config=args, project="birdsong", name=run_id, id=run_id, reinit=True,
)

train_dataset = DcaseDataset(df=train, args=args)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=args.num_workers,
    worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2 ** 32),
)

model, loss_fn = get_model_loss(args)
# model.load_state_dict(torch.load("../models/cnn14_att_64_15sec_gpu_lp0.33_nocall/fold_0_test_2.pth"))
optimizer, scheduler_warmup, scheduler = get_optimizer_scheduler(model, args)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.5, 5)
#loss_fn = torch.nn.BCELoss()

for epoch in range(args.epochs):
    train_loss = finetune_fn(
        train_loader,
        model,
        optimizer,
        scheduler_warmup,
        loss_fn,
        device,
        epoch,
        args,
    )
    #scheduler.step()

wandb.join()
