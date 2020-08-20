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
from loops import train_fn, valid_fn, test_fn
import wandb
import random
from utils import get_learning_rate, isclose, seed_all
from test import get_test_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
seed_all(args)

train = pd.read_csv("../input/train.csv")
# train_nocall = pd.read_csv("../input/env/nocall.csv")
# train_nocall["folder"] = "env/audio/"

train_le = LabelEncoder().fit(train.ebird_code.values)
train["folder"] = "train_audio"
train["ebird_label"] = train_le.transform(train.ebird_code.values)
mapping = pd.Series(train.ebird_code.values, index=train.primary_label).to_dict()
train["ebird_label_secondary"] = train.secondary_labels.apply(
    lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
)
# train_nocall["ebird_label_secondary"] = train_nocall.secondary_labels.apply(
#     lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
# )

if args.add_xeno:
    aux_train = pd.read_csv("../input/train_extended.csv")
    aux_train["folder"] = "xeno-carlo"
    aux_train["ebird_label"] = train_le.transform(aux_train.ebird_code.values)
    aux_train["ebird_label_secondary"] = aux_train.secondary_labels.apply(
        lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
    )

test_samples_1, test_samples_2 = get_test_samples(train_le, args)
#train_nocall["ebird_label"] = train_le.transform(train_nocall.ebird_code.values)

kfold = StratifiedKFold(n_splits=5)
for fold, (t_idx, v_idx) in enumerate(
    kfold.split(train.filename.values, train.ebird_code.values)
):
    wandb.init(
        config=args,
        project="birdsong",
        name="{}_f{}{}".format(args.model, fold, "_" + args.name if args.name else ""),
        id="{}_f{}{}".format(args.model, fold, "_" + args.name if args.name else ""),
        reinit=True,
    )

    # DataLoaders
    train_df = train.loc[t_idx]
    valid_df = train.loc[v_idx]
    if args.add_xeno:
        train_df = pd.concat([train_df, aux_train], axis=0)

    # train_df = pd.concat([train_df, train_nocall], axis=0)

    train_dataset = BirdDataset(df=train_df, args=args)
    valid_dataset = BirdDataset(df=valid_df, args=args, valid=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2 ** 32),
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model, loss_fn = get_model_loss(args)
    optimizer, scheduler_warmup, scheduler = get_optimizer_scheduler(model, args)

    best_acc = 0
    best_test_1 = 0
    best_test_2 = 0

    for epoch in range(args.epochs):
        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            scheduler_warmup,
            loss_fn,
            device,
            epoch,
            args,
        )
        valid_loss, valid_acc = valid_fn(
            valid_loader, model, loss_fn, device, epoch, args
        )
        test_f1_1 = test_fn(model, loss_fn, device, test_samples_1, epoch, "", args)
        test_f1_2 = test_fn(model, loss_fn, device, test_samples_2, epoch, " extended", args)

        print(f"Fold {fold} ** Epoch {epoch+1} **==>** Accuracy = {valid_acc:.4f}")

        # SAVE MODELS
        if valid_acc > best_acc:
            torch.save(model.state_dict(), f"../models/fold_{fold}.pth")
            wandb.save(f"../models/fold_{fold}.pth")
            best_acc = valid_acc

        if test_f1_1 > best_test_1:
            torch.save(model.state_dict(), f"../models/fold_{fold}_test_1.pth")
            wandb.save(f"../models/fold_{fold}_test_1.pth")
            best_test_1 = test_f1_1

        if test_f1_2 > best_test_2:
            torch.save(model.state_dict(), f"../models/fold_{fold}_test_2.pth")
            wandb.save(f"../models/fold_{fold}_test_2.pth")
            best_test_2 = test_f1_2

        scheduler.step(valid_acc)

        if isclose(optimizer, args.lr_stop):
            print("Exit on LR")
            break

    wandb.join()
