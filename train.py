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
from utils import get_learning_rate, isclose, seed_all, RankOrderedList
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
# train_nocall["ebird_label"] = train_le.transform(train_nocall.ebird_code.values)

kfold = StratifiedKFold(n_splits=5)
for fold, (t_idx, v_idx) in enumerate(
    kfold.split(train.filename.values, train.ebird_code.values)
):
    if args.fold is not None:
        if fold != args.fold:
            continue
    run_id = "{}_f{}{}".format(args.model, fold, "_" + args.name if args.name else "")
    wandb.init(
        config=args, project="birdsong", name=run_id, id=run_id, reinit=True,
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
   # model.load_state_dict(torch.load("../models/cnn14_att_64_15sec_gpu_lp0.33_nocall/fold_0_test_2.pth"))
    optimizer, scheduler_warmup, scheduler = get_optimizer_scheduler(model, args)

    best_acc = 0
    best_test_1 = 0
    best_test_2 = 0
    best_test_2_05 = 0
    ol_test_1 = RankOrderedList()
    ol_test_2 = RankOrderedList()

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
        test_f1_1, _ = test_fn(model, loss_fn, device, test_samples_1, epoch, "", args)
        test_f1_2, test_f1_2_05 = test_fn(
            model, loss_fn, device, test_samples_2, epoch, " extended", args
        )

        print(f"Fold {fold} ** Epoch {epoch+1} **==>** Accuracy = {valid_acc:.4f}")

        # SAVE MODELS
        if valid_acc > best_acc:
            torch.save(model.state_dict(), f"{model_directory}/fold_{fold}.pth")
            wandb.save(f"{model_directory}/fold_{fold}.pth")
            best_acc = valid_acc

        if test_f1_1 > best_test_1:
            torch.save(model.state_dict(), f"{model_directory}/fold_{fold}_test_1.pth")
            wandb.save(f"{model_directory}/fold_{fold}_test_1.pth")
            best_test_1 = test_f1_1
        
        ol_test_1.insert(test_f1_1, lambda rank: torch.save(model.state_dict(), f"{model_directory}/fold_{fold}_test_1_r{rank}.pth"))

        if test_f1_2 > best_test_2:
            torch.save(model.state_dict(), f"{model_directory}/fold_{fold}_test_2.pth")
            wandb.save(f"{model_directory}/fold_{fold}_test_2.pth")
            best_test_2 = test_f1_2

        ol_test_2.insert(test_f1_2, lambda rank: torch.save(model.state_dict(), f"{model_directory}/fold_{fold}_test_2_r{rank}.pth"))

        if test_f1_2_05 > best_test_2_05:
            torch.save(model.state_dict(), f"{model_directory}/fold_{fold}_test_2_05.pth")
            wandb.save(f"{model_directory}/fold_{fold}_test_2_05.pth")
            best_test_2_05 = test_f1_2_05

        scheduler.step(valid_acc)

        if isclose(optimizer, args.lr_stop):
            print("Exit on LR")
            break
        
        if args.turn_off_augs:
            if isclose(optimizer, args.lr_stop / args.lr_drop_rate):
                print("Dropping Augs")
                model.spec_augm_prob = 0.
                train_dataset.aug = valid_dataset.aug
                
        

    wandb.join()
