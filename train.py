import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BirdDataset
from args import args
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from neural import ResNet18
from loops import train_fn, valid_fn
import wandb
import random
from transformers import get_constant_schedule_with_warmup
from utils import (
    get_learning_rate, isclose, Lookahead
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")

# encoding features TODO: should be fixed for inference too
train_le = LabelEncoder().fit(train.ebird_code.values)
train["ebird_label"] = train_le.transform(train.ebird_code.values)

mapping = pd.Series(train.ebird_code.values, index=train.primary_label).to_dict()
train["ebird_label_secondary"] = train.secondary_labels.apply(
    lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
)

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

    ### Dataset / Dataloader ###
    train_df = train.loc[t_idx]
    train_dataset = BirdDataset(df=train_df)

    valid_df = train.loc[v_idx]
    valid_dataset = BirdDataset(df=valid_df, valid=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32)
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32),
    )

    loss_fn = nn.BCEWithLogitsLoss()
    model = torch.hub.load(
        'rwightman/gen-efficientnet-pytorch', 
        'tf_efficientnet_%s_ns' % args.model, 
        pretrained=True,
        num_classes = args.num_classes
    ).cuda()

    ### OPTIMIZATION ###
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_base,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.wd,
    )
    if args.opt_lookahead:
        optimizer = Lookahead(optimizer)
    scheduler_warmup = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_drop_rate, patience=args.lr_patience, threshold=0
    )

    best_acc = 0

    for epoch in range(args.epochs):

        train_loss = train_fn(train_loader, model, optimizer, scheduler_warmup, loss_fn, device, epoch)
        valid_loss, valid_acc = valid_fn(valid_loader, model, loss_fn, device, epoch)
        
        print(f"Fold {fold} ** Epoch {epoch+1} **==>** Accuracy = {valid_acc}")

        if valid_acc > best_acc:
            torch.save(model.state_dict(), f"fold_{fold}.pth")
            wandb.save(f"fold_{fold}.pth")
            best_acc = valid_acc

        scheduler.step(valid_acc)

        if isclose(optimizer, args.lr_stop):
            print("Exit on LR")
            break

    wandb.join()
