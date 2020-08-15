import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import BirdDataset
from args import args
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from neural import ResNet18, Cnn14_DecisionLevelAtt, PANNsLoss
from loops import train_fn, valid_fn, test_fn
import wandb
import random
from transformers import get_constant_schedule_with_warmup
from utils import get_learning_rate, isclose, Lookahead
from collections import OrderedDict
from test import prepare_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

train = pd.read_csv("../input/train.csv")
train["folder"] = "train_audio"

aux_train = pd.read_csv("../input/train_extended.csv")
aux_train["folder"] = "xeno-carlo"

test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")

# encoding features TODO: should be fixed for inference too
train_le = LabelEncoder().fit(train.ebird_code.values)
train["ebird_label"] = train_le.transform(train.ebird_code.values)

mapping = pd.Series(train.ebird_code.values, index=train.primary_label).to_dict()
train["ebird_label_secondary"] = train.secondary_labels.apply(
    lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
)

aux_train["ebird_label"] = train_le.transform(aux_train.ebird_code.values)
aux_train["ebird_label_secondary"] = aux_train.secondary_labels.apply(
    lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
)
test_samples = prepare_test(
    [
        "../input/test/BLKFR-10-CPL_20190611_093000.pt540.mp3",
        "../input/test/ORANGE-7-CAP_20190606_093000.pt623.mp3",
        #"../input/test/SSW49_20170520.wav",
        #"../input/test/SSW50_20170819.wav",
        "../input/test/SSW51_20170819.wav",
        #"../input/test/SSW52_20170429.wav",
        "../input/test/SSW53_20170513.wav",
        "../input/test/SSW54_20170610.wav"
    ],
    pd.read_csv("../input/test/merged_summary.csv"),
    train_le,
    args.melspectrogram_parameters,
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
    if args.add_xeno:
        train_df = pd.concat([train_df, aux_train], axis=0)
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
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2 ** 32),
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if args.model == "cnn14_att":
        model = Cnn14_DecisionLevelAtt(args.num_classes).cuda()
        state = torch.load("Cnn14_DecisionLevelAtt_mAP=0.425.pth")["model"]
        new_state_dict = OrderedDict()
        for k, v in state.items():
            if "att_block." in k and v.dim() != 0:
                print(k)
                new_state_dict[k] = v[: args.num_classes]
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

        loss_fn = PANNsLoss()
    elif args.model == "b4_att":
        from geffnet import tf_efficientnet_b4_ns
        model = tf_efficientnet_b4_ns(pretrained=True, num_classes=args.num_classes).cuda()
        loss_fn = PANNsLoss()
    else:
        model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "tf_efficientnet_%s_ns" % args.model,
            pretrained=True,
            num_classes=args.num_classes,
        ).cuda()
        
        loss_fn = torch.nn.BCEWithLogitsLoss()

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
        optimizer, num_warmup_steps=args.warmup
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_drop_rate,
        patience=args.lr_patience,
        threshold=0,
    )

    best_acc = 0
    best_test = 0

    for epoch in range(args.epochs):

        train_loss = train_fn(
            train_loader, model, optimizer, scheduler_warmup, loss_fn, device, epoch
        )
        valid_loss, valid_acc = valid_fn(valid_loader, model, loss_fn, device, epoch)
        test_f1 = test_fn(model, loss_fn, device, test_samples, epoch)
        print(f"Fold {fold} ** Epoch {epoch+1} **==>** Accuracy = {valid_acc:.4f}")

        if valid_acc > best_acc:
            torch.save(model.state_dict(), f"fold_{fold}.pth")
            wandb.save(f"fold_{fold}.pth")
            best_acc = valid_acc

        if test_f1 > best_test:
            torch.save(model.state_dict(), f"fold_{fold}_test.pth")
            wandb.save(f"fold_{fold}_test.pth")
            best_test = test_f1

        scheduler.step(valid_acc)

        if isclose(optimizer, args.lr_stop):
            print("Exit on LR")
            break

    wandb.join()
