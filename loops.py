from utils import (
    AverageMeter,
    get_position_accuracy,
    get_learning_rate,
    get_f1_micro,
    get_f1_micro_nocall,
)
from tqdm import tqdm
import torch
import wandb
import numpy as np


def onehot(targets, targets_secondary, num_classes, smoothing=0.0):
    size = targets.size(0)
    one_hot = torch.zeros(size, num_classes)
    one_hot.fill_(smoothing / (num_classes - 1))

    if targets_secondary is not None:
        one_hot[targets_secondary.bool()] = 1 - smoothing

    one_hot[torch.arange(size), targets] = 1 - smoothing

    return one_hot


def mixup(data, targets, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    half = data.size(0) // 2

    lam = np.random.beta(alpha, alpha)
    data = lam * data[:half] + (1 - lam) * data[half:]
    targets = lam * targets[:half] + (1 - lam) * targets[half:]

    return data, targets


def train_fn(
    train_loader, model, optimizer, scheduler_warmup, loss_fn, device, epoch, args
):
    total_loss = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0
    batch_idx = 0

    t = tqdm(train_loader)
    for step, d in enumerate(t):
        spect = d["spect"].to(device)
        targets = onehot(
            d["target"],
            d["target_secondary"] if args.secondary else None,
            args.num_classes,
            smoothing=args.smoothing,
        ).to(device)

        if args.mixup > 0:
            half = spect.size(0) // 2
            spect, targets = mixup(spect, targets, args.mixup)

        outputs = model(spect)
        loss = loss_fn(outputs, targets)
        loss.backward()

        accumulated_loss += loss
        if batch_idx % args.batch_accumulation == 0 and batch_idx > 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss = 0
        batch_idx += 1

        if scheduler_warmup is not None and epoch < 3:
            scheduler_warmup.step()


        acc, n_position = get_position_accuracy(
            outputs, d["target"].to(device)[: outputs.size(0)]
        )

        total_loss.update(loss.item(), n_position)
        accuracies.update(acc, n_position)

        t.set_description(
            f"Train E:{epoch+1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}"
        )
    wandb.log(
        {
            "train mAP": accuracies.avg,
            "train loss": total_loss.avg,
            "learning rate": get_learning_rate(optimizer),
        },
        step=epoch,
    )

    return total_loss.avg


def valid_fn(valid_loader, model, loss_fn, device, epoch, args):
    total_loss = AverageMeter()
    accuracies = AverageMeter()
    model.eval()

    t = tqdm(valid_loader)
    outputs_all, targets_all, targets_secondary_all = [], [], []

    for step, d in enumerate(t):
        with torch.no_grad():

            spect = d["spect"].to(device)
            targets = d["target"].to(device)

            outputs = model(spect)

            loss = loss_fn(
                outputs,
                onehot(d["target"], None, args.num_classes, smoothing=0.0).to(device),
            )

            acc, n_position = get_position_accuracy(outputs, targets)

            total_loss.update(loss.item(), n_position)
            accuracies.update(acc, n_position)

            t.set_description(
                f"Eval E:{epoch+1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}"
            )

            outputs_all.append(outputs.detach())
            targets_all.append(targets.detach())
            targets_secondary_all.append(d["target_secondary"].to(device))

    outputs_all = torch.cat(outputs_all, dim=0)
    if args.sigmoid:
        outputs_all = outputs_all.sigmoid()
    targets_all = torch.cat(targets_all, dim=0)
    targets_secondary_all = torch.cat(targets_secondary_all, dim=0)

    best_score = 0.0
    best_threshold = 0.0
    scores_by_t = []
    for t in np.linspace(0.05, 0.95, 19):
        score = get_f1_micro(
            outputs_all, onehot(targets_all, None, args.num_classes, smoothing=0.0), t
        )
        if score > best_score:
            best_score = score
            best_threshold = t
        scores_by_t.append(round(score, 4))
    print(f"valid f1 scores:", scores_by_t)

    best_score_2 = 0.0
    best_threshold_2 = 0.0
    scores_by_t = []
    for t in np.linspace(0.05, 0.95, 19):
        score = get_f1_micro(
            outputs_all,
            onehot(targets_all, targets_secondary_all, args.num_classes, smoothing=0.0).to(device),
            t,
        )
        if score > best_score_2:
            best_score_2 = score
            best_threshold_2 = t
        scores_by_t.append(round(score, 4))
    print(f"valid f1 scores:", scores_by_t)

    wandb.log(
        {
            "valid mAP": accuracies.avg,
            "valid f1 (best)": best_score,
            "valid threshold (best)": best_threshold,
            "valid f1 (best) secondary": best_score_2,
            "valid threshold (best) secondary": best_threshold_2,
            "valid loss": total_loss.avg,
        },
        step=epoch,
    )

    return total_loss.avg, accuracies.avg


def test_fn(model, loss_fn, device, samples, epoch, key, args):
    model.eval()
    scores = []

    with torch.no_grad():
        outputs = []
        for batch in torch.utils.data.DataLoader(
            samples["spect"], batch_size=16, shuffle=False
        ):
            output = model(batch.to(device))
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0)

    # loss = loss_fn(
    #     outputs,
    #     torch.from_numpy(samples["targets"]).to(device)[:, : outputs.size(1)].float(),
    # )
    if args.sigmoid:
        outputs = outputs.sigmoid()

    best_score, best_threshold = 0.0, 0.0
    scores_by_t = []

    for t in np.linspace(0.05, 0.95, 19):
        score = get_f1_micro_nocall(outputs, samples["targets"], t)
        if score > best_score:
            best_score = score
            best_threshold = t

        scores_by_t.append(round(score, 4))

    print(f"test{key} f1 scores:", scores_by_t)

    wandb.log(
        {
            f"test f1 (best){key}": best_score,
            f"test threshold (best){key}": best_threshold,
            #f"test loss{key}": loss.item(),
        },
        step=epoch,
    )

    return best_score
