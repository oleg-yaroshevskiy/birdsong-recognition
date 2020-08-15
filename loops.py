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


def mixup(data, targets, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    # indices = torch.randperm(data.size(0))

    # data = data[indices]
    # targets = targets[indices]
    half = data.size(0) // 2

    lam = np.random.beta(alpha, alpha)
    data = lam * data[:half] + (1 - lam) * data[half:]
    targets = targets[:half] + targets[half:]
    targets = torch.clamp(targets, 0, 1)

    return data, targets


def train_fn(train_loader, model, optimizer, scheduler_warmup, loss_fn, device, epoch):
    total_loss = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0
    batch_idx = 0

    t = tqdm(train_loader)
    for step, d in enumerate(t):
        # try:
        if scheduler_warmup is not None and epoch < 3:
            scheduler_warmup.step()

        spect = d["spect"].to(device)
        # half = spect.size(0) // 2
        # spect, mix_targets = mixup(spect, onehot(d["target"], None, 264, smoothing=0.).to(device))
        outputs = model(spect)

        loss = loss_fn(
            outputs, onehot(d["target"], None, 264, smoothing=0.0).to(device)
        )
        loss.backward()

        accumulated_loss += loss
        if batch_idx % 1 == 0 and batch_idx > 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss = 0
        batch_idx += 1

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        acc, n_position = get_position_accuracy(
            outputs, d["target"].to(device)
        )  # [:half])

        total_loss.update(loss.item(), n_position)
        accuracies.update(acc, n_position)

        t.set_description(
            f"Train E:{epoch+1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}"
        )
        # except Exception as e:
        #     print("error:", e)

    wandb.log(
        {
            "train mAP": accuracies.avg,
            "train loss": total_loss.avg,
            "learning rate": get_learning_rate(optimizer),
        },
        step=epoch,
    )

    return total_loss.avg


def valid_fn(valid_loader, model, loss_fn, device, epoch):
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
                outputs, onehot(d["target"], None, 264, smoothing=0.0).to(device)
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
    targets_all = torch.cat(targets_all, dim=0)
    targets_secondary_all = torch.cat(targets_secondary_all, dim=0)

    best_score = 0.0
    best_threshold = 0.0
    scores_by_t = {}
    for t in np.linspace(0.05, 0.95, 19):
        score = get_f1_micro(
            outputs_all, onehot(targets_all, None, 264, smoothing=0.0), t
        )
        if score > best_score:
            best_score = score
            best_threshold = t
        scores_by_t[round(t, 2)] = round(score, 4)
    print("f1 main targets:", scores_by_t)

    best_score_2 = 0.0
    best_threshold_2 = 0.0
    # print("f1 secondary targets:")
    scores_by_t = {}
    for t in np.linspace(0.05, 0.95, 19):
        score = get_f1_micro(
            outputs_all,
            onehot(targets_all, targets_secondary_all, 264, smoothing=0.0).to(device),
            t,
        )
        if score > best_score_2:
            best_score_2 = score
            best_threshold_2 = t
        # print(f"Threshold: {t:.2f}, f1: {score:.4f})")
        scores_by_t[round(t, 2)] = round(score, 4)
    print("f1 secondary targets:", scores_by_t)

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


def test_fn(model, loss_fn, device, samples, epoch):
    model.eval()
    scores = []

    with torch.no_grad():
        outputs = []
        for batch in torch.utils.data.DataLoader(
            samples["spect"], batch_size=16, shuffle=False
        ):
            # print(batch.shape)
            output = model(batch.to(device))
            outputs.append(output)

    outputs = torch.cat(outputs, dim=0)
    loss = loss_fn(
        outputs, torch.from_numpy(samples["targets"]).to(device)[:, :264].float()
    )

    best_score = 0.0
    best_threshold = 0.0
    scores_by_t = {}
    for t in np.linspace(0.05, 0.95, 19):
        score = get_f1_micro_nocall(outputs, samples["targets"], t)
        if score > best_score:
            best_score = score
            best_threshold = t
        # print(f"Threshold: {t:.2f}, f1: {score:.4f})")
        scores_by_t[round(t, 2)] = round(score, 4)

    print("test f1 targets:", scores_by_t)

    wandb.log(
        {
            "test f1 (best)": best_score,
            "test threshold (best)": best_threshold,
            "test loss": loss.item(),
        },
        step=epoch,
    )

    return best_score
