from utils import AverageMeter, get_position_accuracy, get_learning_rate
from tqdm import tqdm
import torch
import wandb

def onehot(targets, targets_secondary, num_classes, smoothing=0.):
    size = targets.size(0)
    one_hot = torch.zeros(size, num_classes)
    one_hot.fill_(smoothing / (num_classes - 1))
    #one_hot[targets_secondary.bool()] = 1-smoothing
    one_hot[torch.arange(size), targets] = 1-smoothing

    return one_hot

def train_fn(train_loader, model, optimizer, scheduler_warmup, loss_fn, device, epoch):
    total_loss = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0
    batch_idx = 0

    t = tqdm(train_loader)
    for step, d in enumerate(t):
        #try:
        if scheduler_warmup is not None and epoch < 3:
            scheduler_warmup.step()

        spect = d["spect"].to(device)

        outputs = model(spect)

        loss = loss_fn(outputs, onehot(d["target"], d["target_secondary"], 264, smoothing=0.).to(device))
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

        acc, n_position = get_position_accuracy(outputs, d["target"].to(device))

        total_loss.update(loss.item(), n_position)
        accuracies.update(acc, n_position)

        t.set_description(
            f"Train E:{epoch+1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}"
        )
        # except Exception as e:
        #     print("error:", e)

    wandb.log({
        "train mAP": accuracies.avg, 
        "train loss": total_loss.avg,
        "learning rate": get_learning_rate(optimizer)}, step=epoch)

    return total_loss.avg


def valid_fn(valid_loader, model, loss_fn, device, epoch):
    total_loss = AverageMeter()
    accuracies = AverageMeter()

    model.eval()

    t = tqdm(valid_loader)
    for step, d in enumerate(t):

        with torch.no_grad():

            spect = d["spect"].to(device)
            targets = d["target"].to(device)

            outputs = model(spect)

            loss = loss_fn(outputs, onehot(d["target"], d["target_secondary"], 264, smoothing=0.).to(device))

            acc, n_position = get_position_accuracy(outputs, targets)

            total_loss.update(loss.item(), n_position)
            accuracies.update(acc, n_position)

            t.set_description(
                f"Eval E:{epoch+1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}"
            )

    wandb.log({"valid mAP": accuracies.avg, "valid loss": total_loss.avg}, step=epoch)

    return total_loss.avg, accuracies.avg
