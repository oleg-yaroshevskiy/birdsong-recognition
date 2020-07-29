from utils import AverageMeter, get_position_accuracy
from tqdm import tqdm

def train_fn(train_loader, model, optimizer, loss_fn, device, epoch):
    total_loss = AverageMeter()
    accuracies = AverageMeter()

    model.train()

    t = tqdm(train_loader)
    for step, d in enumerate(t):

        spect = d["spect"].to(device)
        targets = d["target"].to(device)

        outputs = model(spect)

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, n_position = get_position_accuracy(outputs, targets)

        total_loss.update(loss.item(), n_position)
        accuracies.update(acc, n_position)

        t.set_description(
            f"Train E:{epoch+1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}"
        )

    return total_loss.avg


def valid_fn(valid_loader, model, device, epoch):
    total_loss = AverageMeter()
    accuracies = AverageMeter()

    model.eval()

    t = tqdm(valid_loader)
    for step, d in enumerate(t):

        with torch.no_grad():

            spect = d["spect"].to(device)
            targets = d["target"].to(device)

            outputs = model(spect)

            loss = loss_fn(outputs, targets)

            acc, n_position = get_position_accuracy(outputs, targets)

            total_loss.update(loss.item(), n_position)
            accuracies.update(acc, n_position)

            t.set_description(
                f"Eval E:{epoch+1} - Loss:{total_loss.avg:0.4f} - Acc:{accuracies.avg:0.4f}"
            )

    return total_loss.avg, accuracies.avg
