import torch
import numpy as np
import torch.nn.functional as F


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def reduce_fn(vals):
    return sum(vals) / len(vals)


class AverageMeter(object):
    """Computes and stores the average and current values"""

    def __init__(self):
        self.reset()

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_position_accuracy(logits, labels):
    predictions = np.argmax(F.softmax(logits, dim=1).cpu().data.numpy(), axis=1)
    labels = labels.cpu().data.numpy()
    total_num = 0
    sum_correct = 0
    for i in range(len(labels)):
        if labels[i] >= 0:
            total_num += 1
            if predictions[i] == labels[i]:
                sum_correct += 1
    if total_num == 0:
        total_num = 1e-7
    return np.float32(sum_correct) / total_num, total_num
