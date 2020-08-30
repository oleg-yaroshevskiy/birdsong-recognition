import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from collections import defaultdict
from itertools import chain
import math
import random
from sklearn.metrics import f1_score

def seed_all(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


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


def get_position_accuracy(logits, labels, threshold=None):
    if threshold is None:
        predictions = np.argmax(F.softmax(logits, dim=1).cpu().data.numpy(), axis=1)
    else:
        predictions = logits.sigmoid().cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    total_num = 0
    sum_correct = 0

    for i in range(len(labels)):
        if labels[i] >= 0:
            total_num += 1
            if threshold is None:
                if predictions[i] == labels[i]:
                    sum_correct += 1
            else:
                if predictions[i][labels[i]] >= threshold:
                    sum_correct += 1

    if total_num == 0:
        total_num = 1e-7

    return np.float32(sum_correct) / total_num, total_num


def get_accuracy(logits, labels):
    #print(logits)
    #print(labels)
    return ((logits >= 0.5).long() == labels).float().mean(), len(logits)

def get_f1_micro(logits, labels, threshold=None):
    probs = logits.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    return f1_score(labels, probs > threshold, average="samples")


def get_f1_micro_nocall(logits, labels, threshold=None, num_classes=264):
    probs = logits.cpu().data.numpy() > threshold

    new_probs = np.zeros((probs.shape[0], num_classes + 1))
    new_probs[:, :num_classes] = probs.astype(int)[:, :num_classes]

    # nocall if all zeros
    for i in range(probs.shape[0]):
        if probs[i].max() == 0:
            new_probs[i, -1] = 1

    return f1_score(labels, new_probs > threshold, average="samples")


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def isclose(optimizer, value):
    return math.isclose(get_learning_rate(optimizer), value, abs_tol=1e-9)


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
