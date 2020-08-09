import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from collections import defaultdict
from itertools import chain
import math


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
    # print(logits.shape)
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
