from torch import nn as nn
from torch.nn import functional as F


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, self.inplace)


def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argmument interface
class Tanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace: bool = False):
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)
