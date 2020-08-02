import pretrainedmodels
from torch import nn
import torch.nn.functional as F
from args import args


class ResNet18(nn.Module):
    def __init__(self, pretrained):
        super(ResNet18, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__["resnet18"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet18"](pretrained=None)

        self.l0 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.l0(x)

        return x
