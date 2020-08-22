import pretrainedmodels
import torch
from torch import nn
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup
from utils import get_learning_rate, isclose, Lookahead
from collections import OrderedDict
import torchlibrosa
from blocks import (
    AttBlock,
    LogMel,
    ConvBlock,
    interpolate,
    pad_framewise_output,
    init_bn,
    init_layer,
)


def get_model_loss(args):
    if args.model == "cnn14_att":
        model = Cnn14_DecisionLevelAtt(args.num_classes, args).cuda()
        state = torch.load("Cnn14_DecisionLevelAtt_mAP=0.425.pth")["model"]

        new_state_dict = OrderedDict()
        for k, v in state.items():
            if ("att_block." in k) and v.dim() != 0:
                print(k)
                new_state_dict[k] = v[: args.num_classes]
            elif "bn0" in k and v.dim() != 0 and args.nmels == 128:
                new_state_dict[k] = torch.cat([v, v])
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

        loss_fn = PANNsLoss()

    elif args.model == "b4_att":
        from geffnet import tf_efficientnet_b4_ns

        model = tf_efficientnet_b4_ns(
            pretrained=True, num_classes=args.num_classes
        ).cuda()
        loss_fn = PANNsLoss()

    elif args.model == "b0_att":
        from geffnet import tf_efficientnet_b0_ns

        model = tf_efficientnet_b0_ns(
            pretrained=True, num_classes=args.num_classes
        ).cuda()
        loss_fn = PANNsLoss()

    else:
        model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "tf_efficientnet_%s_ns" % args.model,
            pretrained=True,
            num_classes=args.num_classes,
        ).cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss()
        args.__dict__["sigmoid"] = True

    return model, loss_fn


def get_optimizer_scheduler(model, args):
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
        verbose=True,
    )

    return optimizer, scheduler_warmup, scheduler


class Cnn14_DecisionLevelAtt(nn.Module):
    def __init__(self, classes_num, config):
        super(Cnn14_DecisionLevelAtt, self).__init__()
        self.interpolate_ratio = 32  # Downsampled ratio

        self.bn0 = nn.BatchNorm2d(config.nmels)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.att_block = AttBlock(2048, classes_num, activation="sigmoid")

        self.init_weight()

        self.logmel = LogMel(config.melspectrogram_parameters)
        self.spec_augm = torchlibrosa.augmentation.SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2 * (config.max_duration // 5),
            freq_drop_width=8,
            freq_stripes_num=2,
        )
        self.spec_augm_prob = config.augm_spec_prob

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, x):
        x = self.logmel(x)
        if self.training:
            x = torch.where(torch.rand(x.size(0)) > 0.33, x, self.spec_augm(x))

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, _, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # # Get framewise output
        # framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)

        # output_dict = {
        #     "framewise_output": framewise_output,
        #     "clipwise_output": clipwise_output,
        # }

        # print(clipwise_output.min(), clipwise_output.max())
        return clipwise_output, segmentwise_output


class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input_, target):
        input_ = torch.where(torch.isnan(input_), torch.zeros_like(input_), input_)
        input_ = torch.where(torch.isinf(input_), torch.zeros_like(input_), input_)
        input_ = torch.clamp(input_, min=0.0, max=1.0)

        target = target.float()

        return self.bce(input_, target)
