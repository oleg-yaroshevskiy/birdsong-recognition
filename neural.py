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
from geffnet import tf_efficientnet_b4_ns, tf_efficientnet_b0_ns, tf_efficientnet_b6_ns


def get_model_loss(args, pretrained=True):
    if args.model == "cnn14_att":
        model = Cnn14_DecisionLevelAtt(args.num_classes, args).cuda()
        if pretrained:
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
    elif args.model == "r38":
        model = ResNet38(args.num_classes, args).cuda()
        if pretrained:
            state = torch.load("ResNet38_mAP=0.434.pth")["model"]

            new_state_dict = OrderedDict()
            for k, v in state.items():
                if "bn0" in k and v.dim() != 0 and args.nmels == 128:
                    new_state_dict[k] = torch.cat([v, v])
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=False)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        args.__dict__["sigmoid"] = True

    elif args.model in ["b0", "b0_att", "b4", "b4_att", "b6"]:
        if "b0" in args.model:
            model = tf_efficientnet_b0_ns(
                args, pretrained=pretrained, num_classes=args.num_classes
            ).cuda()
        if "b6" in args.model:
            model = tf_efficientnet_b6_ns(
                args, pretrained=pretrained, num_classes=args.num_classes
            ).cuda()
        if "b4" in args.model:
            model = tf_efficientnet_b4_ns(
                args, pretrained=pretrained, num_classes=args.num_classes
            ).cuda()
        if "_att" in args.model:
            loss_fn = PANNsLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            args.__dict__["sigmoid"] = True

    elif args.model == "resnest":
        from resnest import resnest50

        model = resnest50(
            pretrained=pretrained, num_classes=args.num_classes
        ).cuda()
        loss_fn = PANNsLoss()

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
            mask = (torch.rand(x.size(0)) > 0.33).cuda()
            x = torch.where(
                torch.repeat_interleave(mask, x.size(2) * x.size(3)).reshape(x.size()),
                x, self.spec_augm(x))

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
        #segmentwise_output = segmentwise_output.transpose(1, 2)

        # # Get framewise output
        # framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)

        # output_dict = {
        #     "framewise_output": framewise_output,
        #     "clipwise_output": clipwise_output,
        # }

        # print(clipwise_output.min(), clipwise_output.max())
        return clipwise_output, segmentwise_output


def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNet38(nn.Module):
    def __init__(self, classes_num, config):
        super(ResNet38, self).__init__()

        self.bn0 = nn.BatchNorm2d(config.nmels)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.classifier = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

        self.logmel = LogMel(config.melspectrogram_parameters)
        self.spec_augm = torchlibrosa.augmentation.SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2 * (config.max_duration // 5),
            freq_drop_width=8,
            freq_stripes_num=2,
        )
        self.spec_augm_prob = config.augm_spec_prob

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.classifier)


    def forward(self, input):
        x = self.logmel(input)
        if self.training:
            mask = (torch.rand(x.size(0)) > 0.33).cuda()
            x = torch.where(
                torch.repeat_interleave(mask, x.size(2) * x.size(3)).reshape(x.size()),
                x, self.spec_augm(x))
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
    
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = self.classifier(x)

        return clipwise_output

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
