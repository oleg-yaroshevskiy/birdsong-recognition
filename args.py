import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ROOT_PATH", type=str, default="../input/")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=16)

parser.add_argument("--model", type=str, default="b4")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=75)

parser.add_argument("--max_duration", type=int, default=5)
parser.add_argument("--warmup", type=int, default=500)  # < 1 epoch
parser.add_argument("--lr_stop", type=str, default="1e-5")
parser.add_argument("--lr_base", type=str, default="1e-3")
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--lr_patience", type=int, default=2)
parser.add_argument("--lr_drop_rate", type=str, default="np.sqrt(0.1)")
parser.add_argument("--opt_lookahead", type=str, default="False")
parser.add_argument("--batch_accumulation", type=int, default=2)

parser.add_argument("--nmels", type=int, default=128)

parser.add_argument("--smoothing", type=float, default=0.2)
parser.add_argument("--mixup", type=float, default=0.0)
parser.add_argument("--secondary", type=str, default="True")
parser.add_argument("--sigmoid", type=str, default="False")

parser.add_argument("--add_xeno", type=str, default="True")

parser.add_argument("--augm_noise_or_bg", type=float, default=0.66)
parser.add_argument("--augm_bg_prob", type=float, default=0.5)
parser.add_argument("--augm_vol_prob", type=float, default=1.0)
parser.add_argument("--augm_noise_prob", type=float, default=0.5)
parser.add_argument("--augm_spec_prob", type=float, default=0.33)
parser.add_argument("--augm_low_pass", type=float, default=0.)

args = parser.parse_args()

for arg in ["opt_lookahead", "add_xeno", "secondary", "sigmoid"]:
    args.__dict__[arg] = args.__dict__[arg] == "True"

for arg in ["lr_base", "lr_drop_rate", "lr_stop"]:
    args.__dict__[arg] = eval(args.__dict__[arg])

args.__dict__["betas"] = (0.9, 0.999)
args.__dict__["num_classes"] = 264
args.__dict__["sample_rate"] = 32000
args.__dict__["melspectrogram_parameters"] = {
    "n_mels": args.nmels,
    "fmin": 20,
    "fmax": 16000,
    "hop_length": 320,
    "n_fft": 1024
    # default hop_length=512 n_fft=2048 n_mels=128
    # TODO: re-run b4 with those
}

print("Initial arguments", args)
