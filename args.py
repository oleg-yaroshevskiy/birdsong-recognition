import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ROOT_PATH", type=str, default="../input/train_audio")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=16)

parser.add_argument("--model", type=str, default="se50")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)

parser.add_argument("--max_duration", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.0009)
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--eps", type=float, default=1e-8)

parser.add_argument("--melspectrogram_parameters_n_mels", type=int, default=128)
parser.add_argument("--melspectrogram_parameters_fmin", type=int, default=20)
parser.add_argument("--melspectrogram_parameters_fmax", type=int, default=16000)

args = parser.parse_args()
print("Initial arguments", args)

args.__dict__["betas"] = (0.9, 0.999)
args.__dict__["num_classes"] = 264
args.__dict__["sample_rate"] = 32000
args.__dict__["melspectrogram_parameters"] = {"n_mels": 128, "fmin": 20, "fmax": 16000}