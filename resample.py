import numpy as np
import librosa
import glob
from multiprocessing import Pool
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

files = glob.glob("../input/train_audio/*/*mp3")
print("Total files", len(files))


def to_numpy(file):
    samples, _ = librosa.load(file, sr=32000)
    np.save(file.replace("mp3", "npy"), samples)


with Pool(12) as p:
    _ = list(tqdm(p.imap(to_numpy, files), total=len(files)))
