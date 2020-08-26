import numpy as np
import librosa
import glob
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")

files = glob.glob("../input/dcase/wav/*.wav")
print("Total files", len(files))


def to_numpy(file):
    try:
        samples, _ = librosa.load(file, sr=32000)
        np.save(file.replace(".wav", ".npy"), samples)
        os.remove(file)
    except Exception as e:
        print(file, e)


with Pool(32) as p:
    _ = list(tqdm(p.imap(to_numpy, files), total=len(files)))
