import numpy as np
import librosa
import glob
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")

files = glob.glob("../input/xeno-carlo/*/*mp3")
print("Total files", len(files))


def to_numpy(file):
    try:
        samples, _ = librosa.load(file, sr=32000)
        np.save(file.replace("mp3", "npy"), samples)
        os.remove(file)
    except:
        print(file)


with Pool(8) as p:
    _ = list(tqdm(p.imap(to_numpy, files), total=len(files)))
