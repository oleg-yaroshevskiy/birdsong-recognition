import numpy as np
import librosa
import glob
import tqdm as tqdm

wav_files = glob.glob("../input/env/filtered*/*.wav")
print("files found", len(wav_files))

audios = []
for file in tqdm.tqdm(wav_files):
    audios.append(librosa.load(file, 32000)[0])

audios = np.hstack(audios)
np.save("../input/env/bg2.npy", audios)