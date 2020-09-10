
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
train = pd.read_csv("../input/train.csv")

train_le = LabelEncoder().fit(train.ebird_code.values)
train["folder"] = "train_audio"
train["ebird_label"] = train_le.transform(train.ebird_code.values)
mapping = pd.Series(train.ebird_code.values, index=train.primary_label).to_dict()
train["ebird_label_secondary"] = train.secondary_labels.apply(
    lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
)

aux_train = pd.read_csv("../input/train_extended.csv")
aux_train["folder"] = "xeno-carlo"
aux_train["ebird_label"] = train_le.transform(aux_train.ebird_code.values)
aux_train["ebird_label_secondary"] = aux_train.secondary_labels.apply(
    lambda x: train_le.transform([mapping[xx] for xx in eval(x) if xx in mapping])
)
train_df = pd.concat([train, aux_train], axis=0)
all_dict = {}
for i, item in tqdm(train_df.iterrows()):
    ebird_code = item.ebird_code
    filename = item.filename
    try:
        secondaries = np.load(f"../oof-preds/{ebird_code}/{filename}.npy", allow_pickle=True).item()["secondary"]
        #all_dict[filename] = {}
        preds = []

        for label in eval(item.secondary_labels):
            #print(label, secondaries)
            if label in secondaries:
                preds.append(secondaries[label])

        if len(preds) > 0:
            all_dict[filename] = np.stack(preds, axis=0)
    except:
        print(filename)

import pickle

pickle.dump(all_dict, open("../secondary.pickle", "wb"), )





