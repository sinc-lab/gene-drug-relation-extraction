import torch as tr
import random
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json
from base_run import run, get_embeddings

KEYWORD_EMB_SIZE = 3

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys

conf = json.load(open(sys.argv[1]))
conf.update(json.load(open(sys.argv[2])))

PUBLICATIONS_PATH = os.path.join(conf["base_dir"], "publications/")
LABELS_PATH = os.path.join(conf["base_dir"], "labels.csv")

if not os.path.isdir(conf["work_dir"]):
    os.mkdir(conf["work_dir"])

labels = pd.read_csv(LABELS_PATH)

# Filter the labels that counts at least 1% of the dataset
interactions = ["no_interaction", "inhibitor", "agonist", "antagonist", "cofactor",
                "binder", "inducer", "antibody"]
ind = [k for k, i in enumerate(labels.loc[:, "interaction"]) if i in
       interactions]
labels = labels.iloc[ind, :].reset_index(drop=True)


# Use the first train partition to explore hyper-parameters
# Reproducibility
random.seed(1)
np.random.seed(1)
tr.manual_seed(1)
if tr.cuda.is_available():
    tr.cuda.manual_seed_all(1)
xval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_ind, _ in xval.split(np.arange(len(labels)), labels["interaction"]):
    break

# Using only train partition to explore hyperparameters
labels = labels.iloc[train_ind].reset_index(drop=True)


print("get embeddings...")
embeddings, emb_path = get_embeddings(conf, labels, PUBLICATIONS_PATH)
print("Done")
conf["emb_size"] += 3

for nfilters in [32]:
    for nblocks in [3, 4]:
        conf["nfilters"] = nfilters
        conf["nblocks"] = nblocks

        run(labels, conf, f"{conf['model']}_nfilters{nfilters}_nblocks{nblocks}", PUBLICATIONS_PATH, embeddings, emb_path)
