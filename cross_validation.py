from dataset import compute_embeddings
import torch as tr
import random
import numpy as np
import os
import pandas as pd
import ipdb
import pickle
import json
from flair.embeddings import FastTextEmbeddings, WordEmbeddings
from base_run import run, get_embeddings
import sys

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
KEYWORD_EMB_SIZE = 3


conf = json.load(open(sys.argv[1]))
conf.update(json.load(open(sys.argv[2])))


PUBLICATIONS_PATH = os.path.join(conf["base_dir"], "publications/")
LABELS_PATH = os.path.join(conf["base_dir"], "labels.csv")

# Main script
if not os.path.isdir(conf["work_dir"]):
    os.mkdir(conf["work_dir"])

labels = pd.read_csv(LABELS_PATH)

# Filter the labels that counts at least 1% of the dataset
interactions = ["no_interaction", "inhibitor", "agonist", "antagonist", "cofactor",
                "binder", "inducer", "antibody"]
ind = [k for k, i in enumerate(labels.loc[:, "interaction"]) if i in
       interactions]
labels = labels.iloc[ind, :].reset_index(drop=True)


print("Preprocessing embeddings...")
conf["emb_size"] += 3
embeddings, emb_path = get_embeddings(conf, labels, PUBLICATIONS_PATH)
print("Done.")

run(labels, conf, f"{conf['model']}",
    PUBLICATIONS_PATH, embeddings, emb_path)

