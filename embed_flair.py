"""Precompute Flair embedding for the DGIsinc dataset"""

from flair.embeddings import FlairEmbeddings, StackedEmbeddings
import json
import os
import pandas as pd
from dataset import embed
import pickle
from tqdm import tqdm
import sys
from sklearn.decomposition import IncrementalPCA
import torch as tr
embedding_model = StackedEmbeddings([
    FlairEmbeddings('pubmed-forward'),
    FlairEmbeddings('pubmed-backward'),
])


conf = json.load(open(sys.argv[1]))


if not os.path.isdir(conf["flair_path"]):
    os.mkdir(conf["flair_path"])

PUBLICATIONS_DIR = os.path.join(conf["base_dir"], "publications/")
LABELS_PATH = os.path.join(conf["base_dir"], "labels.csv")
MAX_LEN = 10000

labels = pd.read_csv(LABELS_PATH)

interactions = labels["interaction"].unique().tolist()

embeddings = None

# Transform and save embedding
embeddings = {}
for n, pmid in enumerate(tqdm(labels["PMID"].unique())):

    with open(f"{PUBLICATIONS_DIR}{pmid}.txt", encoding="utf8") as fin:
        text = fin.read()

    embeddings = embed(text, MAX_LEN, embedding_model)

    pickle.dump(embeddings, open(f"{conf['flair_path']}{pmid}.pk",
                                     "wb"))
    
