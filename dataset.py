from torch.utils.data import Dataset
import numpy as np
import os
import csv
import pickle
import re
from flair.data import Sentence
import torch as tr
import time
from tqdm import tqdm
import string
import flair

def compute_embeddings(labels, emb_name, embedding_model, max_len, publications_dir):
    """ Precompute embeddings for each PMID in the dataset"""
    
    cache = f"/media/DATOS/lbugnon/emb_{emb_name}.pk"
    if os.path.isfile(cache):
        return pickle.load(open(cache, "rb"))
    embeddings = {}
    for k, pmid in enumerate(tqdm(labels["PMID"].unique())):

        with open(f"{publications_dir}{pmid}.txt", encoding="utf8") as fin:
            text = fin.read()

        tokens, keywords, tsize = embed(text, max_len, embedding_model)

        embeddings[pmid] = tokens, keywords
    
    pickle.dump(embeddings, open(cache, "wb"))
    return embeddings

def embed(text, max_len, embedding_model):

    tokens = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
    tokens = Sentence([t for t in tokens if len(t)>2])

    tsize = len(tokens)
    tokens.tokens = tokens.tokens[:max_len]
        
    keywords = get_keywords(tokens)
    
    # Apply word embedding
    flair.device = 'cpu'
    if embedding_model:
        embedding_model.embed(tokens)
        word_emb = tr.cat([token.embedding.unsqueeze(0).detach().cpu() for token in
                       tokens])
    else:
        word_emb = tr.zeros(len(tokens), 1)

    return word_emb, keywords, tsize


def get_keywords(tokens):
    """Get keywords positions"""

    keywords = {}
    for k, token in enumerate(tokens):
        if "xxx" in token.text: # keyword, either gene or drug
            if token.text not in keywords:
                keywords[token.text] = []
            keywords[token.text].append(k)
            
    return keywords

class InteractionsDataset(Dataset):

    def __init__(self, labels, embeddings, max_len, work_dir="tmp", publications_dir=None, emb_path=None):
        """

        :param publications_dir:  Directory with the whole texts.
        :param labels:  DataFrame with examples to load. Columns are: PMID,
        keyterm1, keyterm2, interaction.
        :param tokenizer: Tokenizer used for specific embeding model (
        "fastext" or "biobert")
        :param work_dir: Temporary directory.
        """

        self.embeddings = embeddings
        if not embeddings:
            self.files = {}
            for f in os.listdir(emb_path):
                if "pk" in f:
                    self.files[int(f.split(".")[0])] = emb_path + f
            
        self.max_len = max_len
        
        self.publications_dir = publications_dir

        self.labels = labels
        self.interactions = sorted(np.unique(labels["interaction"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        pmid, target_gene, target_drug, label = self.labels.iloc[item, :]

        if self.embeddings:
            word_emb, keywords = self.embeddings[pmid]

            if type(word_emb) != np.ndarray:
                word_emb = word_emb.clone().detach()
            else:
                word_emb = tr.tensor(word_emb)
        else:
            word_emb, keywords, _ = pickle.load(open(self.files[pmid], "rb"))
            
        # is_gene, is_drug, is_target
        keyword_emb = tr.zeros((word_emb.shape[0], 3))

        for k in keywords:
            if "g" in k:
                keyword_emb[keywords[k], 0] = 1
            if "d" in k:
                keyword_emb[keywords[k], 1] = 1
            if target_drug == k or target_gene == k:
                keyword_emb[keywords[k], 2] = 1

        emb_size = word_emb.shape[1] + 3
        embedding = tr.zeros((self.max_len, emb_size))

        # Concatenate word embeddings and one-hot keyword embeddings
        embedding[:word_emb.shape[0], :] = tr.cat((word_emb, keyword_emb),
                                                  axis=1)

        return embedding.T, self.interactions.index(label), f"{pmid}_{target_gene}_{target_drug}"

    def get_class_weight(self):

        w = tr.zeros(len(self.interactions))
        for k, label in enumerate(self.interactions):
            w[k] = np.sum(self.labels["interaction"] == label)

        weight = ((1/w)/(tr.sum(1/w)))
        return weight

    def get_samples_weights(self):
        """Get samples probabilities given its labels"""

        w = self.get_class_weight()
        weights = [w[self.interactions.index(l)] for l in self.labels[
            "interaction"]]
        return weights
