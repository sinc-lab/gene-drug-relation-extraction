"""Precompute Biobert based word-embedding from papers"""

import json
import os
import pandas as pd
from dataset import get_keywords
import pickle
from tqdm import tqdm
import torch as tr
from transformers import BertTokenizer, AutoConfig, AutoModel
from flair.data import Sentence
import sys
import string

conf = json.load(open(sys.argv[1]))
conf.update(json.load(open(sys.argv[2])))
    
PUBLICATIONS_DIR = os.path.join(conf["base_dir"], "publications/")
LABELS_PATH = os.path.join(conf["base_dir"], "labels.csv")
if not os.path.isdir(conf["biobert_out_path"]):
    os.mkdir(conf["biobert_out_path"])
labels = pd.read_csv(LABELS_PATH)

for f in os.listdir(conf['biobert_path']):
    if ".txt" in f:
        vocab = f
    if ".index" in f:
        index = f
    if ".json" in f:
        bconf = f

tokenizer = BertTokenizer(f"{conf['biobert_path']}{vocab}",
                          do_lower_case=False)
conf_file = f"{conf['biobert_path']}{bconf}"
bert_config = AutoConfig.from_pretrained(conf_file)
emb_size = bert_config.hidden_size
emb_max_len = bert_config.max_position_embeddings

fname = f"{conf['biobert_path']}{index}"
biobert = AutoModel.from_pretrained(fname, from_tf=True,
                                                 config=bert_config)
biobert.to(conf["device"])
biobert.eval()

for param in biobert.parameters():                
    param.requires_grad = False

MAX_LEN = 10000
device = conf["device"]


labels = pd.read_csv(LABELS_PATH)

interactions = labels["interaction"].unique().tolist()


embeddings = {}
for npmid, pmid in enumerate(tqdm(labels["PMID"].unique())):

    if f"{pmid}.pk" in os.listdir(f"{conf['biobert_out_path']}"):
        print(pmid, 'ok')
        continue
    with open(f"{PUBLICATIONS_DIR}{pmid}.txt", encoding="utf8") as fin:
        text = fin.read()

    embedding = tr.zeros((MAX_LEN, emb_size)).to("cpu")
    
    # Tokenize the same way as other embeddings to track entities and use the same classifier model. Internally, the BERT tokenizer is used.
    tokens = text.translate(str.maketrans("", "", string.punctuation)).lower().split()
    tokens = Sentence([t for t in tokens if len(t)>2])
    tsize = len(tokens)
    tokens.tokens = tokens.tokens[:MAX_LEN]
        
    keywords = get_keywords(tokens)

    # Takes the context of each word, compute biobert embedding, join tokens (word tokenization) and save the word embedding.
    WORD_TOKEN_WIN = 200
    win_list, win_start = [], []
    for k, w in enumerate(tokens):
        start = max(k - WORD_TOKEN_WIN // 2, 0)
        end = k + WORD_TOKEN_WIN // 2
        
        win_list.append(' '.join([w.text for w in tokens[start: end]]))
        win_start.append(start)

    tokens = tokenizer(win_list,
                       add_special_tokens=False,
                       return_tensors="pt", truncation=True,
                       max_length=emb_max_len, padding=True)

    batch = 0
    batch_size = 64
    while batch*batch_size < len(win_list):
        start = batch*batch_size
        end = min((batch+1)*batch_size, len(win_list))
        ind = slice(start, end)
        with tr.no_grad():
            out = biobert(
                input_ids=tokens["input_ids"][ind].to(device),
                attention_mask=tokens["attention_mask"][ind].to(device),
                token_type_ids=tokens["token_type_ids"][ind].to(device))
        tr.cuda.empty_cache()

        embedding[ind, :] = out["pooler_output"].detach().to("cpu")
        
        batch += 1


    pickle.dump([embedding, keywords], open(
            f"{conf['biobert_out_path']}{pmid}.pk", "wb"))



    
