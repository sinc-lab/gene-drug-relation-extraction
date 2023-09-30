from dataset import InteractionsDataset, compute_embeddings
import numpy as np
import torch as tr
import random
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import pytorch_lightning as pl
from classifier_model import InteractionCNN
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from flair.embeddings import FastTextEmbeddings, FlairEmbeddings, \
    StackedEmbeddings, WordEmbeddings
import os


def run(labels, conf, name, publications_path, embeddings, emb_path=None):

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    tr.manual_seed(1)
    if tr.cuda.is_available():
        tr.cuda.manual_seed_all(1)

    xval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_ind, test_ind) in enumerate(xval.split(np.arange(len(
            labels)), labels["interaction"])):

        random.seed(1)
        np.random.seed(1)
        tr.manual_seed(1)
        if tr.cuda.is_available():
            tr.cuda.manual_seed_all(1)
        train_ind, optim_ind = train_test_split(train_ind, test_size=.2,
                                                stratify=labels.loc[train_ind,
                                                "interaction"])
        print(f"Partitions size: train {len(train_ind)}, optim {len(optim_ind)}, test"
              f" {len(test_ind)}")

        
        train_dataset = InteractionsDataset(labels.iloc[train_ind],
                                            embeddings=embeddings,
                                            max_len=conf["max_len"],
                                            emb_path=emb_path,
                                            publications_dir=publications_path)
        optim_dataset = InteractionsDataset(labels.iloc[optim_ind],
                                            embeddings=embeddings,
                                            max_len=conf["max_len"],
                                            emb_path=emb_path,
                                            publications_dir=publications_path)
        test_dataset = InteractionsDataset(labels.iloc[test_ind],
                                           embeddings=embeddings,
                                           max_len=conf["max_len"],
                                           emb_path=emb_path,
                                           publications_dir=publications_path)

        train_loader = DataLoader(train_dataset, batch_size=conf["batch_size"],
                                  shuffle=True)
        optim_loader = DataLoader(optim_dataset, batch_size=conf["batch_size"])
        test_loader = DataLoader(test_dataset, batch_size=conf["batch_size"])

        checkpoint_callback = ModelCheckpoint(monitor="valid_loss", mode="min")
        early_stop_callback = EarlyStopping(monitor="valid_loss", patience=30, mode="min")

        logger = TensorBoardLogger(conf["work_dir"], name=name)

        trainer = pl.Trainer(accelerator='cuda', devices=[1], default_root_dir=conf["work_dir"],
                             callbacks=[checkpoint_callback, early_stop_callback],
                             max_epochs=350, 
                             logger=logger)

        if conf["weight_labels"]:
            class_weight = train_dataset.get_class_weight().to(conf["device"])
        else:
            class_weight = None
        model = InteractionCNN(emb_size=conf["emb_size"], nblocks=conf["nblocks"],
                    nfilters=conf["nfilters"], nclasses=len(
                train_dataset.interactions),
                    class_weight=class_weight)
        trainer.fit(model, train_loader, optim_loader)

        trainer.test(model, test_loader, ckpt_path='best')

        # TODO lo  siguiente podria no ir en la versión final
        model.load_from_checkpoint(checkpoint_callback.best_model_path, emb_size=conf["emb_size"], nblocks=conf["nblocks"],
                    nfilters=conf["nfilters"], nclasses=len(
                train_dataset.interactions), class_weight=class_weight)
        model.eval()
        model.to(conf["device"])

        out, ref, case = [], [], []
        for x, y, c in test_loader:
             out.append(model(x.to(conf["device"])).detach().cpu())
             ref.append(y.cpu())
             case.append(c)
        pickle.dump([out, ref, case, train_dataset.interactions],
                    open(f"{conf['work_dir']}{name}_fold{fold}.pk", "wb"))


def get_embeddings(conf, labels, publications_path):
    emb_path = None
    if conf["model"] == "BioBERT":
        print("Loading biobert...")
        embeddings = {}
        emb_size = conf["emb_size"]
        for f in os.listdir(conf["biobert_out_path"]):
            embeddings[int(os.path.splitext(f)[0])] = pickle.load(open(conf["biobert_out_path"] + f, "rb"))
    elif conf["model"] == "Flair":
        embeddings = None # online
        emb_path = conf["flair_path"]
    else:
        if conf["model"] == "FastText":
            embedding_model = FastTextEmbeddings(conf["fasttext_path"])
            
        if conf["model"] == "GloVe":
            print('get glove embeddings...')
            embedding_model = WordEmbeddings("glove")
        if conf["model"] == "Word2vec":
            embedding_model = WordEmbeddings("en-crawl")
        print('compute embeddings')
        embeddings = compute_embeddings(labels, conf["model"], embedding_model,
                                        conf["max_len"],
                                        publications_path)
        del embedding_model

    return embeddings, emb_path
