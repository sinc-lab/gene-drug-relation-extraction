# Gene-drug relation-type extraction in full biomedical texts

This is the source code used in:

L.A. Bugnon, C. Yones, J. Bertinetti, D. Ramírez,  D.H. Milone, G. Stegmayer,
Gene-drug relation-type extraction in full-text biomedical publications, 2023 (under review)

We propose a framework for extracting gene-drug relationship type from full biomedical texts. Differently from many approaches that are designed for in-sentence classification, our approach is based on the premise that entities interactions may appear far away in the text. Using only the raw text and the identification of biomedical entities of interest as inputs, we propose a combination of word-embeddings and a convolutional neural network to cope with text length. 


![Abstract](abstract.svg)

This repository contains the scripts and dataset to reproduce the paper results.
A Python>=3.9 is recomended. Install the required packages with:

    pip install -r requirements.txt

The dataset with texts and labels is in DGIdb_sinc/

## Word embedding preparation

Word2Vec, FastText and GloVe have low computational cost, thus the embeddings are computed in the training script.

In the case of Flair, embeddings need to be precomputed with

   python embed_flair.py conf_flair.json path.json

A similar procedure is required for BioBERT:

   python embed_biobert.py conf_biobert.json path.json

## Hiperparameter optimization
The complete hiperparameter evaluation of the network for each embedding model is done with 
   
   python hp_exploration.py conf_<embedding_name>.json paths.json

This could take several hours.

## Run cross-validation 

To run a complete cross-validation scheme, use the configuration files as the following

   python cross_validation.py conf_{embedding_name}.json paths.json

A summary of the cross validation results  can be viewed using the notebook "summary.ipynb"