# Gene-drug relation-type extraction in full biomedical texts

This package contains the original methods proposed in:

L.A. Bugnon, C. Yones, J. Bertinetti, D. Ramírez,  D.H. Milone, G. Stegmayer,
Gene-drug relation-type extraction in full-text biomedical publications
(2022 - in review)

We propose a framework for extracting gene-drug relationship type from full biomedical texts.
Differently from many approaches that are designed for in-sentence classification, our approach is based on the premise that entities interactions may appear far from the entity sentence. Thus, our approach only requires the raw text and the identification of biomedical entities of interest (BioNER stage).

Each entity is identified (in this case, it is either “gene”, “drug” or “none”) and coded in a one-hot vector. The specific entities that we are looking for are also indicated in this coding. In order to extract language information from text, a pre-trained word-embedding can be used. These embedding models are important as they are able to code the semantics of the words into vector representations.  Finally, a CNN classifier can integrate the information of all the words and entities of the whole text and output a relation-type prediction for the target entities. 

![Abstract](abstract.svg)

This repository contains the scripts and dataset to reproduce the paper results.

## Installation

A Python>=3.9 distribution is required to use this package. It is recomended to create and activate an environment, such as 
    virtualenv -p python3.9 venv
    source venv/bin/activate
    
If you are using a GPU, you should probably install PyTorch first following the instructions in https://pytorch.org/.
Then install the required packages with:

    pip install -r requirements.txt


### TODO poner instrucciones para las cosas que requieren paquetes o binarios 
For 

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


# TODO revisar que quede andando con un solo método, pero sería bueno dejarla corrida con todos los métodos (en lo psoible que sean generados por esta misma carpeta)
Results can be viewed using the notebook "summary.ipynb"