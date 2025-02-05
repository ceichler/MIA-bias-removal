# MIA bias removal

This repository contains the code used in [this paper](https://arxiv.org/abs/2408.05968). 
The overall aim is to construct datasets of members and non-members from Gutenberg minimizing bias. 

Code contained here can be used to produce datasets, analyse ngram distribution overlap, train classifier and evaluate MIAs. If you find it usefull, please cite as follows:

>Cédric Eichler, Nathan Champeil, Nicolas Anciaux, Alexandra Bensamoun, Héber H. Arcolezi, et al.. Nob-MIAs: Non-biased Membership Inference Attacks Assessment on Large Language Models with Ex-Post Dataset Construction. WISE 2024 - 25th International Web Information Systems Engineering conference, Dec. 2024, Doha, Qatar. pp.441-456

## How to get datasets?

### Pipeline to construct the datasets

#### Get the initial dataset
Every pipeline needs the initial dataset. Go to get_data to execute the pipeline fetching and filtering the Gutenberg dataset to build the initial dataset

#### Analyse ngram distribution ofr the dataset minimizing bias
1. Compile bff (see ngram_analysis/bff)
2. Follow the pipeline in ngram_analysis/altered_distribution

#### Organize the randomly sampled and ngram datasets neatly
Use the two sampling scripts in data_models

#### Produce the third, hard to classify dataset
Use the scripts in data_model/data_m3

### Don't want to execute the pipeline?
Datasets can be found [here](https://gitlab.inria.fr/ceichler/nob-mia/)


## Contents

### data_models
    Contains scripts to construct the three dataset used in the paper (random, minimizing ngram bias, minimizing classifiability) from the initial dataset, the related models, the scripts to train them and the result of their evaluation.

### get_data
    Related to the initial dataset, extracted from Gutenberg following the method of [Meeus et al.](https://arxiv.org/abs/2310.15007)
    Contains scripts and a pipeline to replicate the data collection

### MIA_evaluation
    Contains the code necessary to evaluate MIAs on Pythia and OpenLlama with our datasets

### ngram_analysis
    Contains everything needed to compute n-gram overlap distribution and sample a set of non-members minimizing the related bias

