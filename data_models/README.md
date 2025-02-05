Contains scripts to generate the 3 used datasets (randomly sampled, selected to reduce ngram bias, selected to produce unclassifiable data) and to train classifiers, the resulting models and the overall analysis (ROC, ngram distributions...)

## sample_data.sh
    Refer to the randomly sampled dataset. Populates data_m1

    Randomly samples 1000 x2 books 
    Splits books between members and non-members, training data and the rest of the available data, into 4 folders containing (links to) the .txt files  
    Builds .arff datasets
    

## sample_datam2.sh
    Refer to the dataset minimizing ngram overlap bias. Populates data_m2
 
    Randomly samples 500 x2 books, where non-members are taken from the "picked" subset to minimze ngram bias
    Splits books between members and non-members, training data and the rest of the available data, into 4 folders containing (links to) the .txt files  
    Builds .arff datasets 
 

## data_m1/ and data_m2

Empty folders populated by sample_data and sample_datam2.

## data_m3

Contains scripts to build the third, "hard to classify" dataset

## models

    Script to train models and these datasets, output models and their evaluation

## ngram_analysis

    Scripts to compute and plot ngram overlap distributions for the first and second dataset
