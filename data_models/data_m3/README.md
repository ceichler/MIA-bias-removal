Folder related to the third, "hard to classify" dataset. To produce the dataset, execute TrainKfoldsNclassify.py followed by sampleDS3.py

## TrainKfoldsNclassify.py

Train a classifier on the first "train" dataset (randomly sampled), then use it to classify the remainder "rest" of the dataset.
Produce rest_proba_classification.csv

## rest_proba_classification.csv

Contains the "rest" of the first dataset, labeled as members and non members, with the score (between 0 and 1) assigned by the classifier.
Used by sampleDS3.py

## sampleDS3.py

Uses rest_proba_classification.csv to produce the third dataset by selecting the non-members and members minimizing the confidence of the classifier (the closest to 0.5). Sampling is done so as to select the same number of true positive, false positive, true negative and false negative.
