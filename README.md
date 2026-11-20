# Sentiment Analysis on IMDB Movie Reviews

## Overview
Classifies movie reviews as positive or negative using TF-IDF and logistic regression.

## Dataset
IMDB movie review dataset(Kaggle),balanced 25k positive reviews and 25k negative reviews
split:80% train / 20% test(stratified)

## Pipeline
- Text preprocessing(lowercase,remove HTML,tags and punctuation)
- TF-IDF vectorisation(fit on  train data only to avoid data leakage)
- Logistic regression classifier
- Evaluation:accuracy,precision,recall,confusion matrix

## Results
|Version|Accuracy|
|---|---|
|Baseline(unigrams, English stop words removed) |88.85%|
|Improved(unigrams+Biagrams, negations kept) |90.5%|

## What I learned
The baseline classified "it was not good at all" as positive because
sklearn's English stop-word list removes "not". Keeping negations and
adding bigrams fixed it, since "not good" became its own feature.

## Tech
Python,Pandas,Scikit-learn

## Author
Archana Krishna
BCA Student
