import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import names

import sys
import os
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
# import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans

# import logging
# from optparse import OptionParser

import pickle

from sklearn.cluster import KMeans
import nltk

import gensim

from gensim.models import CoherenceModel
from nltk.corpus import stopwords
# import sqlite3 as lite

from sklearn.model_selection import train_test_split


def getData():
    p = '/Users/youssefkindo/Downloads/Tweets_Dataset.txt'

    file = open(p, "r")
    headers = []
    tony_id = []
    sentiment = []
    text = []
    i = 0

    for sent in file:
        if i < 1:
            var = sent.split('\t')
            headers.append(var[0])
            headers.append(var[1])
            headers.append(var[2])
            i = i + 1
        else:
            var = sent.split('\t')
            if len(var) >= 3:
                tony_id.append(var[0])
                sentiment.append(var[1])
                text.append(var[2])
    return text, sentiment


def word_feats(words):
    return dict([(word, True) for word in words])


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def getLabledFeatures(text, sentiment):
    positive_sent = []
    negative_sent = []

    # split from: https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/model_selection/_split.py#L2073

    train, test = train_test_split(text, test_size=0.1, random_state=1234)

    for i in range(len(train)):
        if sentiment[i] == '+':
            positive_sent.append(text[i])
        else:
            negative_sent.append(text[i])

    # positive_sent = remove_stopwords(positive_sent)
    # negative_sent = remove_stopwords(negative_sent)

    positive_vocab = list(sent_to_words(positive_sent))
    negative_vocab = list(sent_to_words(negative_sent))

    positive_features = [(word_feats(pos), '+') for pos in positive_vocab]
    negative_features = [(word_feats(neg), '-') for neg in negative_vocab]
    train_set = negative_features + positive_features
    return train_set


if __name__ == '__main__':
    mdl_file = 'trained-mdl.pkl'
    docs, labels = getData()
    X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=.1, random_state=37654)
    classifier = None
    if os.path.exists(mdl_file):
        print('Reading from the trained model file.')
        with open(mdl_file, 'rb') as f:
            data = pickle.load(f)
            classifier = data['classifier']

    else:
        train_set = getLabledFeatures(X_train, y_train)

        classifier = NaiveBayesClassifier.train(train_set)

    with open(mdl_file, 'wb') as f:
        data = {
            'classifier': classifier,
        }
        pickle.dump(data, f)

    test_data_positive_and_neg = list(sent_to_words(X_test))
    pred = []
    for sent in test_data_positive_and_neg :
        pred.append(classifier.classify(word_feats(sent)))

    classifier.show_most_informative_features()
    confusion_matrix = {"TP": 0, "FN": 0, "TN": 0, "FP": 0}
    for index, value in enumerate(pred):
        if value == y_test[index] and value == "+":
            confusion_matrix["TP"] += 1
        if value == y_test[index] and value == "-":
            confusion_matrix["TN"] += 1
        if value != y_test[index] and value == "+":
            confusion_matrix["FP"] += 1
        if value != y_test[index] and value == "-":
            confusion_matrix["FN"] += 1
    print("--------confusion matrix -----------------")
    tm = "\t +\t\t -\n +\t TP: " + str(confusion_matrix["TP"]) + "\t FN: " + str(
        confusion_matrix["FN"]) + "\n -\t FP: " + str(confusion_matrix["FP"]) + "\t TN: " + str(
        confusion_matrix["TN"])

    print(tm)
    accuracy = (confusion_matrix["TP"] + confusion_matrix["TN"])/len(pred)

    print("\n Accuracy: ", accuracy)

