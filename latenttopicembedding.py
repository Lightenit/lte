#! /usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import division
from collections import defaultdict
import collections
import numpy as np
import random
import nltk
import gensim


def preprocess(doc):
    doc_sen = sent_tokenize(doc)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    predoc = []
    for sen in doc_sen:
        sen_word = [word for word in nltk.word_tokenize(sen) if word not in english_punctuations]
        predoc.append(sen_word)
    return doc

def build_dict(docs):
    dictionary = dict()
    for doc in docs:
        for sen in doc:
            for word in sen:
                dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.vaules(), dictionary.keys()))
    

