#! /usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import division
from collections import defaultdict
import collections
import numpy as np
import random
import nltk
import gensim
import math

def preprocess(doc):
    doc_sen = sent_tokenize(doc)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    predoc = []
    for sen in doc_sen:
        sen_word = [word for word in nltk.word_tokenize(sen) if word not in english_punctuations]
        predoc.append(sen_word)
    return doc

def build_dict(docs_word):
    word_set = set(docs_word)
    dictionary = dict()
    for word in word_set:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.vaules(), dictionary.keys()))
    return dictionary, reversed_dictionary

def doc_tran(docs,dictionary,reversed_dictionary):
    POS_dictionary = dict()
    for POS in ['ADJ','ADV','CNJ','DET','EX','FW','MOD','N','NP','NUM','PRO','P','TO','UH','V','VD','VG','VN','WH']:
        POS_dictionary[POS] = len(POS_dictionary)
    POS_NUM = len(POS_dictionary)
    Tran_Docs = []
    for doc in docs:
        tran_doc = []
        for sen in doc:
            tran_sen = []
            tag_sen = nltk.pos_tag(sen)
            for word,pos in tag_sen:
                tran_sen.append((dictionary[word],POS_dictionary[pos]))
            tran_doc.append(tran_sen)
        Tran_Docs.append(tran_doc)
    return Tran_Docs, POS_NUM



def initial(Docu_NUM,Topic_NUM,Embedding_SIZE,POS_NUM,dictionary,docs):
# Topic INIT for all sent in all doc
    Topic_List = []
    m = np.zeros((Docu_NUM,Topic_NUM))
    n = np.zeros((len(dictionary),Topic_NUM))
    doc_count = 0
    tao = np.random.random(POS_NUM)
    tao[7] = 0.9
    i_word = np.zeros(len(dictionary))
    for doc in docs:
        topic_doc = []
        for sen in doc:
            ran = random.random()
            temp_k = math.floor(temp_k*Topic_NUM)
            topic_doc.append(temp_k)
            m[doc_count,temp_k] = m[doc_count,temp_k] + 1
            for word in sen:
                n[word[0],temp_k] = n[word[0],temp_k] + 1
                i_word[word[0]] = np.random.binomial(1,tao[word[1]])
        Topic_List.append(topic_doc)
# Initial topic vector
    Topic_Vec = np.random.random((Topic_NUM, Embedding_SIZE))
# Initial word vector
    Word_Vec = np.zeros(len(dictionary))
    model = gensim.models.Word2Vec.load_word2vec_format('.bin',binary = Ture)
    for word,index in dictionary:
        try:
            Word_Vec[index] = model.wv[word]
        except:
            Word_Vec[index] = np.random.random(Embedding_SIZE)
    return m,n,tao,Topic_List, Topic_Vec, Word_Vec,i_word

def Update(m,n,tao, Topic_List,Topic_Vec,Word_Vec,i_word,docs):
    for d in range(len(docs)):
        for s in range(len(docs[d])):
            s_topic = Topic_List[d][s]
            m[s_topic,d] = m[s_topic,d] - 1
            for word,_ in docs[d][s]:
                n[word,s_topic] = n[word,s_topic] - 1
            P_z = np.zeros(Topic_NUM)
            for  k in Topic_NUM:
                temp_prod = 1
                for word,pos in docs[d][s]:
                    temp_prod = temp_prod * ((1-tao[pos]) * (n[word,s_topic] + beta)/(sum(n[:,s_topic])+len(dictionary)*beta) + tao[pos] * np.exp((Word_Vec[word]+Topic_Vec[s_topic]).dot(Topic_Vec[s_topic])))
                P_z[k] = temp_prod * (m[s_topic,d] + alpha)
            P_z = P_z/np.linalg.norm(P_z)



    


                





    

