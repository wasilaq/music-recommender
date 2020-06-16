#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:58:19 2020

@author: wasilaq
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from sklearn.decomposition import NMF
from gensim import models, matutils


lyrics_bow = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/lyrics_bow')

# remove stop words from bag of words
stop_words = stopwords.words('english')
cleaned_bow = lyrics_bow
for word in lyrics_bow.columns:        
    if word in stop_words:
        cleaned_bow = cleaned_bow.drop(word, axis=1)


# NMF
def NMF_vars(num_topics, dtm):
    model = NMF(num_topics)
    fitted = model.fit_transform(dtm)
    return model, fitted

def topic_words(model, dtm, num_words=10):
    topic_words = model.components_.round(5)
    for ix, topic in enumerate(topic_words):
        print('Topic {}'.format(ix+1))
        word_list = []
        for i in topic.argsort()[::-1][:num_words]:
            word = dtm.columns[i]
            word_list.append(word)
        print(word_list)
        
def topic_likelihood(fitted_model, topic_num):
    topic = []
    for doc in fitted_model:
        topic.append(doc[topic_num])
    return topic


topic_words(NMF_vars(10,cleaned_bow)[0], cleaned_bow)

# remove words that weren't caught by language detection
non_eng_words = ['ca','de','el','te','en','mi','tu','es','se','que','da','babi','la','e','che','di','non','un','il','le','je','les','et','pas','des','und']
for word in non_eng_words:
    try:        
        cleaned_bow = cleaned_bow.drop(word, axis=1)
    except KeyError:
        pass

# nouns only
nouns_bow = cleaned_bow
for word in cleaned_bow.columns:
    pos = pos_tag(list(word))[0][1]
    if pos != 'NN':
        nouns_bow = nouns_bow.drop(word, axis=1)
        

topic_words(NMF_vars(10,nouns_bow)[0], nouns_bow)

topic_words(NMF_vars(5,nouns_bow.drop('girl', axis=1))[0], nouns_bow.drop('girl', axis=1))

# try different tagger
nouns_bow_2 = cleaned_bow
tagger = UnigramTagger(brown.tagged_sents())
for word in cleaned_bow.columns:
    pos = tagger.tag(list(word))[0][1]
    if pos != 'NN':
        nouns_bow_2 = nouns_bow_2.drop(word, axis=1)
 
for num in range(2,6):
    topic_words(NMF_vars(num,nouns_bow_2)[0], nouns_bow_2)

topic_words(NMF_vars(10,nouns_bow_2)[0], nouns_bow_2)
# remove the word 'total'
topic_words(NMF_vars(10,nouns_bow_2.drop('total',axis=1))[0], nouns_bow_2.drop('total',axis=1))

for num in range(12,16):
    topic_words(NMF_vars(num,nouns_bow_2)[0], nouns_bow_2)
# best # of topics: 13

# LDA
# mapping (dict) of row id to word (token)
def generate_id2word(dtm):
    id2word = {}
    for row in dtm.index:
        for word in dtm.columns:
            if dtm[word].loc[row] > 0:
                id2word[row] = word
    return id2word

def LDA_topics(num_topics, dtm):     
    gensim_corpus = matutils.Sparse2Corpus(dtm.transpose()) # sparse2corpus doesn't work for dataframes
    id2word = generate_id2word(dtm)
    LDA_model = models.LdaModel(corpus=gensim_corpus, num_topics=num_topics, id2word=id2word, passes=5)
    
    return LDA_model.print_topics()

LDA_topics(10, cleaned_bow)
# better results with NMF

# final model
model, fitted_model = NMF_vars(13,nouns_bow_2)
topic_words(model, nouns_bow_2)
topic_likelihood(fitted_model, 1)

all_tracks = cleaned_bow
for num in range(13):
    name = 'topic' + str(num+1)
    all_tracks[name] = topic_likelihood(fitted_model, num)


# pickle dataframes
nouns_bow_2.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/nouns_bow')
all_tracks.to_pickle('/Users/wasilaq/Metis/music-recommender/pickled/lyrics_and_topics')