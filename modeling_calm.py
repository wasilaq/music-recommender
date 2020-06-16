#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:00:32 2020

@author: wasilaq
"""

from classification_modeling import *
import pickle
from numpy import arange
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.dummy import DummyClassifier

targets = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/targets')
emotion_subset = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/emotions_df_subset')

identifiers = ['track_id','song_id','artist','song_title']

y = targets['calm']
y.value_counts() # balanced data

X = emotion_subset.drop(identifiers+['emotion'], axis=1)
X_te, X_test, y_te, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=12)

# keep dummy classifier in code??
cross_val_score(DummyClassifier(random_state=2), X_train, y_train, cv=5, scoring='f1')
# [0.46808511, 0.45      , 0.47      , 0.4825    , 0.45363409]

# logistic regression
for C in arange(.1,1,0.1):
    print(C)
    logistic_reg(X_train, y_train, C=C)
# best logistic regression: C=1
# [0.51629726 0.53856563 0.51902174 0.51535381 0.55613577]


# random forest
for n in range(5,15):
    print('{} is n_estimators'.format(n))
    for depth in [12]:
        print('{} is max depth'.format(depth))
        random_forest(X_train, y_train, n_estimators=n, max_depth=depth)
# best random forest: n_estimators=12, max_depth=12
# [0.60992908 0.61940299 0.60118343 0.59348613 0.63768116]
# [[1856  360]
# [ 279 1789]]


# naive bayes
naive_bayes(X_train, y_train)
# [0.42732558 0.50144928 0.46539028 0.4549483  0.53422819]
# [[1660  556]
# [1256  812]]


# knn
for neighbors in range(2,10):
    print(neighbors)
    knn(X_train, y_train, n_neighbors=neighbors)
# best knn: sampling, 3 neighbors
# [0.56324582 0.54207921 0.5754386  0.57631258 0.58679707]
# [[1719  497]
# [ 457 1611]]


# compare models
best_models = [
    (logistic_reg(X_train, y_train), 'Logistic Regression'),
    (random_forest(X_train, y_train, n_estimators=12, max_depth=12), 'Random Forest'),
    (naive_bayes(X_train, y_train), 'Naive Bayes'),
    (knn(X_train, y_train, n_neighbors=3), 'KNN')
    ]

for model, name in best_models:
    y_prob = (model.fit(X_train, y_train)).predict_proba(X_eval)[:,1]
    
    fpr, tpr, threshold = roc_curve(y_eval, y_prob)
    plt.plot(fpr, tpr)
    print('AUC score for {} is {:.2f}'.format(name, roc_auc_score(y_eval, y_prob)))

plt.title('ROC for "Calm"', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend([model[1] for model in best_models])
# best model: logistic regression
# AUC score for Logistic Regression is 0.63
# AUC score for Random Forest is 0.57
# AUC score for Naive Bayes is 0.62
# AUC score for KNN is 0.58

fitted_model = logistic_reg(X_train, y_train).fit(X_train, y_train)

f1_score(y_test, fitted_model.predict(X_test)) # 0.52

pickle.dump(fitted_model, open('/Users/wasilaq/Metis/music-recommender/pickled/models/model_calm', 'wb'))