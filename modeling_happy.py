#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:43:42 2020

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

y = targets['happy']
X = emotion_subset.drop(identifiers+['emotion'], axis=1)
X_te, X_test, y_te, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=12)

# keep dummy classifier in code??
cross_val_score(DummyClassifier(random_state=2), X_train, y_train, cv=5, scoring='f1')
# [0.14229249, 0.12648221, 0.17391304, 0.14229249, 0.09486166]

# logistic regression
for C in arange(0.1,10,0.1):
    print(C)
    print('Vanilla')
    logistic_reg(X_train, y_train, C=C)
    print('Balanced')
    logistic_reg(X_train, y_train, class_weight='balanced', C=C)
    print('Sampling')
    logistic_reg(X_train, y_train, oversample=True, C=C)
# best logistic regression: balanced, C=5
# [0.31542461 0.31399317 0.27436823 0.306914   0.30155979]


# random forest
for n in range(2,10):
    print('{} is n_estimators'.format(n))
    for depth in range(2,10):
        print('{} is max depth'.format(depth))
        print('Vanilla')
        random_forest(X_train, y_train, n_estimators=n, max_depth=depth)
        print('Balanced')
        random_forest(X_train, y_train, n_estimators=n, max_depth=depth, class_weight='balanced')
        print('Sampling')
        random_forest(X_train, y_train, oversample=True, n_estimators=n, max_depth=depth)
# best random forest: balanced, n_estimators=10, max_depth=2
# [0.3183391  0.33655706 0.28145695 0.26086957 0.29821074]
# [[2409 1476]
# [ 287  397]]


# naive bayes
print('Vanilla')
naive_bayes(X_train, y_train)
print('Sampling')
naive_bayes(X_train, y_train, oversample=True)
# best naive bayes: sampling
# [0.29545455 0.30503979 0.28865979 0.28793774 0.29090909]
# [[3870   15]
# [ 680    4]]


# knn
for neighbors in range(8,15):
    print(neighbors)
    print('Vanilla')
    knn(X_train, y_train, n_neighbors=neighbors)
    print('Sampling')
    knn(X_train, y_train, n_neighbors=neighbors, oversample=True)
# best knn: sampling, 13 neighbors
# [0.27648115 0.29834254 0.28070175 0.25239006 0.31941924]
# [[3872   13]
# [ 662   22]]


# compare models
best_models = [
    (logistic_reg(X_train, y_train, class_weight='balanced', C=5), 'Logistic Regression'),
    (random_forest(X_train, y_train, n_estimators=10, max_depth=2, class_weight='balanced'), 'Random Forest'),
    (naive_bayes(X_train, y_train, oversample=True), 'Naive Bayes'),
    (knn(X_train, y_train, n_neighbors=13, oversample=True), 'KNN')
    ]

for model, name in best_models:
    y_prob = (model.fit(X_train, y_train)).predict_proba(X_eval)[:,1]
    
    fpr, tpr, threshold = roc_curve(y_eval, y_prob)
    plt.plot(fpr, tpr)
    print('AUC score for {} is {:.2f}'.format(name, roc_auc_score(y_eval, y_prob)))

plt.title('ROC for "Happy"', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend([model[1] for model in best_models])
# best model: logistic regression

fitted_model = logistic_reg(X_train, y_train, class_weight='balanced', C=5).fit(X_train, y_train)

f1_score(y_test, fitted_model.predict(X_test)) # 0.32

pickle.dump(fitted_model, open('/Users/wasilaq/Metis/music-recommender/pickled/models/model_happy', 'wb'))