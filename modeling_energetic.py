#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:20:50 2020

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

y = targets['energetic']
y.value_counts() # imbalanced

X = emotion_subset.drop(identifiers+['emotion'], axis=1)
X_te, X_test, y_te, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=12)

# keep dummy classifier in code??
cross_val_score(DummyClassifier(random_state=2), X_train, y_train, cv=5, scoring='f1')
# [0.14229249, 0.12648221, 0.17391304, 0.14229249, 0.09486166]

# logistic regression
for C in arange(0.1,1,0.1):
    print(C)
    print('Vanilla')
    logistic_reg(X_train, y_train, C=C)
    print('Balanced')
    logistic_reg(X_train, y_train, class_weight='balanced', C=C)
    print('Sampling')
    logistic_reg(X_train, y_train, oversample=True, C=C)
# best logistic regression: balanced, C=0.5
# [[0.21449275 0.22727273 0.2234957  0.20224719 0.21902017]


# random forest
for n in range(2,10):
    print('{} is n_estimators'.format(n))
    for depth in [9]:
        print('{} is max depth'.format(depth))
        print('Vanilla')
        random_forest(X_train, y_train, n_estimators=n, max_depth=depth)
        print('Balanced')
        random_forest(X_train, y_train, n_estimators=n, max_depth=depth, class_weight='balanced')
        print('Sampling')
        random_forest(X_train, y_train, oversample=True, n_estimators=n, max_depth=depth)
# best random forest: balanced, n_estimators=5, max_depth=9
# [0.3190184  0.36090226 0.32653061 0.34965035 0.30167598]
# [[3707  330]
# [  32  215]]


# naive bayes
print('Vanilla')
naive_bayes(X_train, y_train)
print('Sampling')
naive_bayes(X_train, y_train, oversample=True)
# best naive bayes: sampling
# [0.29677419 0.23853211 0.22988506 0.224      0.31147541]
# [[3710  327]
# [ 164   83]]


# knn
for neighbors in range(2,15):
    print(neighbors)
    #print('Vanilla')
    #knn(X_train, y_train, n_neighbors=neighbors)
    #print('Sampling')
    knn(X_train, y_train, n_neighbors=neighbors, oversample=True)
# best knn: sampling, 3 neighbors
# [[0.39344262 0.35036496 0.4        0.34710744 0.34285714]
# [[4004   33]
# [ 133  114]]


# compare models
best_models = [
    (logistic_reg(X_train, y_train, class_weight='balanced', C=0.5), 'Logistic Regression'),
    (random_forest(X_train, y_train, n_estimators=5, max_depth=9, class_weight='balanced'), 'Random Forest'),
    (naive_bayes(X_train, y_train, oversample=True), 'Naive Bayes'),
    (knn(X_train, y_train, n_neighbors=3, oversample=True), 'KNN')
    ]

for model, name in best_models:
    y_prob = (model.fit(X_train, y_train)).predict_proba(X_eval)[:,1]
    
    fpr, tpr, threshold = roc_curve(y_eval, y_prob)
    plt.plot(fpr, tpr)
    print('AUC score for {} is {:.2f}'.format(name, roc_auc_score(y_eval, y_prob)))

plt.title('ROC for "Energetic"', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend([model[1] for model in best_models])
# best model: logistic regression
# AUC score for Logistic Regression is 0.78
# AUC score for Random Forest is 0.79
# AUC score for Naive Bayes is 0.73
# AUC score for KNN is 0.69

fitted_model = logistic_reg(X_train, y_train, class_weight='balanced', C=0.5).fit(X_train, y_train)

f1_score(y_test, fitted_model.predict(X_test)) # 0.16

pickle.dump(fitted_model, open('/Users/wasilaq/Metis/music-recommender/pickled/models/model_energetic', 'wb'))