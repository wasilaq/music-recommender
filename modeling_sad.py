#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:36:17 2020

@author: wasilaq
"""

from classification_modeling import *
import pickle
from numpy import arange
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

targets = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/targets')
emotion_subset = pd.read_pickle('/Users/wasilaq/Metis/music-recommender/pickled/emotions_df_subset')

identifiers = ['track_id','song_id','artist','song_title']

y = targets['sad']
X = emotion_subset.drop(identifiers+['emotion'], axis=1)
X_te, X_test, y_te, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25, random_state=12)

# logistic regression
for C in arange(0.1,10,0.1):
    print(C)
    print('Vanilla')
    logistic_reg(X_train, y_train, C=C)
    print('Balanced')
    logistic_reg(X_train, y_train, class_weight='balanced', C=C)
    print('Sampling')
    logistic_reg(X_train, y_train, oversample=True, C=C)
# best logistic regression: balanced, C=1
# [0.35555556 0.39502333 0.42944785 0.41627543 0.40330579]


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
# best random forest: balanced, n_estimators=4, max_depth=2
# [0.40168539 0.46403712 0.43221477 0.43376623 0.47073171]
# [[1225 1729]
# [ 433  897]]


# naive bayes
print('Vanilla')
naive_bayes(X_train, y_train)
print('Sampling')
naive_bayes(X_train, y_train, oversample=True)
# best naive bayes: sampling
# [0.30327869 0.3312369  0.35564854 0.32067511 0.31838565]
# [[2669  285]
# [1127  203]]


# knn
for neighbors in range(2,20):
    print(neighbors)
    print('Vanilla')
    knn(X_train, y_train, n_neighbors=neighbors)
    print('Sampling')
    knn(X_train, y_train, n_neighbors=neighbors, oversample=True)
# best knn: sampling, 13 neighbors
# [0.4128     0.38629283 0.40425532 0.44216691 0.40298507]
# [[2770  184]
# [1044  286]]


# compare models
best_models = [
    (logistic_reg(X_train, y_train, class_weight='balanced', C=1), 'Logistic Regression'),
    (random_forest(X_train, y_train, n_estimators=4, max_depth=2, class_weight='balanced'), 'Random Forest'),
    (naive_bayes(X_train, y_train, oversample=True), 'Naive Bayes'),
    (knn(X_train, y_train, n_neighbors=13, oversample=True), 'KNN')
    ]

for model, name in best_models:
    y_prob = (model.fit(X_train, y_train)).predict_proba(X_eval)[:,1]
    
    fpr, tpr, threshold = roc_curve(y_eval, y_prob)
    plt.plot(fpr, tpr)
    print('AUC score for {} is {:.2f}'.format(name, roc_auc_score(y_eval, y_prob)))

plt.title('ROC for "Sad"', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend([model[1] for model in best_models])
# best model: random forest
# AUC score for Logistic Regression is 0.55
# AUC score for Random Forest is 0.59
# AUC score for Naive Bayes is 0.54
# AUC score for KNN is 0.56

fitted_model = random_forest(X_train, y_train, n_estimators=4, max_depth=2, class_weight='balanced').fit(X_train, y_train)

f1_score(y_test, fitted_model.predict(X_test)) # 0.48

pickle.dump(fitted_model, open('/Users/wasilaq/Metis/music-recommender/pickled/models/model_sad', 'wb'))