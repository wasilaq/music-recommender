#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:27:07 2020

@author: wasilaq
"""

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def print_metrics(model, X, y, scoring='f1', oversample=False):
    if oversample==True:
        pipeline = make_pipeline(
            StandardScaler(), RandomOverSampler(random_state=11), model
            )
    else:
        pipeline = make_pipeline(
            StandardScaler(), model
            )
    score = cross_val_score(pipeline, X, y, scoring=scoring)    
    fitted_model = model.fit(X, y)
    cm = confusion_matrix(y, fitted_model.predict(X))
    print(score)
    print(cm) # confusion matrix to check if all values are classified in same class
# create class w/ functions for printing??
    
def logistic_reg(X, y, oversample=False, class_weight=None, C=1.0):
    model = LogisticRegression(class_weight=class_weight, C=C, random_state=11, max_iter=500)
    print_metrics(model, X, y, oversample=oversample)
    return model

def random_forest(X, y, n_estimators, max_depth, class_weight=None, oversample=False):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight, random_state=11)
    print_metrics(model, X, y, oversample=oversample)
    return model
    
def naive_bayes(X, y, oversample=False):
    model = GaussianNB()
    print_metrics(model, X, y, oversample=oversample)
    return model

def knn(X, y, n_neighbors, oversample=False):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    print_metrics(model, X, y, oversample=oversample)
    return model