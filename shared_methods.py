#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" shared_methods.py: Contains common methods shared across learners.
@author: Dominic Frecentese
"""

import sklearn as skl
from sklearn import model_selection as ms
import numpy as np
import time

def kfold_validation(learner, x, y, k=5):
    kf = ms.KFold(n_splits=k, shuffle=True, random_state=15)
    kfold_train_accuracy = []
    kfold_validation_accuracy = []
    
    for train_index, validation_index in kf.split(x):
        x_train, x_validation = x[train_index], x[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]
        learner = learner.fit(x_train, y_train)
        y_train_predict = learner.predict(x_train)
        y_validation_predict = learner.predict(x_validation)
        train_error = skl.metrics.zero_one_loss(y_train, y_train_predict)
        validation_error = skl.metrics.zero_one_loss(y_validation, y_validation_predict)

        kfold_train_accuracy.append(1 - train_error)
        kfold_validation_accuracy.append(1 - validation_error)
    
    train_accuracy = sum(kfold_train_accuracy) / len(kfold_train_accuracy)
    validation_accuracy = sum(kfold_validation_accuracy) / len(kfold_validation_accuracy)
    
    return train_accuracy, validation_accuracy

def kfold_validation_nn(learner, x, y, k=5):
    kf = ms.KFold(n_splits=k, shuffle=True, random_state=15)
    kfold_train_accuracy = []
    kfold_validation_accuracy = []
    
    for train_index, validation_index in kf.split(x):
        x_train, x_validation = x[train_index], x[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]
        learner = learner.partial_fit(x_train, y_train, np.unique(y_train))

        y_train_predict = learner.predict(x_train)
        y_validation_predict = learner.predict(x_validation)
        train_error = skl.metrics.zero_one_loss(y_train, y_train_predict)
        validation_error = skl.metrics.zero_one_loss(y_validation, y_validation_predict)

        kfold_train_accuracy.append(1 - train_error)
        kfold_validation_accuracy.append(1 - validation_error)
    
    train_accuracy = sum(kfold_train_accuracy) / len(kfold_train_accuracy)
    validation_accuracy = sum(kfold_validation_accuracy) / len(kfold_validation_accuracy)
    
    return train_accuracy, validation_accuracy

def print_accuracy(train_accuracy, test_accuracy):
    print "Train Accuracy:"
    print "  " + str(train_accuracy)
    print
    print "Test Accuracy:"
    print "  " + str(test_accuracy)
    print

def get_model_accuracy(learner, data, learner_name):
    start = time.time()
    learner.fit(data.x_train, data.y_train)
    end = time.time()
    train_time = end - start
    
    start = time.time()
    y_train_predict = learner.predict(data.x_train)
    end = time.time()
    query_time = end - start

    y_test_predict = learner.predict(data.x_test)
    train_accuracy = (1 - skl.metrics.zero_one_loss(data.y_train, y_train_predict)) * 100
    test_accuracy = (1 - skl.metrics.zero_one_loss(data.y_test, y_test_predict)) * 100
    return [learner_name, train_accuracy, test_accuracy, train_time, query_time]
