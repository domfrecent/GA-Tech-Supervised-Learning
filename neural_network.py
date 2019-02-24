#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" neural_network.py: Class containing common Neural Network learner methods.
Class is used loosely, the only class variable is data.

@author: Dominic Frecentese
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared_methods as sm
from sklearn import model_selection as ms, neural_network as sknn
import numpy as np

class NeuralNetwork(object):
    
    def __init__(self, data):
        self.data = data
    
    def show_learning_curve(self, learner):
        train_errors = []
        validation_errors = []
        step_size = self.data.x_train.shape[0] / 25        
        train_examples = xrange(10, self.data.x_train.shape[0], step_size)
        for i in train_examples:
            x_temp, x_remainder, y_temp, y_remainder = ms.train_test_split(self.data.x_train, self.data.y_train, train_size=i, random_state=11)
            train_accuracy, validation_accuracy = sm.kfold_validation(learner, x_temp, y_temp, k=5)
            train_errors.append(1 - train_accuracy)
            validation_errors.append(1 - validation_accuracy)

        img_path = self.data.image_dir + 'nn_learning_curve.png'
        plt.plot(train_examples, train_errors, label='Train Error')
        plt.plot(train_examples, validation_errors, label='Validation Error')
        plt.xlabel('Training Examples')
        plt.ylabel('Error')
        plt.title(self.data.name + ' Neural Network Learning Curve')
        plt.legend()
        plt.savefig(img_path)
        plt.show()
        plt.close()

    def find_optimal_hyperparameters(self):
        param_grid = {"hidden_layer_sizes": [(i,) for i in xrange(2,9)],
            "alpha": 10.0 ** -np.arange(1, 7),
            "activation":['logistic', 'relu']
             }
        
        nn = sknn.MLPClassifier()
        grid_search = ms.GridSearchCV(nn, param_grid=param_grid, verbose=True)
        grid_search.fit(self.data.x_train, self.data.y_train)
        
        print grid_search.best_params_
        return grid_search.best_params_
        
    def get_learning_curve_by_epoch(self, learner, max_iter):
        train_errors = []
        validation_errors = []
        epochs = xrange(1, max_iter, 10)
        for i in epochs:
            train_accuracy, validation_accuracy = sm.kfold_validation_nn(learner, self.data.x_train, self.data.y_train)
            train_errors.append(1 - train_accuracy)
            validation_errors.append(1 - validation_accuracy)
        
        img_path = self.data.image_dir + 'nn_error_by_epoch.png'
        plt.plot(epochs, train_errors, label='Train Error')
        plt.plot(epochs, validation_errors, label='Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(self.data.name + ' Neural Network Error by Epoch')
        plt.legend()
        plt.savefig(img_path)
        plt.show()
        plt.close()
        
        
    