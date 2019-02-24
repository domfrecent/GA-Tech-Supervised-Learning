#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" knn.py: Class containing common KNN learner methods.
Class is used loosely, the only class variable is data.

@author: Dominic Frecentese
"""

import matplotlib
matplotlib.use('Agg')		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
import sklearn as skl
import shared_methods as sm
from sklearn import model_selection as ms
import numpy as np

class KNN(object):
    
    def __init__(self, data):
        self.data = data
    
    def find_optimal_k(self):
        n_neighbors = xrange(1,150)
        train_accuracies = []
        validation_accuracies = []
        for i in n_neighbors:
            knnl = skl.neighbors.KNeighborsClassifier(n_neighbors=i, weights='uniform', p=1)
            train_acc, validation_acc = sm.kfold_validation(knnl, self.data.x_train, self.data.y_train, k=5)
            train_accuracies.append(train_acc)
            validation_accuracies.append(validation_acc)
        
        img_path = self.data.image_dir + 'knn_num_neighbors.png'
        plt.plot(n_neighbors, train_accuracies, label='Train Accuracy')
        plt.plot(n_neighbors, validation_accuracies, label='Validation Accuracy')
        plt.xlabel('# Neighbors')
        plt.ylabel('Accuracy')
        plt.title(self.data.name + ' KNN Accuracy by # Neighbors')
        plt.legend()
        plt.savefig(img_path)
        plt.show()
        plt.close()
        
        temp_validation_acc = np.array(validation_accuracies)
        min_error = temp_validation_acc.max()
        optimal_num_neighbors = temp_validation_acc.argmax() + 1
        
        return min_error, optimal_num_neighbors

        
    def find_optimal_hyperparameters(self):
        param_grid = {"n_neighbors": xrange(10,200),
            "p": [1,2],
            "weights":['uniform', 'distance']
             }
        
        knn_learner = skl.neighbors.KNeighborsClassifier()
        grid_search = ms.GridSearchCV(knn_learner, param_grid=param_grid, verbose=True)
        grid_search.fit(self.data.x_train, self.data.y_train)
        
        print grid_search.best_params_
        return grid_search.best_params_
    
    def show_learning_curve(self, learner):
        train_errors = []
        validation_errors = []
        train_examples = xrange(80, self.data.x_train.shape[0], 25)
        for i in train_examples:
            x_temp, x_remainder, y_temp, y_remainder = ms.train_test_split(self.data.x_train, self.data.y_train, train_size=i, random_state=11)
            train_accuracy, validation_accuracy = sm.kfold_validation(learner, x_temp, y_temp, k=5)
            train_errors.append(1 - train_accuracy)
            validation_errors.append(1 - validation_accuracy)

        img_path = self.data.image_dir + 'knn_learning_curve.png'
        plt.plot(train_examples, train_errors, label='Train Error')
        plt.plot(train_examples, validation_errors, label='Validation Error')
        plt.xlabel('Training Examples')
        plt.ylabel('Error')
        plt.title(self.data.name + ' KNN Learning Curve')
        plt.legend()
        plt.savefig(img_path)
        plt.show()
        plt.close()
