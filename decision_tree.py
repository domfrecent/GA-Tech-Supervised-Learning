#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" decision_tree.py: Class containing common Decision Tree learner methods.
Class is used loosely, the only class variable is data.
@author: Dominic Frecentese
"""

import matplotlib
matplotlib.use('Agg')		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
from sklearn import tree, model_selection as ms
import numpy as np
import shared_methods as sm

class DecisionTree(object):
    
    def __init__(self, data):
        self.data = data

    def get_optimal_leaf_size(self, leaf_sizes):    
        train_accuracys = []
        validation_accuracys = []
        for i in leaf_sizes:
            dtl = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=i)
            train_accuracy, validation_accuracy = sm.kfold_validation(dtl, self.data.x_train, self.data.y_train, k=5)
            train_accuracys.append(train_accuracy)
            validation_accuracys.append(validation_accuracy)
            
        img_path = self.data.image_dir + 'dt_leaf_size_accuracy.png'
        plt.plot(leaf_sizes, train_accuracys, label='Train Accuracy')
        plt.plot(leaf_sizes, validation_accuracys, label='Validation Accuracy')
        plt.xlabel('Leaf Size')
        plt.ylabel('Accuracy')
        plt.title(self.data.name + ' Decision Tree Accuracy by Leaf Size')
        plt.legend()
        plt.show()
        plt.savefig(img_path)
        plt.close()
    
        temp_validation_acc = np.array(validation_accuracys)
        min_error = temp_validation_acc[1:].max()
        optimal_leaf_size = temp_validation_acc[1:].argmax() + 2
        return min_error, optimal_leaf_size

    def show_learning_curve(self, learner):
        train_errors = []
        validation_errors = []
        train_examples = xrange(10, self.data.x_train.shape[0], 25)
        for i in train_examples:
            x_temp, x_remainder, y_temp, y_remainder = ms.train_test_split(self.data.x_train, self.data.y_train, train_size=i, random_state=11)
            train_accuracy, validation_accuracy = sm.kfold_validation(learner,
                                                                      x_temp,
                                                                      y_temp,
                                                                      k=5)
            train_errors.append(1 - train_accuracy)
            validation_errors.append(1 - validation_accuracy)
            
        img_path = self.data.image_dir + 'dt_learning_curve.png'
        plt.plot(train_examples, train_errors, label='Train Error')
        plt.plot(train_examples, validation_errors, label='Validation Error')
        plt.xlabel('Training Examples')
        plt.ylabel('Error')
        plt.title(self.data.name + ' Decision Tree Learning Curve')
        plt.legend()
        plt.savefig(img_path)
        plt.show()
        plt.close()
