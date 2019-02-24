#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" support_vector_machine.py: Class containing common SVM learner methods.
Class is used loosely, the only class variable is data.
@author: Dominic Frecentese
"""

import matplotlib
matplotlib.use('Agg')		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
import sklearn as skl
import shared_methods as sm
from sklearn import model_selection as ms

class SupportVectorMachine(object):
    
    def __init__(self, data):
        self.data = data
    
    def show_learning_curve(self, learner):
        train_errors = []
        validation_errors = []
        train_examples = xrange(10, self.data.x_train.shape[0], 25)
        for i in train_examples:
            x_temp, x_remainder, y_temp, y_remainder = ms.train_test_split(self.data.x_train, self.data.y_train, train_size=i, random_state=11)
            train_accuracy, validation_accuracy = sm.kfold_validation(learner, x_temp, y_temp, k=5)
            train_errors.append(1 - train_accuracy)
            validation_errors.append(1 - validation_accuracy)
            
        img_path = self.data.image_dir + 'svm_learning_curve.png'
        plt.plot(train_examples, train_errors, label='Train Error')
        plt.plot(train_examples, validation_errors, label='Validation Error')
        plt.xlabel('Training Examples')
        plt.ylabel('Error')
        plt.title(self.data.name + ' SVM Learning Curve')
        plt.legend()
        plt.savefig(img_path)
        plt.show()
        plt.close()
        
    def find_optimal_hyperparameters(self, param_grid):   
        svm = skl.svm.SVC()
        grid_search = ms.GridSearchCV(svm, param_grid=param_grid, verbose=True)
        grid_search.fit(self.data.x_train, self.data.y_train)
        
        print grid_search.best_params_
        return grid_search.best_params_