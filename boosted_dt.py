#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" boosted_dt.py: Class containing common Boosting learner methods.
Class is used loosely, the only class variable is data.

@author: Dominic Frecentese
"""

import matplotlib
matplotlib.use('Agg')		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
from sklearn import ensemble, model_selection as ms
import shared_methods as sm

class BoostedDT():
    
    def __init__(self, data):
        self.data = data
    
    def find_optimal_num_estimators(self, base_learner, learning_rate):
        num_estimators = [1,25,50,75,100,150,200]
        print num_estimators
        train_accuracies = []
        validation_accuracies = []
        for i in num_estimators:
            print i
            boost_learner = ensemble.AdaBoostClassifier(base_learner, n_estimators=i, learning_rate=learning_rate)
            train_acc, validation_acc = sm.kfold_validation(boost_learner, self.data.x_train, self.data.y_train, k=5)
            
            train_accuracies.append(train_acc)
            validation_accuracies.append(validation_acc)
        
        img_path = self.data.image_dir + 'boosted_dt_num_estimators.png'
        plt.plot(num_estimators, train_accuracies, label='Train Accuracy')
        plt.plot(num_estimators, validation_accuracies, label='Validation Accuracy')
        plt.xlabel('# Estimators')
        plt.ylabel('Accuracy')
        plt.title(self.data.name + ' Boosted DT Accuracy')
        plt.legend()
        plt.savefig(img_path)
#        plt.show()
        plt.close()
    
    def get_learning_curve(self, learner):
        step_size = self.data.x_train.shape[0] / 25
        steps = xrange(10, self.data.x_train.shape[0], step_size)
        train_errors = []
        validation_errors = []
        
        x_vals = []
        for i in steps:
            print i
            x_vals.append(i)
            x_temp, x_remainder, y_temp, y_remainder = ms.train_test_split(self.data.x_train, self.data.y_train, train_size=i, random_state=11)
    #        print x_temp, y_temp
            train_accuracy, validation_accuracy = sm.kfold_validation(learner, x_temp, y_temp, k=5)
            train_errors.append(1 - train_accuracy)
            validation_errors.append(1 - validation_accuracy)
            
        img_path = self.data.image_dir + 'boosted_dt_learning_curve.png'
        plt.plot(steps, train_errors, label='Train Error')
        plt.plot(steps, validation_errors, label='Validation Error')
        plt.xlabel('Training Examples')
        plt.ylabel('Error')
        plt.title(self.data.name + ' Boosted Decision Tree Learning Curve')
        plt.legend()
        plt.savefig(img_path)
        plt.show()
        plt.close()
    
    def find_optimal_hyperparameters(self, base_learner, param_grid):
        
        boost_learner = ensemble.AdaBoostClassifier(base_learner)
        grid_search = ms.GridSearchCV(boost_learner, param_grid=param_grid, verbose=True)
        grid_search.fit(self.data.x_train, self.data.y_train)
        print grid_search.best_params_
        
        return grid_search.best_params_['learning_rate'], grid_search.best_params_['n_estimators']
        
