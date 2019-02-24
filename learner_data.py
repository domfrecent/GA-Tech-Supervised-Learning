#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" learner_data.py: Class for learner data.
Provides common representation of data and methods that can be used across
learners and datasets.

@author: Dominic Frecentese
"""

import pandas as pd
import numpy as np
from sklearn import model_selection as ms
from sklearn import preprocessing

class LearnerData(object):
    
    def __init__(self, name, path, delimeter, header_names = []):
        self.name = name
        self.df = self.read_csv(path, delimeter, header_names)
        image_dir = 'report-images/' + name.lower().replace(' ', '-') + '/'
        self.image_dir = image_dir
        
    def read_csv(self, path, delimeter, header_names):
        if header_names == []:    
            df = pd.read_csv(path, sep=delimeter)
        else:
            df = pd.read_csv(path, sep=delimeter, header=None, names=header_names)  
        return df
    
    def drop_rows_missing_data(self):
        self.df = self.df.replace('?', np.nan)
        self.df = self.df.dropna(axis=0)
    
    def drop_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)
        
    def one_hot_encode(self, columns):
        for column in columns:
            self.df[column] = pd.Categorical(self.df[column])
            temp_one_hot = pd.get_dummies(self.df[column], prefix='is')
            self.df = pd.concat([self.df, temp_one_hot], axis=1)
            self.df = self.df.drop([column], axis=1)
    
    def split_train_test(self, target_column, train_size=0.8):
        y = self.df[target_column]
        x = self.df.drop([target_column], axis=1)
        
        y = y.values
        x = x.values
        x = self.scale_features(x)
        
        x_train, x_test, y_train, y_test = ms.train_test_split(x, y, train_size=train_size, random_state=11)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
    
    def scale_features(self, x):
        return preprocessing.scale(x)
    
    def discretize_abalone(self):
        self.df['Rings-B'] = ''
        self.df['Rings-B'][self.df['Rings'] <= 5] = '1-5'
        self.df['Rings-B'][(self.df['Rings'] > 5) & (self.df['Rings'] <= 10)] = '6-10'
        self.df['Rings-B'][(self.df['Rings'] > 10) & (self.df['Rings'] <= 15)] = '10-15'
        self.df['Rings-B'][(self.df['Rings'] > 15)] = '15+'
        self.df = self.df.drop(['Rings'], axis=1)