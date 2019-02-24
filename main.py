#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Examine model performance across multiple learners and datasets.

@author: Dominic Frecentese
"""

import sklearn as skl
from sklearn import neural_network as sknn
from sklearn import svm
import matplotlib.pyplot as plt
import shared_methods as sm
import decision_tree as dt
import learner_data as ld
import boosted_dt as bdt
import knn
import neural_network as nn
import support_vector_machine as support_vm
from tabulate import tabulate


if __name__ == "__main__":
    """
    Import 2 datasets and create 5 ML models for each. 
    
    Datasets:
        Mammogram Mass
        Abalone Age
        
    ML Models:
        Decision Tree
        AdaBoost
        K-Nearest Nieghbor
        Neural Network
        Support Vector Machine
    """ 

    # Toggle functionality is provided for each dataset/model combination.
    mammogram_dt_on = True
    mammogram_boost_on = True
    mammogram_knn_on = True
    mammogram_neural_net_on = True
    mammogram_svm_on = True
    abalone_dt_on = True
    abalone_boost_on = True
    abalone_knn_on = True
    abalone_neural_net_on = True
    abalone_svm_on = True

    mammogram_path = 'data/mammographic-mass/mammographic_masses.txt'
    mammogram = ld.LearnerData('Mammogram', mammogram_path, ',')
    mammogram.drop_columns(['BI-RADS Assessment'])
    mammogram.drop_rows_missing_data()
    mammogram.split_train_test(target_column='Severity', train_size=0.8)
    mammogram_results = []
    headers=['Learner', 'Train Acc(%)', 'Test Acc(%)', 'Train Time(s)', 'Query Time(s)']

    # Mammogram - Decision Tree
    if mammogram_dt_on:
        mammogram_dt = dt.DecisionTree(mammogram)

        leaf_sizes = xrange(1, 200)
        min_error, optimal_leaf_size = mammogram_dt.get_optimal_leaf_size(leaf_sizes)
        dtl = skl.tree.DecisionTreeClassifier(criterion='gini',
                                              min_samples_leaf=optimal_leaf_size)
        mammogram_dt.show_learning_curve(dtl)
        accuracy = sm.get_model_accuracy(dtl, mammogram, 'Decision Tree')
        results = [accuracy]
        mammogram_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    # Mammogram - Boosting
    if mammogram_boost_on:
        mammogram_bdt = bdt.BoostedDT(mammogram)
        param_grid = {"n_estimators": [1,25,50,75,100,150,200],
            "learning_rate": [0.001, 0.01, 0.03, 0.1, 0.3]
             }
#        learning_rate, n_estimators = mammogram_bdt.find_optimal_hyperparameters(dtl, param_grid)
        learning_rate = 0.001
        n_estimators = 200
        boost_learner = skl.ensemble.AdaBoostClassifier(dtl, n_estimators=n_estimators, learning_rate=learning_rate)

        mammogram_bdt.get_learning_curve(boost_learner)
        mammogram_bdt.find_optimal_num_estimators(dtl, learning_rate)

        accuracy = sm.get_model_accuracy(boost_learner, mammogram, 'Boosting (DT)')
        results = [accuracy]
        mammogram_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    # Mammogram - K-Nearest Neighbors
    if mammogram_knn_on:
        mammogram_knn = knn.KNN(mammogram)
        min_error, optimal_neighbors = mammogram_knn.find_optimal_k()
        knnl = skl.neighbors.KNeighborsClassifier(n_neighbors=38, weights='uniform', p=1)
        mammogram_knn.show_learning_curve(knnl)

        accuracy = sm.get_model_accuracy(knnl, mammogram, 'KNN')
        results = [accuracy]
        mammogram_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    # Mammogram Neural Network
    if mammogram_neural_net_on:
        mammogram_nn = nn.NeuralNetwork(mammogram)
#        optimal_params = mammogram_nn.find_optimal_hyperparameters()
        nnl = sknn.MLPClassifier(solver='adam', alpha=0.001, \
                                hidden_layer_sizes=(8), \
                                activation='logistic', \
                                random_state=10,
                                max_iter=250)
        mammogram_nn.get_learning_curve_by_epoch(nnl, 200)
        mammogram_nn.show_learning_curve(nnl)

        accuracy = sm.get_model_accuracy(nnl, mammogram, 'Neural Network')
        results = [accuracy]
        mammogram_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    # Mammogram - Support Vector Machine
    if mammogram_svm_on:
        mammogram_svm = support_vm.SupportVectorMachine(mammogram)
        param_grid = {"kernel": ['linear'],
            "C": [0.001, 0.01, 0.1, 1, 10, 30],
            "gamma":[0.001, 0.01, 0.1, 1]
             }
#        mammogram_svm.find_optimal_hyperparameters()
        svml = svm.SVC(kernel='rbf', C=10, gamma=0.01)
        mammogram_svm.show_learning_curve(svml)
        
        accuracy = sm.get_model_accuracy(svml, mammogram, 'SVM (RBF)')
        results = [accuracy]
        mammogram_results.append(accuracy)
        
        svml = svm.SVC(kernel='linear', C=10, gamma=0.01)
        accuracy = sm.get_model_accuracy(svml, mammogram, 'SVM (Linear)')
        results.append(accuracy)
        mammogram_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    abalone_path = 'data/abalone/abalone.txt'
    names = ['Sex',
             'Length',
             'Diameter',
             'Height',
             'Whole weight',
             'Shucked weight',
             'Viscera weight',
             'Shell weight',
             'Rings']
    abalone = ld.LearnerData('Abalone', abalone_path, ',', header_names=names)
    abalone.one_hot_encode(['Sex'])

    plt.hist(abalone.df['Rings'])
    plt.xlabel('Rings')
    plt.ylabel('# Examples')
    plt.savefig(abalone.image_dir + 'rings_distribution.png')
    plt.close()

    abalone.discretize_abalone()
    abalone.split_train_test(target_column='Rings-B', train_size=0.8)
    print abalone.df.groupby('Rings-B').count()
    abalone_results = []

    # Abalone Age - Decision Tree
    if abalone_dt_on:
        abalone_dt = dt.DecisionTree(abalone)

        leaf_sizes = xrange(1,200)
        min_error, optimal_leaf_size = abalone_dt.get_optimal_leaf_size(leaf_sizes)
        print optimal_leaf_size, min_error
        dtl = skl.tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=optimal_leaf_size)

        abalone_dt.show_learning_curve(dtl)
        accuracy = sm.get_model_accuracy(dtl, abalone, 'Decision Tree')
        results = [accuracy]
        abalone_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print         

    # Abalone Age - Boosting
    if abalone_boost_on:
        abalone_bdt = bdt.BoostedDT(abalone)
        param_grid = {"n_estimators": [1,25,50,75,100,150,200],
            "learning_rate": [0.001, 0.01, 0.03, 0.1, 0.3]
             }
#        learning_rate, n_estimators = abalone_bdt.find_optimal_hyperparameters(dtl, param_grid)
        learning_rate = 0.01
        n_estimators = 50
        boost_learner = skl.ensemble.AdaBoostClassifier(dtl, n_estimators=n_estimators, learning_rate=learning_rate)

        abalone_bdt.get_learning_curve(boost_learner)
        abalone_bdt.find_optimal_num_estimators(dtl, learning_rate=learning_rate)
        accuracy = sm.get_model_accuracy(boost_learner, abalone, 'Boosting (DT)')
        results = [accuracy]
        abalone_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    # Abalone Age - K-Nearest Neighbors
    if abalone_knn_on:
        abalone_knn = knn.KNN(abalone)
        min_error, optimal_neighbors = abalone_knn.find_optimal_k()
        knnl = skl.neighbors.KNeighborsClassifier(n_neighbors=optimal_neighbors, weights='uniform', p=1)
        abalone_knn.show_learning_curve(knnl)

        accuracy = sm.get_model_accuracy(knnl, abalone, 'KNN')
        results = [accuracy]
        abalone_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    # Abalone Age - Neural Network
    if abalone_neural_net_on:
        abalone_nn = nn.NeuralNetwork(abalone)
#        optimal_params = abalone_nn.find_optimal_hyperparameters()
        nnl = sknn.MLPClassifier(solver='adam', alpha=0.01, \
                                hidden_layer_sizes=(8), \
                                activation='relu', \
                                random_state=10,
                                max_iter=250)
        abalone_nn.get_learning_curve_by_epoch(nnl, 200)
        abalone_nn.show_learning_curve(nnl)

        accuracy = sm.get_model_accuracy(nnl, abalone, 'Neural Network')
        results = [accuracy]
        abalone_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    # Abalone Age - Support Vector Machine
    if abalone_svm_on:
        abalone_svm = support_vm.SupportVectorMachine(abalone)
        param_grid = {"kernel": ['linear'],
            "C": [0.001, 0.01, 0.1, 1, 10, 30],
            "gamma":[0.001, 0.01, 0.1, 1]
             }
#        abalone_svm.find_optimal_hyperparameters()
        svml = svm.SVC(kernel='rbf', C=30, gamma=0.1)
        abalone_svm.show_learning_curve(svml)
        
        accuracy = sm.get_model_accuracy(svml, abalone, 'SVM (RBF)')
        results = [accuracy]
        abalone_results.append(accuracy)

        svml = svm.SVC(kernel='linear', C=10, gamma=0.001)
        accuracy = sm.get_model_accuracy(svml, abalone, 'SVM (Linear)')
        results.append(accuracy)
        abalone_results.append(accuracy)
        print(tabulate(results, headers=headers))
        print

    print(tabulate(abalone_results, headers=headers))
