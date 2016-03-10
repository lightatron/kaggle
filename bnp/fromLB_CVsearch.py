# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:55:25 2016

@author: ben
"""

import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics as metrics
from sklearn.grid_search import RandomizedSearchCV

trainfile = "train.csv"
testfile = "test.csv"
outfile = "predictions.csv"

print('Load data...')
train = pd.read_csv(trainfile)
target = train['target'].values
train = train.drop(['ID','target'],axis=1)
test = pd.read_csv(testfile)
id_test = test['ID'].values
test = test.drop(['ID'],axis=1)

# this is Ben's
#train.drop(['v22', 'v56', 'v91'], axis=1, inplace=True)
#test.drop(['v22', 'v56', 'v91'], axis=1, inplace=True)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -9999 #train_series.mean()
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -9999 #train_series.mean()  #TODO

X_train = train
X_test = test

extc_base = ExtraTreesClassifier(n_jobs=-1)
param_dist = {"criterion" : ["gini", "entropy"],
              "n_estimators" : [700, 1000, 1500],
              "max_depth" : [20, 30, 60],
              "min_samples_split" : [2, 3],
              "min_samples_leaf" : [1, 2],
              "bootstrap" : [True, False],
              "max_features" : [0.5, 0.75] }  
              
n_iter_search = 80
random_search = RandomizedSearchCV(extc_base, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=4,
                                   scoring="log_loss",
                                   verbose=3)
                        
random_search.fit(train, target)
extc = random_search.best_estimator_ # choose the best one
extc.fit(X_train, target)
y_pred = extc.predict_proba(X_test)           
pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('extra_trees.csv',index=False)
              
#now fit the calibrated classifier
clf = CalibratedClassifierCV(extc, cv=5)
clf.fit(X_train, target)
y_pred = clf.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('extra_trees_calib.csv',index=False)