# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:51:35 2016

@author: ben
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
import csv as csv
import sklearn.metrics as metrics
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.grid_search import RandomizedSearchCV
from time import time

trainfile = "train.csv"
testfile = "test.csv"
outfile = "predictions.csv"

train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)

ids = test['ID'].values
test = test.drop(['ID'], axis=1)
train = train.drop(['ID'], axis=1)

# has 18211/114321 unique values
target = train['target']
train.drop(['target'], axis=1, inplace=True)
train.drop(['v22', 'v56', 'v91'], axis=1, inplace=True)
test.drop(['v22', 'v56', 'v91'], axis=1, inplace=True)

types = train.dtypes
categorical_features = list(types[(types != "float64") & (types != "int64")].index)
numerical_features = list(types[(types == "float64")].index)
hybrid_features = list(types[(types == "int64")].index)
cat_hyb_features = list(types[((types != "float64") & (types != "int64")) | (types == "int64")].index)

"""
steps: there are categorical as well as numerical data - think about this.
#here are a lot of nans in both training and testing - in fact 0.33 of the data is nans!

interestingly there are several rows where ALL NON ACTEGORICAL features 
are nans - this is fairly constant. Maybe write something like predicting the values
of this null elements from the non-null categorical features. Or just fill in as means.

if H = train[categorical_features], shows 6% of categorical features are null, make these the most common values
"""

def fill_to_most_common(df, feat):
    for feat in feat:
        nan_index = df.loc[:, feat][df.loc[:, feat].isnull() == True].index
        mode_val = df.loc[:, feat][df.loc[:, feat].isnull() != True].mode().values
        df.loc[nan_index, feat] = mode_val
        
    return df

def make_categorical_numerical(df):
    pass

train = fill_to_most_common(train, categorical_features)
test = fill_to_most_common(test, categorical_features)
train = fill_to_most_common(train, hybrid_features)
test = fill_to_most_common(test, hybrid_features)

assert train[categorical_features].isnull().sum().sum() == 0
assert test[categorical_features].isnull().sum().sum() == 0

for feat in categorical_features:
    cat_enc = LabelEncoder()
    cat_enc.fit(pd.concat([train[feat], test[feat]]))
    train[feat] = cat_enc.transform(train[feat])
    test[feat] = cat_enc.transform(test[feat])
    
# now impute missing values
imp = Imputer(strategy="mean")
imp.fit(train)
train.iloc[:, :] = imp.transform(train)
imp = Imputer(strategy="mean")   
imp.fit(test)
test.iloc[:, :] = imp.transform(test)

# now one-hot encode categorical features
enc2 = OneHotEncoder(sparse=False)
enc2.fit(train[categorical_features])
new_train_feats = pd.DataFrame(enc2.transform(train[categorical_features]), index=train.index)
new_test_feats = pd.DataFrame(enc2.transform(test[categorical_features]), index=test.index)

train = pd.concat([train, new_train_feats], axis=1)
test = pd.concat([test, new_test_feats], axis=1)
train.drop(categorical_features, axis=1, inplace=True)
test.drop(categorical_features, axis=1, inplace=True)

#train.fillna(0, inplace=True)
#test.fillna(0, inplace=True)

#for col in numerical_features:
#    imp = Imputer()
#    imp.fit(train)
#    train.loc[:, col] = train.loc[:, col].notnull() == True
#    test.loc[:, col].isnull() = test.loc[:, col].notnull().mean()
  
if __name__ == "__main__":
#    train, train_cv, target, target_cv = train_test_split(train, target, test_size=0.05, random_state=42)  
#    clf = GradientBoostingClassifier(n_estimators=300, 
#                                     max_depth=5, max_features='sqrt', 
#                                     learning_rate=0.1,
#                                     loss='deviance')
                                     
    clf = GradientBoostingClassifier(n_estimators=350)
    param_dist = {"loss" : ["deviance", "exponential"],
                  "learning_rate" : [0.01, 0.03, 0.05, 0.07, 0.1],
                  #"n_estimators" : [300, 400, 500, 900],
                  "max_depth" : [5, 6, 7],
                  "min_samples_split" : [2, 3, 4, 5],
                  "min_samples_leaf" : [1, 2, 3],
                  "subsample" : [1.0, 0.99, 0.985],
                  "max_features" : [0.05 ,0.1, 0.2] }                             
     
    n_iter_search = 100
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=4,
                                       scoring="log_loss",
                                       n_jobs=6, verbose=3)
    
    start = time()                         
    random_search.fit(train, target)  
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
      
    clf2 = random_search.best_estimator_
    clf = CalibratedClassifierCV(clf2, method='sigmoid', cv=5)
    #clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, n_jobs=6)
    clf.fit(train, target)
    
    y_pred_prob = clf.predict_proba(test)[:, 1]
    
    predictions_file = open(outfile, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID","PredictedProb"])
    open_file_object.writerows(zip(ids, y_pred_prob))
    predictions_file.close()
    print 'Done.'

#for col in train[hybrid_features]:
#    print col, np.unique(train[col]).shape

y_hat = clf.predict(train_cv)
print metrics.classification_report(target_cv, y_hat)
print metrics.accuracy_score(target_cv, y_hat)
print "log loss: "  + str(metrics.log_loss(target_cv, clf.predict_proba(train_cv)))

calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
calibrated_clf.fit(train, target)


#clf = GradientBoostingClassifier(init=None, learning_rate=0.03, loss='exponential',
#              max_depth=6, max_features=0.5, max_leaf_nodes=None,
#              min_samples_leaf=1, min_samples_split=4,
#              min_weight_fraction_leaf=0.0, n_estimators=300,
#              presort='auto', random_state=None, subsample=0.975,
#              verbose=0, warm_start=False)

#GradientBoostingClassifier(init=None, learning_rate=0.03, loss='deviance',
#              max_depth=7, max_features=0.2, max_leaf_nodes=None,
#              min_samples_leaf=2, min_samples_split=4,
#              min_weight_fraction_leaf=0.0, n_estimators=350,
#              presort='auto', random_state=None, subsample=1.0, verbose=0,
#              warm_start=False)
