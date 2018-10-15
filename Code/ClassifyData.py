import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, datasets, svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def classify(type='logistic'):
    # Import filtered data en labels
    trainData = pd.read_csv('data/filtered/train_data.csv', header=None)
    trainLabels = pd.read_csv('data/filtered/train_labels.csv', header=None)[0]
    testData = pd.read_csv('data/filtered/test_data.csv', header=None)

    # classify using type
    if type == 'logistic':
        predict = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=2000).fit(trainData,trainLabels).predict(testData)

    return predict
