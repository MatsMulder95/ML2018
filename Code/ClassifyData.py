import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, datasets, svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def classify(trainData, trainLabels, testData, type='logistic',run='non_local'):

    # Split data on local run
    if run == 'local':
        sub_index = 3000
        train_data_sub1 = trainData.sample(sub_index)
        train_labels_sub1 = trainLabels.iloc[train_data_sub1.index]
        train_data_sub2 = trainData.iloc[~trainData.index.isin(train_data_sub1.index)]
        train_labels_sub2 = trainLabels.iloc[train_data_sub2.index]

        trainData = train_data_sub1.reset_index(drop=True)
        trainLabels = train_labels_sub1.reset_index(drop=True)
        testData = train_data_sub2.reset_index(drop=True)
        testLabels = train_labels_sub2.reset_index(drop=True)

    # Classify using type
    if type == 'logistic':
        predict = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=2000).fit(trainData,trainLabels).predict(testData)

    # Console print result
    if run == 'local':
        result = 0
        for i in range(0, testLabels.shape[0]):
            if predict[i] == testLabels[i]:
                result = result + 1
        print(result / testLabels.shape[0])

    return predict
