import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors, datasets
from matplotlib.colors import ListedColormap
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import csv

from ClassifyData import *

# importing data
train_data = pd.read_csv('data/train_data.csv', header=None)
train_labels = pd.read_csv('data/train_labels.csv', header=None)[0]
test_data = pd.read_csv('data/test_data.csv', header=None)

train_data['label'] = train_labels

# Create random subsets
sub_index = train_data.shape[0] // 2
train_data_sub1 = train_data.sample(sub_index)
train_labels_sub1 = train_labels.iloc[train_data_sub1.index]
train_data_sub2 = train_data.iloc[~train_data.index.isin(train_data_sub1.index)]
train_labels_sub2 = train_labels.iloc[train_data_sub2.index]

train_data_sub1 = train_data_sub1.reset_index(drop=True)
train_labels_sub1 = train_labels_sub1.reset_index(drop=True)
train_data_sub2 = train_data_sub2.reset_index(drop=True)
train_labels_sub2 = train_labels_sub2.reset_index(drop=True)

# Set input
X_train = train_data_sub1   # Input train data
y_train = train_labels_sub1    # Input train labels
X_validate = train_data_sub2 # Input validate data
y_validate = train_labels_sub2   # Input validate labels

# Set input
X_train = train_data_sub1   # Input train data
y_train = train_labels_sub1    # Input train labels
X_validate = train_data_sub2 # Input validate data
y_validate = train_labels_sub2   # Input validata labels

# Kneighbors
#predict = neighbors.KNeighborsClassifier(20, 'uniform',algorithm='auto').fit(X_train, y_train).predict(X_validate)

#Kneighbors
#predict = neighbors.KNeighborsClassifier(20, 'uniform',algorithm='auto').fit(X_train, y_train).predict(X_validate)


#predict = svm.SVC(kernel='poly', degree=3).fit(X_train,y_train).predict(X_validate)

#Logistic BEST!!!
#predict = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=2000).fit(X_train, y_train).predict(X_validate)

predict = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 10), random_state=1, activation='logistic', max_iter=2100).fit(X_train, y_train).predict(X_validate)

#Validate output
result = 0
for i in range(0,X_validate.shape[0]):
    if predict[i] == y_validate[i]:
        result = result + 1
result = result / X_validate.shape[0]

print(result)

#Write to CSV
with open("data/output.csv", "w", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["Sample_id","Sample_label"])
    for i in range(0,X_validate.shape[0]):
        wr.writerow([i+1,str(predict[i].item())])
