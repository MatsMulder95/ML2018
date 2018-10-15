
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
train_data = pd.read_csv('data/filtered/selected_data.csv',index_col=0).iloc[:,:-1]
train_labels = pd.read_csv('data/filtered/selected_data.csv',index_col=0).iloc[:,-1]
test_data = pd.read_csv('data/filtered/selected_test.csv', index_col=0)

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)

#train_data['label'] = train_labels


# Set input
X_train = train_data   # Input train data
y_train = train_labels    # Input train labels
X_validate = test_data  # Input validate data


#Logistic BEST!!!
predict = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=2000).fit(X_train, y_train).predict(X_validate)
predict1 = LogisticRegression(random_state=1, solver='lbfgs', multi_class='ovr', max_iter=2000).fit(X_train, y_train).predict(X_validate)



#Write to CSV
with open("data/output.csv", "w", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["Sample_id","Sample_label"])
    for i in range(0,X_validate.shape[0]):
        wr.writerow([i+1,str(predict[i].item())])
