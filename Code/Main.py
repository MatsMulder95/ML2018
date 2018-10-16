from csvWrite import *
from ClassifyData import *

# Import en Filter data
trainData = pd.read_csv('data/filtered/selected_data.csv',index_col=0).iloc[:,:-1]
trainLabels = pd.read_csv('data/filtered/selected_data.csv',index_col=0).iloc[:,-1]
testData = pd.read_csv('data/filtered/selected_test.csv', index_col=0)

# Train data
predict = classify(trainData, trainLabels, testData, 'logistic', 'local')

# Visualization

# Write to csv
writetocsv(predict)
