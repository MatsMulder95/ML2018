import pandas as pd

testLabels = pd.read_csv('data/filtered/test_labels.csv', header=None)[0]


def getResult(predict):
    result = 0
    for i in range(0, testLabels.shape[0]):
        if predict[i] == testLabels[i]:
            result = result + 1
    return result / testLabels.shape[0]
