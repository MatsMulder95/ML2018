import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time

##importing data

data = pd.read_csv('data/train_data.csv',header= None)
labels = pd.read_csv('data/train_labels.csv',header= None)

##normalizing data
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data)
dataScaled = pd.DataFrame(np_scaled)

##setting up list of markers and colors for plotting
labelStyles = ['.','o','v','^','s','*','+','d','<','>']
colorStyles = ['k','r','g','b','y','c','m','purple','pink','orange']

##renaming columns to format %band_number%_%name_of_statistic%
statistic = ['mean','median','variance','kurtosis','skewness','min','max']
for i in range(1,25):
    for s in range(7):
        dataScaled = dataScaled.rename(columns = {(i-1)+s*24:str(i)+'_'+ statistic[s]})
        data = data.rename(columns = {(i-1)+s*24:str(i)+'_'+ statistic[s]})

##adding label data
data['label'] = labels
dataScaled['label'] = labels

##example use of filter based on new column naming
dataMeans = dataScaled.filter(regex='mean')

##creating a plot of means for each band *option for one color per genre and 1 vs all possible
for label in range(1,11):#this is for iterating through all labels
    for index in range(len(dataMeans.index)):
##    marker = labelStyles[dataScaled.loc[index,'label']-1]
##    color = colorStyles[dataScaled.loc[index,'label']-1]
        if dataScaled.loc[index,'label'] ==label:
            color = 'k'
        else:
            color = 'r'
        plt.scatter(range(1,25),dataMeans.loc[index], c = color, alpha = 0.005)

    plt.show()

print("ready")

    


