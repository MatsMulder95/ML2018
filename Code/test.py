import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import progressbar
import time
import random

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
measurements = ['mean', 'stddev','min','max']
for i in range(1,25):
    for s in range(7):
        dataScaled = dataScaled.rename(columns = {(i-1)+s*24:str(i)+'_Band_'+ statistic[s]})
        data = data.rename(columns = {(i-1)+s*24:str(i)+'_Band_'+ statistic[s]})
for i in range(1,13):
    for s in range(4):
        dataScaled = dataScaled.rename(columns = {168 + (i-1)+s*12:str(i)+'_Class_'+measurements[s]})
        data = data.rename(columns = {168 + (i-1)+s*12:str(i)+'_Class_'+measurements[s]})
for i in range(1,13):
    for s in range(4):
        dataScaled = dataScaled.rename(columns = {216 + (i-1)+s*12:str(i)+'_Coefficients_'+measurements[s]})
        data = data.rename(columns = {216 + (i-1)+s*12:str(i)+'_Coefficients_'+measurements[s]})        

##adding label data
data['label'] = labels
dataScaled['label'] = labels

##example use of filter based on new column naming
dataMeans = dataScaled.filter(regex='mean')
data.to_csv('data/filtered/ordered_data.csv')
genreLength = []
indexSelect = []
for i in range(1,11):
    genreLength.append(labels.values[:,0].tolist().count(i))
lenMin = np.min(genreLength)
for genre in range(1,11):
    genreIndex = data.loc[data.label==genre].index.tolist()
    genreSelect = random.sample(genreIndex,lenMin)
    indexSelect.extend(genreSelect)
filteredData = data.loc[indexSelect]
    
