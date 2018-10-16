import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import progressbar
import time
import random

##importing data

data = pd.read_csv('data/train_data.csv',header= None)
test = pd.read_csv('data/test_data.csv',header=None)
labels = pd.read_csv('data/train_labels.csv',header= None)
data.iloc[np.where(data.values==1000000)]=1E-6

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
        test = test.rename(columns = {(i-1)+s*24:str(i)+'_Band_'+ statistic[s]})
for i in range(1,13):
    for s in range(4):
        dataScaled = dataScaled.rename(columns = {168 + (i-1)+s*12:str(i)+'_Class_'+measurements[s]})
        data = data.rename(columns = {168 + (i-1)+s*12:str(i)+'_Class_'+measurements[s]})
        test = test.rename(columns = {168 + (i-1)+s*12:str(i)+'_Class_'+measurements[s]})
for i in range(1,13):
    for s in range(4):
        dataScaled = dataScaled.rename(columns = {216 + (i-1)+s*12:str(i)+'_Coefficients_'+measurements[s]})
        data = data.rename(columns = {216 + (i-1)+s*12:str(i)+'_Coefficients_'+measurements[s]})
        test = test.rename(columns = {216 + (i-1)+s*12:str(i)+'_Coefficients_'+measurements[s]})        

##adding label data
data['label'] = labels
dataScaled['label'] = labels

##example use of filter based on new column naming
dataMeans = dataScaled.filter(regex='mean')
data.to_csv('data/filtered/ordered_data.csv')
test.to_csv('data/filtered/ordered_test_data.csv')
columnList = []
columnList.extend(data.filter(regex='mean').columns.tolist())
columnList.extend(data.filter(regex='variance').columns.tolist())
columnList.extend(data.filter(regex='kurtosis').columns.tolist())
columnList.extend(data.filter(regex='skewness').columns.tolist())
columnList.extend(data.filter(regex='stddev').columns.tolist())
testFeaturesSelection = test[columnList]
testFeaturesSelection.to_csv('data/filtered/selected_test.csv')
columnList.extend(['label'])
featureSelection = data[columnList]
featureSelection.to_csv('data/filtered/selected_data.csv')

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
filteredData.to_csv('data/filtered/filtered_data.csv')
##means = []
##variances = []
##for label in range(1,11):
##    genre = data.loc[data.label==label]
##    for band in range(1,25):
##        averageMean = genre[str(band)+'_Band_mean'].mean()
##        averageVariance = np.sqrt((np.sum(genre[str(band)+'_Band_variance']**2)+np.sum((genre[str(band)+'_Band_mean']-averageMean)**2))/data.shape[0])
##        means.append(averageMean)
##        variances.append(averageVariance)
####    for cls in range(1,13):
####        averageMean = genre[str(cls)+'_Class_mean'].mean()
####        averageVariance = np.sqrt((np.sum(genre[str(cls)+'_Class_stddev']**4)+np.sum((genre[str(cls)+'_Band_mean']-averageMean)**2))/data.shape[0])
####        means.append(averageMean)
####        variances.append(averageVariance)
####    for coef in range(1,13):
####        averageMean = genre[str(coef)+'_Coefficients_mean'].mean()
####        averageVariance = np.sqrt((np.sum(genre[str(coef)+'_Coefficients_stddev']**4)+np.sum((genre[str(coef)+'_Band_mean']-averageMean)**2))/data.shape[0])
####        means.append(averageMean)
####        variances.append(averageVariance)
##    plt.errorbar(range(1,25),means[24*(label-1):24*label],variances[24*(label-1):24*label],label = 'genre'+ str(label))
##plt.title('Rhythem genre average data')
##plt.legend()
##plt.show()
##
##means = []
##stddev = []
##for label in range(1,11):
##    genre = data.loc[data.label==label]
##    for Class in range(1,13):
##        averageMean = genre[str(Class)+'_Class_mean'].mean()
##        averageStddev = np.sqrt((np.sum(genre[str(Class)+'_Class_stddev']**2)+np.sum((genre[str(Class)+'_Class_mean']-averageMean)**2))/data.shape[0])
##        means.append(averageMean)
##        stddev.append(averageStddev)
##    plt.errorbar(range(1,13),means[12*(label-1):12*label],stddev[12*(label-1):12*label],label = 'genre'+ str(label))
##plt.title('Pitch genre average data')
##plt.legend()
##plt.show()

##means = []
##stddev = []
##for label in range(1,11):
##    genre = data.loc[data.label==label]
##    for Coefficients in range(1,13):
##        averageMean = genre[str(Coefficients)+'_Coefficients_mean'].mean()
##        if averageMean==1000000:
##            averageMean=1E-6
##        averageStddev = np.sqrt((np.sum(genre[str(Coefficients)+'_Coefficients_stddev']**2)+np.sum((genre[str(Coefficients)+'_Coefficients_mean']-averageMean)**2))/data.shape[0])
##        means.append(averageMean)
##        stddev.append(averageStddev)
##    plt.errorbar(range(1,13),means[12*(label-1):12*label],stddev[12*(label-1):12*label],label = 'genre'+ str(label))
##plt.title('MFCCs genre average data')
##plt.legend()
##plt.show()

####creating a plot of means for each band *option for one color per genre and 1 vs all possible
##for label in range(1,11):#this is for iterating through all labels
##    for index in range(len(dataMeans.index)):
####    marker = labelStyles[dataScaled.loc[index,'label']-1]
####    color = colorStyles[dataScaled.loc[index,'label']-1]
##        if dataScaled.loc[index,'label'] ==label:
##            color = 'k'
##        else:
##            color = 'r'
##        plt.scatter(range(1,25),dataMeans.loc[index], c = color, alpha = 0.005)
##
##    plt.show()

    


    
