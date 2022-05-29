'''
Predicting house prices using a dense NN.
'''

# %%
# dependencies
from threading import activeCount
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import models, layers
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# %%
# load dataset into train and test sets
(trainData, trainTargets), (testData, testTargets) = boston_housing.load_data()

# %%
# standard scale the data
scaler = StandardScaler()
trainData = scaler.fit_transform(trainData)
testData = scaler.transform(testData)

# %%
def buildModel():
    '''function to build model so it may be repeated easily.'''

    # define model structure
    model = models.Sequential()
    model.add(layers.Dense(64, 
                           activation='relu', 
                           input_shape=(trainData.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    
    # compile model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

# %%
# instantiate a k fold cross validator
kf = KFold(n_splits=3)
kf.get_n_splits(trainData)

# %%
# split the data and loop n_splits times
splitHistories = {}
i=0
fig,ax = plt.subplots(dpi=100)
for trainIndex, valIndex in kf.split(trainData):
    xTrain, xVal = trainData[trainIndex], trainData[valIndex]
    yTrain, yVal = trainTargets[trainIndex], trainTargets[valIndex]
    # instantiate model
    model = buildModel()
    # fit model
    splitHistories[i] = model.fit(xTrain,yTrain, 
                                  validation_data=(xVal,yVal),
                                  epochs=100, 
                                  batch_size=8, 
                                  verbose=0)
    valMae = pd.Series(splitHistories[i].history['val_mae'])
    trainMae = pd.Series(splitHistories[i].history['mae'])
    ax.plot(valMae.rolling(5).mean()[15:].index,
            valMae.rolling(5).mean()[15:], 
            label=i, c='C'+str(i))
    ax.plot(trainMae.rolling(5).mean()[15:].index,
            trainMae.rolling(5).mean()[15:], 
            label='train '+str(i), c='C'+str(i), ls='--')
    i=i+1
ax.set_xlabel('epoch'); ax.set_ylabel('5 Epoch Rolling Avg Val Set MAE');
ax.legend();
plt.show()

# %%
