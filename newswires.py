'''
Multi-class classification using deep learning in keras.
'''

# %%
# Dependencies
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt

# %%
(trainData, trainLabels), \
(testData, testLabels) = reuters.load_data(num_words=10000)
# %%
# define a function to vectorize the data
def vectorizeSequence(sequences, dimension=10000):
    '''
    convert data into 2d vectors with length dimension.
    '''

    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1

    return results

# perform the vectorization
xTrain = vectorizeSequence(trainData)
xTest = vectorizeSequence(testData)

# %%
# import onehot encoder
from keras.utils.np_utils import to_categorical
oneHotTrainLabels = to_categorical(trainLabels)
oneHotTestLabels = to_categorical(testLabels)

# %%
# build up a dense nn model
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000,)))
# predict 46 different classes - softmax outputs a probability distribution
model.add(keras.layers.Dense(46, activation='softmax'))

# %%
# compile model
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# create a validation set
valSize = 1000
xVal = xTrain[:valSize]
yVal = oneHotTrainLabels[:valSize]

# fit the model
history = model.fit(xTrain[valSize:], 
                    oneHotTrainLabels[valSize:],
                    epochs=20,
                    batch_size=32,
                    validation_data=[xVal,yVal])
# %%
epochs = range(1,len(history.history['loss'])+1)
plt.plot(epochs, history.history['loss'], label='trainLoss')
plt.plot(epochs, history.history['val_loss'], label='valLoss')
plt.xticks(epochs)
plt.legend()
plt.xlabel('Epoch'); plt.ylabel('Loss')

# %%
# build up a dense nn model
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000,)))
# predict 46 different classes - softmax outputs a probability distribution
model.add(keras.layers.Dense(46, activation='softmax'))

# compile model
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# fit the model
history = model.fit(xTrain, 
                    oneHotTrainLabels,
                    epochs=3,
                    batch_size=32)

# %%
# evaluate performance on test data
model.evaluate(xTest, oneHotTestLabels)

# %%
model.predict(xTest[:1]).shape
# %%
