'''
Multi-class classification using deep learning in keras.
'''

# %%
# Dependencies
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.datasets import reuters
# %%
(trainData, trainLabels), \
(testData, testLabels) = reuters.load_data(num_words=10000)
# %%
# define a function to vectorize the data
def vectorizeSequence(sequences, dimension=10000):
    '''
    convert data into 2d vectors with length dimension.
    '''

    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1

    return results

# perform the vectorization
xTrain = vectorizeSequence(trainData)
xTest = vectorizeSequence(testData)

# %%
# import onehot encoder
from keras.utils.np_utils import to_categorical
oneHotTrainLabels = to_categorical(train_labels)
oneHotTestLabels = to_categorical(test_labels)