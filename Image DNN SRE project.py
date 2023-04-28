#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

mnist = keras.datasets.mnist # replace mnist with fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data() # replace mnist with fashion_mnist

# Splitting data set
X_valid, X_train = X_train_full[ 0:1000] / 255, X_train_full[1000: ]/255
X_test = X_test/255
y_valid, y_train = y_train_full[ 0:1000], y_train_full[1000: ]
for i in range(0, 1000):
    print(y_valid[i])
# printing the data at index 0
for r in range(0,1000):
    X_valid[r]
    pixelvalues=[]
    for i in range (0,28):
        for f in range(0,28):
            pixelvalues.append(X_valid[r][i][f])
            ++f
        ++i

    for i in range(0, 28*28):
        if pixelvalues[i] <0.5:
            pixelvalues[i]= 0
        elif pixelvalues[i] > 0.5:
            pixelvalues[i] =1
        ++i

    pixelnumber = list(range(1,28*28+1))

    df = pd.DataFrame(list(zip(pixelnumber, pixelvalues)),
                   columns =['PixelNumber', 'grayvalues'])
 
    x_train = df.iloc[:785,0:1]
    y_train = df.iloc[:785,1:2]

    from sklearn.neural_network import MLPClassifier
    modelClass1 = MLPClassifier(max_iter=1000, hidden_layer_sizes=(64, 64, 64, 64), activation='relu')
    modelClass1.fit(x_train,y_train)
    
    
    ++r
    
    weights = modelClass1.coefs_[1:-1]
    hidLayerOneTwo = list(weights[0])
    hidLayerTwoThree = weights[1]
    hidLayerThreeFour = weights[2]
    

    
    array1 = weights[0][0]
    for i in range(1, len(weights[0])):
        array1 = np.vstack((array1, weights[0][i]))

    array2 = weights[1][0]
    for i in range(1, len(weights[1][i])):
        array2 = np.vstack((array2, weights[1][i]))

    array3 = weights[2][0]
    for i in range(1, len(weights[2][i])):
        array3 = np.vstack((array2, weights[2][i]))

    stacked_array = weights[2][0]
    for i in range(1, len(weights[2][i])):
        stacked_array = np.vstack((stacked_array, weights[2][i]))
    dir_path = 'C:/Users/mwarn/OneDrive/Desktop/weight_files/'
    file_name = f'weights_{r+1}'
    
    np.savetxt(os.path.join(dir_path, file_name), stacked_array)






# In[7]:





# In[ ]:




