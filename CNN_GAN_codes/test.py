# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 03:46:08 2018

@author: APU
"""

import json
import numpy as np
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Flatten

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.models import model_from_json, load_model
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()

# reshape to be [samples][pixels][width][height]
# gray scale, pixel dimension is 1.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs , explore more [transforming the vector of class integers into a binary matrix]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


with open('E:\TTU\ML Projects\GAN\cnn_2_architecture.json','r') as jrf:
    loaded_model = model_from_json(jrf.read())
    
loaded_model.load_weights('E:\TTU\ML Projects\GAN\cnn_2_weights.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print (loaded_scores)
print(loaded_model.metrics_names[1]," : %.2f%%" % (loaded_scores[1]*100))
print(loaded_model.metrics_names[0]," : %.2f%%" % (loaded_scores[0]*100))


"""
import h5py
hf = h5py.File('E:\TTU\ML Projects\GAN\cnn_2_weights.h5', 'r')
list(hf)

hf['flatten_1'][:]

hf.keys()
"""