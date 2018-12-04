# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:37:15 2018

@author: APU
"""

# Plot ad hoc mnist instances
import numpy as np
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers import Flatten

from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D



#mnist = tf.keras.datasets.mnist

# reproductibility
seed_val = 5
np.random.seed(seed_val)

#load dataset (60k as train, 10k as test)
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

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]

print (X_train.shape[1]," ",X_train.shape[2]," ",num_pixels)
print (X_train.shape)
print (X_test.shape)

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

print (X_train.shape)
print (X_test.shape)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs , explore more [transforming the vector of class integers into a binary matrix]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define baseline model , one hidden layer, same number of neurons(784),
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    #softmax activation, on the output layer, turn the outputs into probability value
    #select one class prediction from 0-9
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax')) 
	# Compile model
   #Logarithmic loss, ADAM gradient descent algorithm to learn the weights
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
# verbose 2, reduce the output to one line for each training epoch (5 times), updates per 500 images
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=500, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print (scores)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))




"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

sample_image = mnist.train.next_batch(1)[0]
print(sample_image.shape)

sample_image = sample_image.reshape([28, 28])
plt.imshow(sample_image, cmap='Greys')
"""





