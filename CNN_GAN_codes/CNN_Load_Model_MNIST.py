# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:33:32 2018

@author: APU
"""

# Plot ad hoc mnist instances
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

from contextlib import redirect_stdout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.models import model_from_json, load_model

#CNN Implementation
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


# define the larger model
def CNN_larger_model():
	# create model
	cnn_model = Sequential()
	cnn_model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
	cnn_model.add(Conv2D(15, (3, 3), activation='relu'))
	cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
	cnn_model.add(Dropout(0.2))
	cnn_model.add(Flatten())
	cnn_model.add(Dense(128, activation='relu'))
	cnn_model.add(Dense(50, activation='relu'))
	cnn_model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return cnn_model


# build the model
cnn_model_2 = CNN_larger_model()
# Fit the model
# verbose 2, reduce the output to one line for each training epoch (5 times), updates per 500 images
cnn_model_2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)
# Final evaluation of the model
cnn_scores_2 = cnn_model_2.evaluate(X_test, y_test, verbose=0)
print (cnn_scores_2)
print(cnn_model_2.metrics_names[1],": %.3f%%" % (cnn_scores_2[1]*100))

pred = cnn_model_2.predict(X_test)
print (pred)
with open('E:/TTU/ML Projects/GAN/pixel.txt', 'w+') as f:
    with redirect_stdout(f):
        pred


pred = np.argmax(pred,axis=1)



#print (pred )

print (np.sum(np.argmax(y_test, axis=1)==pred)/np.size(pred))
#print (np.argmax(pred,axis=1))


# save the model
cnn_model_2.save_weights('cnn_2_weights.h5')
with open('cnn_2_architecture.json','w') as jwf:
    jwf.write(cnn_model_2.to_json())
    
print("....CNN MODEL 2 saved into JSON.....")

# load the model
"""
with open('cnn_2_architecture.json','r') as jrf:
    loaded_model = model_from_json(jrf.read())
    
loaded_model.load_weights('cnn_2_weights.h5')

loaded_scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print (loaded_scores)
print("Loaded Model Accuracy: %.2f%%" % (loaded_model.metrics_names[1], loaded_scores[1]*100))
"""