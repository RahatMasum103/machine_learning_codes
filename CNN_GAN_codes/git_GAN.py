# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:10:48 2018

@author: APU
"""
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from keras.utils import np_utils
from contextlib import redirect_stdout
from keras import backend as K

K.set_image_dim_ordering('th')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#import sys

import numpy as np

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
pred = np.argmax(pred,axis=1)

print (np.sum(np.argmax(y_test, axis=1)==pred)/np.size(pred))



with open('E:/TTU/ML Projects/GAN/CNN_model_summary.txt', 'w+') as f:
    with redirect_stdout(f):
        cnn_model_2.summary()
#print (np.argmax(pred,axis=1))

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()
        
        with open('E:/TTU/ML Projects/GAN/Generator_model_summary.txt', 'w+') as f:
            with redirect_stdout(f):
                model.summary()
       

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        with open('E:/TTU/ML Projects/GAN/Discriminator_model_summary.txt', 'w+') as f:
            with redirect_stdout(f):
                model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):           
            #  Train Discriminator       

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
           
            #  Train Generator
       
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        plt.imshow(gen_imgs[0, :,:,0], cmap='gray') 
        # print (gen_imgs[0,0:28,0:28,0])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        find_label = gen_imgs.reshape(r*c, 1, 28, 28).astype('float32')
        
        #find_label = find_label/255
        pred = cnn_model_2.predict(find_label)
        
        # print (np.sum(np.argmax(y_test, axis=1)==pred)/np.size(pred))
        print (np.argmax(pred,axis=1))
        #print (pred)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                #plt.imshow(gen_imgs[cnt, :,:,0], cmap='gray')                
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("E:/TTU/ML Projects/GAN/gen_image/%d.png" % (epoch))
        #plt.savefig("E:/TTU/ML Projects/GAN/gen_image/%d.png" % (epoch+1))
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=1000, batch_size=32, sample_interval=200)