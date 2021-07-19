import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import (Dense, Reshape, Flatten, LeakyReLU,ReLU,LayerNormalization, Dropout, BatchNormalization)
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise ,Conv1DTranspose
from keras.models import Sequential, Model
from keras.layers.convolutional import UpSampling2D,UpSampling1D, Conv2D,Conv1D,Conv2DTranspose

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

import matplotlib.pyplot as plt

import sys

import numpy as np


def build_generator(latent_space, n_features):
    
    model = tf.keras.Sequential()
    
    a=n_features*2
    b=n_features*2
    
    model.add(Dense(a*b, use_bias=False, input_shape=(latent_space,)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Reshape((a, b)))
    
    model.add(Conv1DTranspose(n_features*6, kernel_size=1, strides=1, padding='same', use_bias=True))#*4
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1DTranspose(n_features*3, kernel_size=1, strides=1, padding='same', use_bias=True))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Flatten()) 
    model.add(Dense(n_features, use_bias=False, activation='tanh'))
    
    return model

def build_discriminator(n_features,latent_space,BATCH_SIZE):

    model = tf.keras.Sequential()
    
    a=n_features*2
    b=n_features*2

    model.add(Dense(a*b, use_bias=False, input_shape=([n_features])))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((a, b)))
    
    model.add(Conv1D(n_features*6, kernel_size=1, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(n_features*3, kernel_size=1, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(1))

    return model

              


