import keras
import tensorflow as tf
import tensorflow.keras.backend as backend
from keras import Input, Model
from keras.models import Sequential, Model
from keras.layers import ( Dense, Reshape, Flatten, LeakyReLU, ReLU, ELU , LayerNormalization,Softmax, Dropout, BatchNormalization)
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation
from keras.layers import merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as backend

import matplotlib.pyplot as plt

import numpy as np


def build_encoder(Z, nodes, n_features):
    
    x_shape=(n_features,)
    x = Input(shape=x_shape)#input=n_features=2 or 3

    h = Dense(nodes,input_shape=(n_features,))(x) #nodes=4
    h = (BatchNormalization())(h)
    h = ReLU()(h)

    
   
    nodes=nodes*2 #nodes*4=8
    h = Dense(nodes)(h)
    h = (BatchNormalization())(h)
    h = ReLU()(h)
    
    nodes=nodes/2 #nodes/2=4
    h = Dense(nodes)(h)
    h = (BatchNormalization())(h)
    h = ReLU()(h)
    
    mu = Dense(Z)(h)#Z=2
    sigma = Dense(Z)(h)#Z=2
    
    latent_repr = Lambda(lambda p: p[0] + backend.random_normal(backend.shape(p[0])) * backend.exp(p[1]/2))([mu, sigma])
    #lamda=Z=2/2*2
    
    return Model(x , latent_repr)



def build_decoder(Z,nodes, n_features):
    
    decoder = Sequential()
    decoder.add(Dense(Z,input_shape=(Z,)))#Z=2
    
    n = nodes #nodes=4
    decoder.add(Dense(n))
    decoder.add(BatchNormalization())
    decoder.add(ReLU())
    
    n = nodes*2 #nodes=8
    decoder.add(Dense(n))
    decoder.add(BatchNormalization())
    decoder.add(ReLU())
    
    n = nodes #nodes=4
    decoder.add(Dense(n))
    decoder.add(BatchNormalization())
    decoder.add(ReLU())
    
    decoder.add(Dense(n_features, activation='tanh'))#output=n_features=2=input
    
    return decoder


def build_discriminator(Z):
    discriminator = Sequential()
    discriminator.add(Input(shape=(Z,)))
    discriminator.add(ReLU())


    discriminator.add(Flatten())
    discriminator.add(Dense(1))#, activation="sigmoid"))
    
    return discriminator
   




