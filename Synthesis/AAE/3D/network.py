import keras ##old
import tensorflow as tf
import tensorflow.keras.backend as backend
from keras import Input, Model
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, ReLU, Dropout, BatchNormalization, Activation
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise ,Conv1DTranspose
from keras.layers.convolutional import UpSampling2D,UpSampling1D, Conv2D,Conv1D,Conv2DTranspose


from keras.layers import merge, Lambda
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as backend
from keras.constraints import Constraint
from keras.initializers import RandomNormal

from sklearn.datasets import make_swiss_roll

import matplotlib.pyplot as plt

import numpy as np

class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}
    
    
    
def build_encoder(Z, nodes, n_features):
    
    x_shape=(n_features,)

    input_enc = Input(shape=x_shape)
    a = nodes*2
    b = nodes*2
    
    enc = Dense(a*b, use_bias=False)(input_enc)#16

    enc = BatchNormalization()(enc)#(4,4)
    enc = ReLU()(enc)
    enc = Reshape((a, b))(enc)
    enc = BatchNormalization()(enc)
    
    enc = Conv1DTranspose(nodes*4, kernel_size=1, strides=1, padding='same', use_bias=False)(enc)#(4,8) #
    enc = BatchNormalization()(enc)#nodes*4 #
    enc = ReLU()(enc) #
    
    enc = Conv1DTranspose(nodes, kernel_size=1, strides=1, padding='same', use_bias=False)(enc)#(4,2)
    enc = BatchNormalization()(enc)
    enc = ReLU()(enc)
    
    #enc = Conv1DTranspose(nodes*2, kernel_size=1, strides=1, padding='same', use_bias=True)(enc)#(4,2) #
    #enc = BatchNormalization()(enc)#
    #enc = ReLU()(enc)#
    
    enc = Flatten()(enc)
    
    mu = Dense(Z, use_bias=False, activation='tanh')(enc) #Z
    sigma = Dense(Z, use_bias=False, activation='tanh')(enc)

    # The latent representation ("fake") in a Gaussian distribution is then compared to the "real" arbitrary Gaussian
    # distribution fed in the Discriminator
    latent_repr = Lambda(lambda p: p[0] + backend.random_normal(backend.shape(p[0])))([mu, sigma])
    latent_repr = BatchNormalization()(latent_repr)
    #latent_repr = ReLU()(latent_repr)
    
    #latent_repr = Lambda(lambda p: p[0] + backend.random_normal(backend.shape(p[0])) * backend.exp(p[1]/2))([mu, sigma])
    generator_encoder = Model(input_enc, latent_repr, name='Encoder')#nd.exp(p[1] / 2))
    generator_encoder.summary()
    
    return generator_encoder



###decoder, is different with encoder, performes better
def build_decoder(Z, nodes, n_features):

    # Input to the decoder is the latent space from the encoder
    input_dec = Input(shape=(Z,))#Z
    
    dec = Dense(nodes*15,activation='relu', use_bias=True)(input_dec)
    dec = BatchNormalization()(dec)
    
    dec = Dense(nodes*5,activation='relu')(dec)
    dec = BatchNormalization()(dec)
    
    dec = Dense(nodes*5,activation='relu')(dec)
    dec = BatchNormalization()(dec)
    
    output_dec = Dense(n_features, activation='tanh', use_bias=True)(dec)
    generator_decoder = Model(input_dec, output_dec, name='Decoder')

    generator_decoder.summary()
    return generator_decoder


def build_discriminator(Z, nodes):
    
    
    a=nodes*2
    b=nodes*2
    
    in_disc = Input(shape=(Z))
    
    disc = Dense(a*b, use_bias=False)(in_disc)
    disc = BatchNormalization()(disc)
    disc = ReLU()(disc)
    disc = Reshape((a, b))(disc)
    
    disc = Conv1D(nodes*4, kernel_size=1, strides=1, padding='same', use_bias=False)(disc)
    disc = BatchNormalization()(disc)
    disc = ReLU()(disc)
    
    disc = Conv1D(nodes*2, kernel_size=1, strides=1, padding='same', use_bias=False)(disc)
    disc = BatchNormalization()(disc)
    disc = ReLU()(disc)

    disc = Flatten()(disc)
    disc_output = Dense(1, activation='sigmoid', use_bias=False)(disc)


    discriminator = Model(in_disc, disc_output, name='Discriminator')

    return discriminator
   






