import keras ##old
import tensorflow as tf
import tensorflow.keras.backend as backend
from keras import Input, Model
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, LeakyReLU, ReLU, Dropout, BatchNormalization,LayerNormalization, Activation
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
    
    
    
def build_encoder(Z, nodes, n_features, use_bias):
    
    x_shape=(n_features,)

    input_enc = Input(shape=x_shape)

    
    enc = Dense(1000, use_bias=use_bias)(input_enc)
    enc = Dropout(0.2)(enc)
    enc = LayerNormalization()(enc)
    enc = ReLU()(enc)
    
    enc = Dense(1000, use_bias=use_bias)(enc)
    enc = LayerNormalization()(enc)
    enc = ReLU()(enc)
    
    mu = Dense(Z, use_bias=use_bias, activation='tanh')(enc) #Z
    mu = LayerNormalization()(mu)
    
    sigma = Dense(Z, use_bias=use_bias, activation='tanh')(enc)
    sigma = LayerNormalization()(sigma)

    # The latent representation ("fake") in a Gaussian distribution is then compared to the "real" arbitrary Gaussian
    # distribution fed in the Discriminator
    latent_repr = Lambda(lambda p: p[0] + backend.random_normal(backend.shape(p[0] * backend.exp(p[1]/2))))([mu, sigma])




    generator_encoder = Model(input_enc, latent_repr, name='Encoder')
    generator_encoder.summary()
    
    return generator_encoder



###decoder, is different with encoder, performes better
def build_decoder(Z, var, n_features, use_bias):

    # Input to the decoder is the latent space from the encoder
    input_dec = Input(shape=(Z,))#Z
    
    dec = Dense(1000, use_bias=use_bias)(input_dec)
    dec = ReLU()(dec)
    
    dec = Dense(1000, use_bias=use_bias)(dec)
    dec = ReLU()(dec)
    
    dec = Dense(1000, use_bias=use_bias)(dec)
    
    output_dec = Dense(n_features, activation='tanh', use_bias=use_bias)(dec)
    generator_decoder = Model(input_dec, output_dec, name='Decoder')

    generator_decoder.summary()
    return generator_decoder


def build_discriminator(Z, nodes):
    
    
    in_disc = Input(shape=(Z))
    
    disc = Dense(1000, use_bias=False)(in_disc)
    disc = ReLU()(disc)
    
    disc = Dense(1000, use_bias=False)(disc)
    disc = ReLU()(disc)

    disc_output = Dense(1, activation='sigmoid', use_bias=False)(disc)

    discriminator = Model(in_disc, disc_output, name='Discriminator')

    return discriminator
   





