import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    Dense, Reshape, Flatten, LeakyReLU,ReLU,
    LayerNormalization, Dropout, BatchNormalization
    )

def build_generator(latent_space, n_var, n_features,use_bias,activation1 = 'relu',activation2='tanh'):

    model = tf.keras.Sequential()
    model.add(Dense(n_var*15, input_shape=(latent_space,), activation=activation1, use_bias=use_bias))#n_var*3
    model.add(BatchNormalization())
    model.add(Dense(n_var*5, activation=activation1))  #n_var*4                # 10
    model.add(BatchNormalization())
    model.add(Dense((n_var*5), activation=activation1))  #n_var*4        # 25
    model.add(BatchNormalization())
    model.add(Dense(n_features, activation=activation2, use_bias=use_bias))

    return model

def build_critic(n_var, use_bias, n_features):
    n_var = n_var*2
    model = tf.keras.Sequential()
    model.add(Dense(n_var*5, input_shape=(n_features,), use_bias=use_bias))#n_var*3
    #model.add(LayerNormalization())
    model.add(ReLU())#LeakyReLu
    #model.add(Dropout(0.2))
    model.add(Dense(n_var*15))#n_var*5
    model.add(ReLU())
    #model.add(Dropout(0.2))
    model.add(Dense(n_var*5))#n_var*5
    model.add(ReLU())
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1))

    return model

