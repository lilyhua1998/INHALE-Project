from __future__ import print_function, division
import tensorflow as tf
import tensorflow.keras.backend as backend
from tensorflow import reduce_mean

import keras
from keras import losses, backend
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.models import Sequential, Model
from keras.constraints import Constraint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
import functools
import time
import gc

from sklearn.datasets import make_swiss_roll

from network_ReLU import build_discriminator, build_encoder, build_decoder


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


class AAE():
    def __init__(self,Z, n_features,BATCH_SIZE,GANorWGAN,nodes,n_dis,n_decoder):
        
        # Wasserstein loss
        def wasserstein_loss(y_true, y_pred):
            return backend.mean(y_true * y_pred)
        

        self.n_features = n_features
        self.c1_hist = []
        self.c2_hist = []
        self.g_hist = []
        self.endis_hist = []
        self.autoencoder_hist = []
        self.decoder_hist = []
        self.D_loss = []
        self.BATCH_SIZE = BATCH_SIZE
        self.Z = Z
        self.constraint = 10
        #self.n_autoencoder= n_autoencoder
        self.n_decoder= n_decoder
        self.n_dis=n_dis
        #self.n_endis=n_endis
        self.nodes=nodes
        self.GANorWGAN = GANorWGAN
        
        self.lr_start = 0.001 
        self.lr_mid = 0.0001 
        self.lr_end = 0.0001
        
        # for prediction function
        self.mse = tf.keras.losses.MeanSquaredError()
        
        #self.optimizer = tf.keras.optimizers.RMSprop()
        self.optimizer1 = tf.keras.optimizers.Adam(1e-4)
        self.optimizer2 = tf.keras.optimizers.Adam(1e-4)#Nadam
        if GANorWGAN == 'WGAN':
            self.loss = wasserstein_loss
        elif GANorWGAN == 'GAN':
            self.loss = 'binary_crossentropy'



        # Build and compile the discriminator
        self.discriminator = build_discriminator(Z, nodes)
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer1)

        # Build the encoder / decoder
        self.encoder = build_encoder(Z, self.nodes, n_features)
        self.decoder = build_decoder(Z,nodes, n_features)
        

        X = Input(shape=(n_features,))
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(X)
        
        reconstructed_X = self.decoder(encoded_repr)
        
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = True

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        #encoder/discriminator
        self.endis=Model(inputs=X, outputs=validity)
        self.endis.compile(loss=self.loss, optimizer=self.optimizer1)
        
        #decoder/latent real
        self.decoder=build_decoder(Z,nodes, n_features)
        self.decoder.compile(loss='binary_crossentropy', optimizer=self.optimizer2)
        
        # saving the events of the generator and critic
        endis_log_dir = './content/endis'
        discriminator_log_dir = './content/discriminator'
        decoder_log_dir = './content/decoder'
        self.endis_summary_writer = tf.summary.create_file_writer(endis_log_dir)
        self.discriminator_summary_writer = tf.summary.create_file_writer(discriminator_log_dir)
        self.decoder_summary_writer = tf.summary.create_file_writer(decoder_log_dir)
        
        ######################
        
        # preprocessing
        
    def preproc(self, X_train, y_train, scaled):
        sample_data = np.concatenate((X_train, y_train), axis=1)
        
        if scaled == '-1-1':
            scaler = MinMaxScaler(feature_range=(-1,1))
            X_train_scaled = scaler.fit_transform(sample_data)
        elif scaled =='0-1':
            scaler = MinMaxScaler(feature_range=(0,1))
            X_train_scaled = scaler.fit_transform(sample_data)

        train_dataset = X_train_scaled.reshape(-1, self.n_features).astype('float32')
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.shuffle(len(X_train_scaled))
        train_dataset = train_dataset.batch(self.BATCH_SIZE)
            
        num=1
            
        for data in train_dataset:
            print("data shape_"+str(num),data.shape)
            num+=1
                
        print("Cycles: ",num-1)
        
        
        return train_dataset, scaler, X_train_scaled 
    
    ####################Training######################
    #@tf.function
    def train(self,Z,BATCH_SIZE, dataset, epochs, scaler, scaled,X_train_scaled):

        # Adversarial ground truths
        valid = -np.ones([self.BATCH_SIZE,1])
        fake = np.ones([self.BATCH_SIZE,1])

        for epoch in range(epochs):
            c1_tmp, c2_tmp,decoder_tmp = list(), list(), list()
            print("Epoch {}/{}".format(epoch+1, epochs))
            
            if epoch <= 75: #50
                self.lr_start = self.lr_start
            elif epoch <=100: #100
                self.lr_start = self.lr_mid
            else:
                self.lr_start = self.lr_end
            
            for batch in dataset:
                if batch.shape[0]==self.BATCH_SIZE:
                    batch=batch

                for _ in range(self.n_dis):
                    # Train the discriminator
                    latent_fake = self.encoder.predict(batch)
                    latent_real = np.random.normal(loc=0, scale=1, size=([self.BATCH_SIZE, self.Z]))
                    
                    d_loss_real = self.discriminator.train_on_batch(latent_real, valid) #output error
                    c1_tmp.append(d_loss_real)
                    d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake) #output error
                    c2_tmp.append(d_loss_fake)
                    
                    endis_loss=self.endis.train_on_batch(batch, valid)

                self.c1_hist.append(np.add(np.mean(c1_tmp),np.mean(c2_tmp))*0.5) #原本0.5
                self.endis_hist.append(endis_loss)
                
                #if self.endis_hist[epoch]<= 0.3:
                    #self.encoder.trainable = False

                ## train decoder(normal distribution) to X_train_scaled
                for _ in range(self.n_decoder):
                    latent_real = np.random.normal(loc=0, scale=1, size=([self.BATCH_SIZE, self.Z]))
                    dencoder_loss= self.decoder.train_on_batch(latent_real, batch)
                    decoder_tmp.append(dencoder_loss)
                self.decoder_hist.append(np.mean(decoder_tmp))
                
                                    
                

                        
            # Plot the progress
            print("[Dis(w): %f], [endis(w)]: %f, [decoder(mse)]: %f]" % (self.c1_hist[epoch], self.endis_hist[epoch], self.decoder_hist[epoch]))
            
            # Checkpoint progress: Plot losses and predicted data
            if epoch == epochs:
                #self.plot_loss(epochs)
                #self.plot_values(epochs,X_train_scaled)
                self.encoder.save('AAE/Models/encoder_' + str(self.Z) + '_' + str(epochs), save_format='tf')
                self.decoder.save('AAE/Models/decoder_'+ str(self.Z) + '_' + str(epochs), save_format='tf')
                self.discriminator.save('AAE/Models/discriminator_' + str(self.Z) + '_' + str(epochs), save_format='tf')
            
    def plot_loss(self, epochs):
        fig = plt.figure()
        plt.subplot()
        plt.ylim(-1,1)
        plt.plot(self.c1_hist)
        plt.plot(self.c2_hist)
        plt.plot(self.g_hist[0])#train generator, decoder output=valid~real

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("W Loss (decoder-discriminator)")
        plt.legend(['d_valid(-1)','d_fake(1)', 'g_loss(0)'])
        plt.savefig('AAE/Losses/14W Loss (decoder-discriminator)'+ str(epochs) + '_' + str(self.Z) + '.png')

        plt.subplot()
        plt.plot(self.g_hist)
        plt.xlabel('Epoch')
        plt.ylabel('Mean squared error')
        plt.title('MSE loss (encoder-decoder)')
        
        plt.savefig('AAE/Losses/MSE loss (encoder-decoder)_'+ str(epochs) + '_' + str(self.Z) + '.png')
        #plt.close()
            

        
            
    def mse_loss(self, inp, outp):
        """
        Calculates the MSE loss between the x-coordinates
        """
        inp = tf.reshape(inp, [-1, self.n_features])
        outp = tf.reshape(outp, [-1, self.n_features])
        
        print("input:",inp.shape)
        print("input:",outp.shape)
        return self.mse(inp[:,0], outp[:,0])
      

    def opt_step(self, latent_values, real_coding):
        """
        Minimizes the loss between generated point and inputted point
        """
        with tf.GradientTape() as tape:
            tape.watch(latent_values)
            gen_output = self.decoder(latent_values, training=False)
            loss = self.mse_loss(real_coding, gen_output)

        gradient = tape.gradient(loss, latent_values)
        self.optimizer.apply_gradients(zip([gradient], [latent_values]))

        return loss

            
            
    def optimize_coding(self, real_coding):
        """
        Optimizes the latent space values
        """
        latent_values = np.random.normal(0, 1, size=([len(real_coding), self.Z]))
        latent_values = tf.Variable(latent_values)


        loss = []

        for epoch in range(10000):
            loss.append(self.opt_step(latent_values, real_coding).numpy())

        return latent_values            
            
    def predict(self, input_data, scaler):

        predicted_vals = np.zeros((1, self.n_features))
        unscaled = scaler.inverse_transform(input_data)
        latent_values = self.optimize_coding(input_data)

        predicted_vals_1 = scaler.inverse_transform((self.decoder.predict(tf.convert_to_tensor(latent_values))).reshape(len(input_data), self.n_features))
        predicted_vals_1 = predicted_vals_1.reshape(len(input_data), self.n_features)


        predicted_vals = predicted_vals_1[1:,:]
        return predicted_vals






