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
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import *
import functools
import time
import gc
from keras.initializers import RandomNormal


from network import build_discriminator, build_encoder, build_decoder


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
    def __init__(self, Z, n_features, batch_size, GANorWGAN, nodes):
        
        ###################### Wasserstein Loss #######################
        
        def wasserstein_loss(y_true, y_pred):
            return backend.mean(y_true * y_pred)
        
        ###################### Parameters #######################

        self.Z = Z
        self.n_features = n_features
        self.nodes = nodes

        self.constraint = 10#0.01
        self.GANorWGAN = GANorWGAN
        self.batch_size=batch_size

        self.c1_hist = []
        self.c2_hist = []
        self.g1_hist = []
        self.g2_hist = []
        
        self.lr_start = 0.001 
        self.lr_mid = 0.0001 
        self.lr_end = 0.0001
  
        
#DEFINE_float("keep_prob", 0.9, "dropout rate")
#DEFINE_float("lr_start", 0.001, "initial learning rate")
#DEFINE_float("lr_mid", 0.0001, "mid learning rate")
#DEFINE_float("lr_end", 0.0001, "final learning rate")

        
        # for prediction function
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.optimizers.Adam(self.lr_start)
        
        
        if GANorWGAN == 'WGAN':
            self.loss = wasserstein_loss
        elif GANorWGAN == 'GAN':
            self.loss = 'binary_crossentropy'


        # Build the discriminator
        self.discriminator = build_discriminator(self.Z, self.nodes)
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer)
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = True
        

        # Build the encoder
        self.encoder = build_encoder(self.Z, self.nodes, self.n_features)
   
        # Build the decoder
        self.decoder = build_decoder(self.Z, self.nodes, self.n_features)

        
        # Stacking together: The adversarial_autoencoder model
        #x_shape=(self.n_features,)
        real_input = Input(shape=(self.n_features,))
        encoder_output = self.encoder(real_input)
        decoder_output = self.decoder(encoder_output)
        discriminator_output = self.discriminator(encoder_output)

        self.aae = Model(real_input, [decoder_output, discriminator_output], name = 'AAE')
        self.aae.compile(loss=[self.mse, self.loss], loss_weights=[0.999, 0.01], optimizer=self.optimizer)#100
        
        ###################### Preprocessing #######################
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
        train_dataset = train_dataset.shuffle(len(X_train_scaled)).batch(self.batch_size)
            
        num=1
            
        for data in train_dataset:
            print("data shape_"+str(num),data.shape)
            num+=1
                
        print("Cycles: ",num-1)
        
        
        return train_dataset, scaler, X_train_scaled 


    
    ####################Training######################

    def train(self, Z, batch_size, dataset, epochs, scaler, X_train_scaled, scaled, X_train, y_train ):

        if self.GANorWGAN == 'WGAN':
            real = -np.ones(self.batch_size)
            fake = np.ones(self.batch_size)

        if self.GANorWGAN == 'GAN':
            real = np.ones(self.batch_size)
            fake = np.zeros(self.batch_size)

        # Training the model
        for epoch in range(epochs):
            c1_tmp, c2_tmp, g1_tmp, g2_tmp = list(), list(), list(), list()
            
            if epoch <= 75: #50
                self.lr_start = self.lr_start
            elif epoch <=100: #100
                self.lr_start = self.lr_mid
            else:
                self.lr_start = self.lr_end
            
            for batch in dataset:
                # Randomly selected noise
                #X= np.concatenate((X_train, y_train), axis=1)
                #noise = self.gaussian_mixture.reshape([self.batch_size, self.Z])
                noise = tf.random.normal([self.batch_size, self.Z])

                # Generate a batch of new outputs (in the latent space) predicted by the encoder
                gen_data = self.encoder.predict(batch)

                # Train the discriminator
                # The arbitrary noise is considered to be a "real" sample
                d_loss_real = self.discriminator.train_on_batch(noise, real)
                c1_tmp.append(d_loss_real)
                # The latent space generated by the encoder is considered a "fake" sample
                d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
                c2_tmp.append(d_loss_fake)

                self.c1_hist.append(np.mean(c1_tmp))
                self.c2_hist.append(np.mean(c2_tmp))


                # Training the stacked model
                g_loss = self.aae.train_on_batch(batch, [batch, real])
                g1_tmp.append(g_loss[0])
                g2_tmp.append(g_loss[1])
                
                self.g1_hist.append(np.mean(g1_tmp))
                self.g2_hist.append(np.mean(g2_tmp))
                


            print("%d [D real: %f, D fake: %f], [Enc/Dec loss: %f, Enc/Dis: %f]" % (epoch+1, self.c1_hist[epoch], self.c2_hist[epoch], self.g1_hist[epoch],self.g2_hist[epoch]))

            # Checkpoint progress: Plot losses and predicted data
            #if epoch == epochs:
                #self.plot_loss(epoch)
                #self.plot_values(X_train_scaled, X_train, y_train)
                
        # saving the events of the generator and critic
        decoder_log_dir = './content/decoder'
        encoder_log_dir = './content/encoder'
        discriminator_log_dir = './content/discriminator'
        self.decoder_summary_writer = tf.summary.create_file_writer(decoder_log_dir)
        self.encoder_summary_writer = tf.summary.create_file_writer(encoder_log_dir)
                
        self.encoder.save('./content/encoder_'+str(epoch+1) )
        self.decoder.save('./content/decoder_'+str(epoch+1))
        self.discriminator.save('./content/discriminator_'+str(epoch+1) )
        print('save model')
                
   # Plots the (W)GAN related losses at every sample interval

    def plot_loss(self, epoch):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.plot(self.c1_hist, c='red')
        plt.plot(self.c2_hist, c='blue')
        plt.plot(self.g_hist[0][0], c='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("GAN Loss per Epoch")
        plt.legend(['C real', 'C fake', 'Generator'])

        plt.subplot(1,2,2)
        plt.plot(self.g_hist[0][1], c='green')
        plt.xlabel('Epoch')
        plt.ylabel('Mean squared error')
        plt.title('MSE loss')
        plt.savefig('./result/AAE/loss'+'.png')
        plt.close()
        
        
            
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

