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
    def __init__(self,i, Z, n_features, batch_size, GANorWGAN, nodes, var, use_bias):
        
        ###################### Wasserstein Loss #######################
        
        def wasserstein_loss(y_true, y_pred):
            return backend.mean(y_true * y_pred)
        
        ###################### Parameters #######################

        self.Z = Z
        self.n_features = n_features
        self.nodes = nodes
        self.var = var
        self.use_bias= use_bias
        self.i = i

        self.constraint = 0.01   #0.01
        self.GANorWGAN = GANorWGAN
        self.batch_size=batch_size

        self.c1_hist = []
        self.c2_hist = []
        self.g1_hist = []
        self.g2_hist = []
        
        self.lr_start = 1e-3
        self.lr_mid = 1e-4
        self.lr_end = 1e-4
  
        
        
        # for prediction function
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.optimizers.Adam(self.lr_start, beta_1=0.9)
        self.optimizer1 = tf.optimizers.Adam(self.lr_start, beta_1=0.9)
        
        
        
        if GANorWGAN == 'WGAN':
            self.loss = wasserstein_loss
        elif GANorWGAN == 'GAN':
            self.loss = 'binary_crossentropy'


        # Build the discriminator
        self.discriminator = build_discriminator(self.Z, self.nodes)
        self.discriminator.compile(loss=self.loss, optimizer=self.optimizer1)
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False
        

        # Build the encoder
        self.encoder = build_encoder(self.Z, self.nodes, self.n_features)
   
        # Build the decoder
        self.decoder = build_decoder(self.Z, self.var, self.n_features, self.use_bias)

        
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

    def train(self,i, Z, batch_size, dataset, epochs, scaler, X_train_scaled, scaled, X_train, y_train ):

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
            if (epoch+1) % 500 == 0:
                # saving the events of the generator and critic
                decoder_log_dir = './AAE/Models/decoder'
                encoder_log_dir = './AAE/Models/encoder'
                discriminator_log_dir = './AAE/Models/discriminator'
                self.decoder_summary_writer = tf.summary.create_file_writer(decoder_log_dir)
                self.encoder_summary_writer = tf.summary.create_file_writer(encoder_log_dir)
                
                self.encoder.save(  './AAE/Models/encoder_v'+str(i)+'_'+str(epoch+1)  )
                self.decoder.save(  './AAE/Models/decoder_v'+str(i)+'_'+str(epoch+1)  )
                self.discriminator.save(  './AAE/Models/discriminator_v'+str(i)+'_'+str(epoch+1) )
                print('save model')
                
                self.plot_loss(i, epoch)
                self.plot_latent(i, scaler, scaled, X_train, y_train, X_train_scaled, epoch)
                
                

                
   # Plots the (W)GAN related losses at every sample interval


    def plot_latent(self, i, scaler, scaled, X_train, y_train, X_train_scaled, epoch):
        latent_values = tf.random.normal([1000, self.Z])
        predicted_values = self.decoder(latent_values)

        predicted_values2 = self.decoder(self.encoder(X_train_scaled))
        predicted_values3 = self.encoder(X_train_scaled)
        #predicted_values4 = scaler.inverse_transform(X_train_scaled)
        if scaled == '-1-1':
            predicted_values = scaler.inverse_transform(predicted_values)
            predicted_values2 = scaler.inverse_transform(predicted_values2)
            
        elif scaled =='0-1':
            predicted_values = scaler.inverse_transform(predicted_values)
            predicted_values2 = scaler.inverse_transform(predicted_values2)
            
        if self.Z>=6:
            # set up a figure twice as wide as it is tall
            fig = plt.figure(figsize=plt.figaspect(0.5))
            aa = fig.add_subplot(1, 2, 1, projection='3d')
            aa.scatter(predicted_values3[:,0],predicted_values3[:,1],predicted_values3[:,2],c='pink')
            aa.scatter(latent_values[:,0],latent_values[:,1],latent_values[:,2],c='grey')
            aa.set_ylabel('Y')
            aa.set_zlabel('Z')
            aa.set_xlabel('X')
            aa.set_title('Latent Space 3D - 0,1,2')
    
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.scatter(predicted_values3[:,3],predicted_values3[:,4],predicted_values3[:,5],c='pink')
            ax.scatter(latent_values[:,3],latent_values[:,4],latent_values[:,5],c='grey')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlabel('X')
            ax.set_title('Latent Space 3D - 3,4,5')
            
            
            plt.savefig('AAE/Result/v'+str(i)+'_latent_space 3D_'+str(epoch+1)+'.png')
            plt.show()
    
    
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey=False, sharex=False)
            axes[0][0].scatter(predicted_values3[:,0],predicted_values3[:,1],c='pink')
            axes[0][0].scatter(latent_values[:,0],latent_values[:,1],c='grey')
            axes[0][0].set_xlabel("X")
            axes[0][0].set_ylabel("Y")
            axes[0][0].set_title("Latent Space 2D- XY")

            axes[0][1].scatter(predicted_values3[:,1],predicted_values3[:,2],c='pink')
            axes[0][1].scatter(latent_values[:,1],latent_values[:,2],c='grey')
            axes[0][1].set_xlabel("Y")
            axes[0][1].set_ylabel("Z")
            axes[0][1].set_title("Latent Space 2D- YZ")

            axes[0][2].scatter(predicted_values3[:,0],predicted_values3[:,2],c='pink')
            axes[0][2].scatter(latent_values[:,0],latent_values[:,2],c='grey')
            axes[0][2].set_xlabel("X")
            axes[0][2].set_ylabel("Z")
            axes[0][2].set_title("Latent Space 2D- XZ")
    
            axes[1][0].scatter(predicted_values3[:,3],predicted_values3[:,4],c='pink')
            axes[1][0].scatter(latent_values[:,3],latent_values[:,4],c='grey')
            axes[1][0].set_xlabel("X")
            axes[1][0].set_ylabel("Y")
            axes[1][0].set_title("Latent Space 2D- 3,4")
    
            axes[1][1].scatter(predicted_values3[:,4],predicted_values3[:,5],c='pink')
            axes[1][1].scatter(latent_values[:,4],latent_values[:,5],c='grey')
            axes[1][1].set_xlabel("Y")
            axes[1][1].set_ylabel("Z")
            axes[1][1].set_title("Latent Space 2D- 4,5")
    
            axes[1][2].scatter(predicted_values3[:,3],predicted_values3[:,5],c='pink')
            axes[1][2].scatter(latent_values[:,3],latent_values[:,5],c='grey')
            axes[1][2].set_xlabel("X")
            axes[1][2].set_ylabel("Z")
            axes[1][2].set_title("Latent Space 2D- 3,5")
    
            plt.tight_layout()
            plt.savefig('AAE/Result/v'+str(i)+'_Latent_Space 2D_'+str(epoch+1)+'.png')
            plt.show()
            
        else:
            print('Z < 6')
    
        ############################################
        ###############Latent Space#################
        ############################################
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharey=False, sharex=False)
        axes[0].scatter(predicted_values2[:,0],predicted_values2[:,1],)#encoder/decoder
        #axes[0].scatter(predicted_values4[:,0],predicted_values4[:,1],c='grey')#X_trained_scaled
        axes[0].set_ylabel('Y')
        axes[0].set_xlabel('X')
        axes[0].set_title('Autoencoder (X_trained_scaled)')

        axes[1].scatter(predicted_values[:,0],predicted_values[:,1],c='red') #decoder(latent space)
        #axes[1].scatter(predicted_values4[:,0],predicted_values4[:,1],c='grey')#X_trained_scaled
        axes[1].set_ylabel('Y')
        axes[1].set_xlabel('X')
        axes[1].set_title('Decoder (latent space)')
    
        plt.tight_layout()
        plt.savefig('AAE/Result/v_'+str(i)+'_epochs_'+str(epoch+1)+'.png')
        plt.show()
    
        ############################################
        ###############Discriminator################
        ############################################
        
        print('encoder output shape',predicted_values3.shape)
        
        
    
        resolution_x = 1000
        resolution_y = 1000
        x = np.linspace(np.min(-1), np.max(1), resolution_x)
        y = np.linspace(np.min(-1), np.max(1), resolution_y)
        x, y = np.meshgrid(x, y)
        
                
        random_points = np.concatenate((x, y), axis=1)
        random_points = random_points.reshape(-1, self.n_features).astype('float32')
        
        encoder_output = self.encoder(random_points)

        z = self.discriminator(encoder_output)
        z = np.array(z)
        z = z.reshape((len(x), len(y)))

        fig = plt.figure()
        plt.contourf(x, y, z)
        plt.colorbar()
        
        plt.savefig('AAE/Result/'+'countour_line_v'+str(i)+'_epochs'+str(epoch+1)+'.png')
        plt.show()
        
        resolution_x = 1000
        resolution_y = 1000
        x = np.linspace(np.min(-1), np.max(1), resolution_x)
        y = np.linspace(np.min(-1), np.max(1), resolution_y)
        x, y = np.meshgrid(x, y)
        
                
        random_points = np.concatenate((x, y), axis=1)
        random_points = random_points.reshape(-1, self.n_features).astype('float32')
        
        encoder_output = self.encoder(random_points)

        z = self.discriminator(encoder_output)
        z = np.array(z)
        z = z.reshape((len(x), len(y)))

        fig = plt.figure()
        plt.contourf(x, y, z)
        plt.colorbar()
        
        plt.savefig('AAE/Result/'+'countour_line_v'+str(i)+'_epochs'+str(epoch+1)+'.png')
        plt.show()


    
    def plot_loss(self, i, epoch):
        print('D_Loss: ')
        fig, ax = plt.subplots(1,1, figsize=[10,5])
        ax.plot(self.c1_hist, c='red')
        ax.plot(self.c2_hist, c='blue')

        ax.legend(['D real', 'D fake'])
        ax.grid()
        
        plt.tight_layout()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("AAE D_Loss per Epoch")
        
        plt.savefig('AAE/Losses/D_loss_v'+str(i)+'_epochs'+str(epoch+1)+'.png')
        plt.show()


        print('G_Loss: ')
        fig, ax = plt.subplots(1,1, figsize=[10,5])
        ax.plot(self.g1_hist, c='orange')
        ax.plot(self.g2_hist, c='green')

        ax.legend(['g_Decoder', 'g_Discriminator'])
        ax.grid()

        plt.tight_layout()
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("AAE G_Loss per Epoch")
        
        plt.savefig('AAE/Losses/G_loss_v'+str(i)+'_epochs'+str(epoch+1)+'.png')
        plt.show()
        
        
            
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
