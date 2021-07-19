from __future__ import print_function, division

import keras
from keras.optimizers import Adam
from keras import losses
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.models import Sequential, Model

import matplotlib.pyplot as plt
from network import build_discriminator, build_generator
import tensorflow as tf
from tensorflow import reduce_mean
from sklearn.preprocessing import *

import numpy as np
import functools
import time
import gc



class DC_GAN():
    def __init__(self, n_features,latent_space, BATCH_SIZE, nodes, i):

        #self.X_shape = (n_features,)
        self.latent_space = latent_space
        self.n_features = n_features
        self.BATCH_SIZE = BATCH_SIZE
        self.nodes = nodes
        self.i = i
        
        # for prediction function
        self.cross = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer2 = tf.keras.optimizers.RMSprop(learning_rate=1e-4)#1e-9
        #optimizer for gan
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4)#1e-4

        # Build the discriminator
        self.discriminator = build_discriminator(self.n_features,self.nodes, self.latent_space,self.BATCH_SIZE)
        self.discriminator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.discriminator_optimizer = self.optimizer

        # Build the generator
        self.generator = build_generator(self.latent_space,self.nodes, self.n_features)
        self.generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.generator_optimizer = self.optimizer
        
        # Stacking together
        self.dcgan = Sequential([self.generator, self.discriminator])

    ####################define loss######################
    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross(tf.ones_like(real_output), real_output)
        fake_loss = self.cross(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output):
        return self.cross(tf.ones_like(fake_output), fake_output)
    
        ###############Compile model############
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

    @tf.function
    def train_step(self,batch):

        if batch.shape[0]==self.BATCH_SIZE:
            noise = tf.random.normal([self.BATCH_SIZE, self.latent_space],0.0,0.1)
        else:
            noise = tf.random.normal([batch.shape[0]%self.BATCH_SIZE, self.latent_space],0.0,0.1)
            
            
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_x = self.generator(noise, training=True)
            
            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_x, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            
       #Train G
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    
        self.generator_mean_loss(gen_loss)
        self.discriminator_mean_loss(disc_loss)

    ####################Preprocessing######################  
        
    def preproc(self, X_train, y_train, scaled):

        sample_data = np.concatenate((X_train, y_train), axis=1)


        if scaled == '-1-1':
            scaler = MinMaxScaler(feature_range=(-1,1))
            X_train_scaled = scaler.fit_transform(sample_data)
        elif scaled =='0-1':
            scaler = MinMaxScaler(feature_range=(0,1))
            X_train_scaled = scaler.fit_transform(sample_data)

        train_dataset = X_train_scaled.reshape(-1, self.n_features)
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
    
    def train(self,i, dataset, epochs, scaler, scaled, X_train, y_train):
        
        hist = []
        
        for epoch in range(epochs):
            start = time.time()
            print("Epoch {}/{}".format(epoch+1, epochs))

            for batch in dataset:
                self.train_step(batch)
                
            hist.append([self.generator_mean_loss.result().numpy(), self.discriminator_mean_loss.result().numpy()])

            self.generator_mean_loss.reset_states()
            self.discriminator_mean_loss.reset_states()
                
            i=self.i
            if ((epoch+1) % 5000 == 0):
                #save the model
                self.dcgan.save('./GANs/Models/GAN_'+'v'+str(i)+'_epoch='+str(epoch+1)+'.h5')
                self.generator.save('GANS/Models/generator_'+'v'+str(i)+'_epoch='+str(epoch+1))
                self.discriminator.save('GANS/Models/discriminator_'+'v'+str(i)+'_epoch='+str(epoch+1))
                
                self.plot_loss(i, hist, epoch)
                self.plot_latent(i, scaler, scaled, X_train, y_train, epoch)
                
                print('save the model & result')
                
            
        return hist
    
    
    def plot_latent(self, i, scaler, scaled, X_train, y_train, epoch):
        
        #sampling from the latent space without prediction
        latent_values = tf.random.normal([1000, self.latent_space], mean=0.0, stddev=0.1)
        #predict the labels of the data values on the basis of the trained model.
        predicted_values = self.generator.predict(latent_values)
        predicted_values[:,:]=(predicted_values[:,:])
        predicted_values = scaler.inverse_transform(predicted_values)
        
        plt.scatter(X_train, y_train, c='r')
        plt.scatter(predicted_values[:,0],predicted_values[:,1], c='green')
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.title('generated result')
        plt.tight_layout()
        plt.savefig('GANs/Result/'+'v'+str(i)+'_epoch='+str(epoch+1)+'.png')
        plt.show()

    
    
    def plot_loss(self,i, hist, epoch):
        fig, ax = plt.subplots(1,1, figsize=[10,5])
        #ax.set_ylim([-0.5,0.8])
        ax.set_title('Loss')
        ax.plot(hist)
        ax.legend(['loss_gen', 'loss_disc'],)

        #ax.set_yscale('log')
        plt.grid()
        plt.tight_layout()
        
        plt.savefig('GANS/Losses/GANS_loss_'+'v'+str(i)+'_epoch='+str(epoch+1)+'.png')
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
            gen_output = self.generator(latent_values, training=False)
            loss = self.mse_loss(real_coding, gen_output)

        gradient = tape.gradient(loss, latent_values)
        self.optimizer2.apply_gradients(zip([gradient], [latent_values]))

        return loss

            
            
    def optimize_coding(self, real_coding):
        """
        Optimizes the latent space values
        """
        latent_values = tf.random.normal([len(real_coding), self.latent_space],0.0,0.1)
        latent_values = tf.Variable(latent_values)


        loss = []

        for epoch in range(10000):
            loss.append(self.opt_step(latent_values, real_coding).numpy())

        return latent_values            
            
    def predict(self, input_data, scaler):

        predicted_vals = np.zeros((1, self.n_features))
        input_data
        unscaled = scaler.inverse_transform(input_data)
        latent_values = self.optimize_coding(input_data)


        predicted_vals_1 = scaler.inverse_transform((self.generator.predict(tf.convert_to_tensor(latent_values)).reshape(len(input_data), self.n_features)))
        predicted_vals_1 = predicted_vals_1.reshape(len(input_data), self.n_features)


        predicted_vals = predicted_vals_1[1:,:]
        return predicted_vals



