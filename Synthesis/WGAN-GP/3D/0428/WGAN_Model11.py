import numpy as np
import functools
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.optimizers import Adam
from network_3D11 import build_critic, build_generator
from tensorflow import reduce_mean
import gc
from sklearn.preprocessing import *


class WGAN():
    def __init__(self,k, n_features,latent_space,BATCH_SIZE,n_var ,use_bias, lr =0.0001, activation1 = 'relu', activation2='tanh'): #relu,tanh #lr=0.0001 

        self.n_features = n_features
        self.n_var = n_var
        self.BATCH_SIZE = BATCH_SIZE
        self.latent_space = latent_space
        self.n_critic = 5
        self.k = k

        # building the components of the WGAN-GP
        self.generator = build_generator(self.latent_space, n_var,self.n_features,use_bias=use_bias,activation1=activation1,activation2=activation2)
        self.critic = build_critic(n_var,use_bias,self.n_features)
        self.wgan = keras.models.Sequential([self.generator, self.critic])

        # setting hyperparemeters of the WGAN-GP
        self.generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.critic_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
       
        # saving the events of the generator and critic
        generator_log_dir = './content/generator'
        critic_log_dir = './content/critic'
        self.generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
        self.critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)

        # for prediction function
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(5e-7)


    ############################################################################
    ############################################################################
    # preprocessing

    def preproc(self, X_train, y_train, scaled):
        """
        Prepares the data for the WGAN-GP by splitting the data set
        into batches and normalizing it between -1 and 1.
        """
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


    ############################################################################
    ############################################################################
    # training

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def critic_loss(self, real_output,fake_output):
        return tf.reduce_mean(fake_output)-tf.reduce_mean(real_output)
        #return tf.reduce_mean(real_output)-tf.reduce_mean(fake_output)

    def gradient_penalty(self, f, real, fake):
        """
        WGAN-GP uses gradient penalty instead of the weight
        clipping to enforce the Lipschitz constraint.
        """

        # alpha = tf.random.normal([self.BATCH_SIZE, self.n_var], mean=0.0, stddev=0.1)
        alpha = tf.random.uniform(shape=[real.shape[0], self.n_features], minval=-1., maxval=1.)
        interpolated = real + alpha * (fake - real)

        with tf.GradientTape() as t:
            t.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        grad = t.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grad)) + 1e-12)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp


    @tf.function
    def train_G(self, batch):
        """
        The training routine for the generator
        """
        if batch.shape[0]==self.BATCH_SIZE:
            noise = tf.random.normal([self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=0.1)
        else:
            noise = tf.random.normal([batch.shape[0]%self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=0.1)

        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            fake_output = self.critic(generated_data, training=True)
            gen_loss = self.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        # return tf.math.abs(gen_loss)
        return gen_loss

    @tf.function
    def train_D(self, batch):
        """
        The training routine for the critic
        """
        if batch.shape[0]==self.BATCH_SIZE:
            noise = tf.random.normal([self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=0.1)
        else:
            noise = tf.random.normal([batch.shape[0]%self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=0.1)

        with tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)

            real_output = self.critic(batch, training=True)
            fake_output = self.critic(generated_data, training=True)

            disc_loss = self.critic_loss(real_output, fake_output)
            gp = self.gradient_penalty(functools.partial(self.critic, training=True), batch, generated_data)

            disc_loss += gp*10.0 

        gradients_of_critic = disc_tape.gradient(disc_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))

        # return tf.math.abs(disc_loss)
        return disc_loss
    # @tf.function
    def train(self, dataset, epochs, scaler, scaled, X_train=None, y_train=None):
        """
        Training the WGAN-GP
        """

        hist = []
        for epoch in range(epochs):

            start = time.time()
            print("Epoch {}/{}".format(epoch+1, epochs))

            for batch in dataset:

                for _ in range(self.n_critic):
                    self.train_D(batch)
                    disc_loss = self.train_D(batch)
                    self.critic_mean_loss(disc_loss)


                gen_loss = self.train_G(batch)
                self.generator_mean_loss(gen_loss)
                self.train_G(batch)

            with self.generator_summary_writer.as_default():
                tf.summary.scalar('generator_loss', self.generator_mean_loss.result(), step=epoch)

            with self.critic_summary_writer.as_default():
                tf.summary.scalar('critic_loss', self.critic_mean_loss.result(), step=epoch)

            hist.append([self.generator_mean_loss.result().numpy(), self.critic_mean_loss.result().numpy()])

            self.generator_mean_loss.reset_states()
            self.critic_mean_loss.reset_states()

            # outputting loss information
            print("critic: {:.6f}".format(hist[-1][1]), end=' - ')
            print("generator: {:.6f}".format(hist[-1][0]), end=' - ')
            print('{:.0f}s'.format( time.time()-start))

            if ((epoch+1)%10000 == 0) :
               
                #save the model
                self.wgan.save('./content/'+'wgan_v'+str(self.k)+'_epochs_'+str(epoch+1))
                self.generator.save('GANS/Models/generator_v'+str(self.k)+'_epochs_'+str(epoch+1))
                self.critic.save('GANS/Models/critic_v'+str(self.k)+'_epochs_'+str(epoch+1))
                
                self.plot_loss(self.k, epoch, hist)
                self.plot_latent(self.k, scaler, scaled, X_train, y_train, epoch)
                
                

        return hist
    
    
   # Plots the (W)GAN related losses at every sample interval


    def plot_latent(self, k, scaler, scaled, X_train, y_train, epoch):
        
        #sampling from the latent space without prediction
        latent_values = tf.random.normal([1000, self.latent_space], mean=0.0, stddev=0.1)
        
        #predict the labels of the data values on the basis of the trained model.
        predicted_values = self.generator.predict(latent_values)

        predicted_values[:,:]=(predicted_values[:,:])
        predicted_values2 = scaler.inverse_transform(predicted_values)

    
        print("Predicted Values:",predicted_values.shape)
        ab = plt.subplot(projection='3d')
        #ab.scatter(X_train[:,0],X_train[:,1],y_train, c='r')
        ab.scatter(predicted_values[:,0],predicted_values[:,1],predicted_values[:,2], c='green')
        ab.set_ylabel('Y')
        ab.set_zlabel('Z')
        ab.set_xlabel('X')
    
    
        plt.savefig('GANS/Result/Latent/v_'+str(k)+'_epochs_'+str(epoch+1)+'.png')
        print("save latent space")
        
        plt.show()
        
        ############################################
        
        x = predicted_values2[:,0]
        y = predicted_values2[:,1]
        z = predicted_values2[:,2]
        colour = self.critic(predicted_values)
        
        ab = plt.subplot(projection='3d')
        p = ab.scatter(x, y, z, c=colour)
        #ab.scatter(x, y, z, c=colour)
        ab.set_ylabel('Y')
        ab.set_xlabel('X')
        ab.set_zlabel('Z')
        
        plt.colorbar(p)
        plt.tight_layout()

        plt.savefig('GANS/Result/'+'countour_points_v'+str(k)+'_epochs'+str(epoch+1)+'.png')
        print("save countour line")
        plt.show()
        ##########################################
        
        #np.random.seed(100000)
        #a = np.random.uniform(-1, 1, size = (100000,3)) 


        #x = a[:,0]
        #y = a[:,1]
        #z = a[:,2]
        x_num = 100
        y_num = 100
        z_num = 100
        
        x = np.linspace(start=-1, stop=1, num=x_num)
        y = np.linspace(start=-1, stop=1, num=y_num)
        z = np.linspace(start= 0, stop=1, num=z_num)
        
        xyz = np.zeros((x_num*y_num*z_num, 3))
        for i in range(x_num):
            for j in range(y_num):
                for k in range(z_num):
                    xyz[i*x_num*y_num+j*z_num+k][0] = x[i]
                    xyz[i*x_num*y_num+j*z_num+k][1] = y[j]
                    xyz[i*x_num*y_num+j*z_num+k][2] = z[k]
                
        colour = self.critic(xyz)

        ax = plt.subplot(projection='3d')
        p = ax.scatter(xyz[:,0] , xyz[:,1], xyz[:,2], c=colour, cmap='Reds')
        plt.colorbar(p)
        plt.savefig('GANS/Result/'+'countour_mesh_v'+str(k)+'_epochs'+str(epoch+1)+'.png')
        plt.show()
        

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)
        #axes[0].scatter(X_train[:,0],X_train[:,1])
        axes[0].scatter(xyz[:,0] , xyz[:,1] , c=colour, cmap='Reds')
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
    

        #axes[1].scatter(X_train[:,1],y_train)
        axes[1].scatter(xyz[:,1],xyz[:,2] , c=colour, cmap='Reds')
        axes[1].set_xlabel("Y")
        axes[1].set_ylabel("Z")
    

        #axes[2].scatter(X_train[:,0],y_train)
        axes[2].scatter(xyz[:,0], xyz[:,2] , c=colour, cmap='Reds')
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Z")
    
        plt.tight_layout()
        plt.savefig('GANS/Result/'+'countour_mesh_slice_v'+str(k)+'_epochs'+str(epoch+1)+'.png')
        plt.show()
        
    
    
        
        '''''''''''''''''''''''
        x_num = 100
        y_num = 100
        z_num = 100
        
        x = np.linspace(start=-1, stop=1, num=x_num)
        y = np.linspace(start=-1, stop=1, num=y_num)
        z = np.linspace(start=-1, stop=1, num=z_num)
        
        xyz = np.zeros((x_num*y_num*z_num, 3))
        for i in range(x_num):
            for j in range(y_num):
                for k in range(z_num):
                    xyz[i*y_num+j][0] = x[i]
                    xyz[i*y_num+j][1] = y[j]
                    xyz[i*y_num+j][2] = z[k]
        
        disc_output = self.critic(xyz)
        disc_output = disc_output.numpy().reshape(x_num,y_num,z_num).T
        
        fig, ax = plt.subplots(1,1, figsize=(9,5))
        sns.heatmap(disc_output, ax=ax)
        ax.invert_yaxis()
        ax.invert_xaxis()
       
        plt.savefig('GANS/Result/'+'heatmap_v'+str(k)+'_epochs'+str(epoch+1)+'.png')
        plt.show()
   
        '''''''''''''''''''''''
       
       
    
    def plot_loss(self, k, epoch, hist):
        print('Loss: ')
        fig, ax = plt.subplots(1,1, figsize=[10,5])
        #plt.ylim([-0.5,0.8])
        ax.plot(hist)
        ax.legend(['loss_gen', 'loss_critic'],)
        
        #ax.set_yscale('log')
        ax.grid()
        plt.tight_layout()
        plt.savefig('GANS/Losses/GANS_loss_v'+str(k)+'_epochs'+str(epoch+1)+'.png')
        
        print("save loss")

        plt.show()
    ############################################################################
    ############################################################################
    # prediction

    def mse_loss(self, inp, outp):
        """
        Calculates the MSE loss between the x-coordinates
        """
        inp = tf.reshape(inp, [-1, self.n_features])
        outp = tf.reshape(outp, [-1, self.n_features])
        
        print("input:",inp.shape)
        print("input:",outp.shape)
        return self.mse(inp[:,0], outp[:,0])
        #return self.mse(inp[:,0], outp[:,0])
      


    def opt_step(self, latent_values, real_coding):
        """
        Minimizes the loss between generated point and inputted point
        """
        with tf.GradientTape() as tape:
            tape.watch(latent_values)
            gen_output = self.generator(latent_values, training=False)
            loss = self.mse_loss(real_coding, gen_output)

        gradient = tape.gradient(loss, latent_values)
        self.optimizer.apply_gradients(zip([gradient], [latent_values]))

        return loss

    def optimize_coding(self, real_coding):
        """
        Optimizes the latent space values
        """
        latent_values = tf.random.normal([len(real_coding), self.latent_space], mean=0.0, stddev=0.1)
        latent_values = tf.Variable(latent_values)


        loss = []

        for epoch in range(10000):
            # #print(loss[-1])
            # if loss[-1] > loss[-2]:
            #     lr = lr * 0.1
            #     self.optimizer = tf.keras.optimizers.Adam(lr
            
            loss.append(self.opt_step(latent_values, real_coding).numpy())

        return latent_values

    # def predict(self, input_data, scaler):
    #     """
    #     Optimizes the latent space of the input then produces a prediction from
    #     the generator.
    #     """
    #     predicted_vals = np.zeros((1, self.n_features))
    #
    #     for n in range(len(input_data)):
    #         print("Optimizing latent space for point ", n, " / ", len(input_data))
    #         real_coding = input_data[n].reshape(1, self.n_features)
    #         real_coding = tf.constant(real_coding)
    #         real_coding = tf.cast(real_coding, dtype=tf.float32)
    #
    #
    #         latent_values = self.optimize_coding(real_coding)
    #
    #         predicted_vals_1 = scaler.inverse_transform((self.generator.predict(tf.convert_to_tensor(latent_values)).reshape(1, self.n_features)))
    #         predicted_vals_1 = predicted_vals_1.reshape(1, self.n_features)
    #         predicted_vals = np.concatenate((predicted_vals, predicted_vals_1), axis=0)
    #
    #     predicted_vals = predicted_vals[1:,:]
    #     return predicted_vals

    # def predict(self, input_data, scaler):
    #     """
    #     Optimizes the latent space of the input then produces a prediction from
    #     the generator.
    #     """
    #     predicted_vals = np.zeros((1, self.n_features))
    #
    #     unscaled = scaler.inverse_transform(input_data)
    #
    #     for n in range(len(input_data)):
    #         print("Optimizing latent space for point ", n, " / ", len(input_data))
    #         real_coding = input_data[n].reshape(1,-1)
    #         real_coding = tf.constant(real_coding)
    #         real_coding = tf.cast(real_coding, dtype=tf.float32)
    #
    #         latent_values = self.optimize_coding(real_coding)
    #
    #
    #         predicted_vals_1 = scaler.inverse_transform((self.generator.predict(tf.convert_to_tensor(latent_values)).reshape(l, self.n_features)))
    #         predicted_vals_1 = predicted_vals_1.reshape(1, self.n_features)
    #         # print(predicted_vals_1)
    #         # predicted_vals_1[0,0] = unscaled[n,0]
    #         # print(predicted_vals_1)
    #         predicted_vals = np.concatenate((predicted_vals, predicted_vals_1), axis=0)
    #
    #     predicted_vals = predicted_vals[1:,:]
    #     return predicted_vals

    def predict(self, input_data, scaler):
        """
        Optimizes the latent space of the input then produces a prediction from
        the generator.
        """
        predicted_vals = np.zeros((1, self.n_features))
        unscaled = scaler.inverse_transform(input_data)
        latent_values = self.optimize_coding(input_data)


        predicted_vals_1 = scaler.inverse_transform((self.generator.predict(tf.convert_to_tensor(latent_values)).reshape(len(input_data), self.n_features)))
        predicted_vals_1 = predicted_vals_1.reshape(len(input_data), self.n_features)


        predicted_vals = predicted_vals_1[1:,:]
        return predicted_vals




    # Single Input is implemented above for prediction across the whole range
    # please uncomment the function below and comment out the function with the
    # same name above for it to run

    # def mse_loss(self, inp, outp):
    #     inp = tf.reshape(inp, [-1, self.n_features])
    #     outp = tf.reshape(outp, [-1, self.n_features])
    #     return self.mse(inp, outp)
