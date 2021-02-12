import os
import csv
import argparse
from time import time
import matplotlib.pyplot as plt
import numpy as np

# Tensorflow/Keras
import tensorflow as tf
from keras.models import Model

from AE2 import cnn_1dae
from SOM import SOMLayer


def som_loss(weights, distances):
    return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))


class DESOM:
    def __init__(self, input_dims, map_size, latent_dims):
        self.input_dims = input_dims
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.pretrained = False
        self.latent_dims=latent_dims
        
    def initialize(self, ae_act='relu', ae_init='glorot_uniform'):
        self.autoencoder, self.encoder, self.decoder = cnn_1dae(
            input_dims=self.input_dims,latent_dims=self.latent_dims)
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
        # Create DESOM model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.autoencoder.output, som_layer])
        
    @property
    def prototypes(self):
        return self.model.get_layer(name='SOM').get_weights()[0]
    
    
    def compile(self, gamma, optimizer):
        self.model.compile(loss={'decoder_0': 'mse', 'SOM': som_loss},
                           loss_weights=[1, gamma],
                           optimizer=optimizer)
        
    def predict(self, x):
        _, d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1)
    
    def decode(self, x):
        return self.decoder.predict(x)
    
    def map_dist(self, y_pred):
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp-labels)//self.map_size[1]
        d_col = np.abs(tmp%self.map_size[1]-labels%self.map_size[1])
        return d_row + d_col
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DESOM model

        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE

        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True
