from tensorflow import keras # using Tensorflow's Keras API
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np


def cnn_1dae(input_dims,latent_dims):
    #------------------- Encoder--------------    
    #Input layer
    input_layer = Input(shape=(input_dims,), name='input')
    x = Dense(450, activation='relu')(input_layer)
    #x = Dense(400, activation='relu')(x)
    #x= BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)
    encoded = Dense(latent_dims, activation='relu', name='z')(x)

    encoder = Model(input_layer, encoded, name='encoder')
    
    x = Dense(300,activation='relu',name='decoder_2')(encoded)
    #x = Dense(400,activation='relu',name='decoder_2')(x)
    #x= BatchNormalization()(x)
    x = Dense(450,activation='relu',name='decoder_1')(x) #layer needs to be copied below for decoder
    decoded = Dense(input_dims,activation='linear',name='decoder_0')(x) #layer needs to be copied below for decoder
    autoencoder = Model(input_layer, decoded, name='autoencoder')
    
    #stand alone decoder
    encoded_input = Input(shape=(latent_dims,))
    x=encoded_input
    #x = autoencoder.get_layer('decoder_3')(x)
    x = autoencoder.get_layer('decoder_2')(x)
    x = autoencoder.get_layer('decoder_1')(x)
    decoded = autoencoder.get_layer('decoder_0')(x)
    
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    print (autoencoder.summary())
    return (autoencoder, encoder, decoder)