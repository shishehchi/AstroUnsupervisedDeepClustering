"""
Implementation of the Deep Embedded Self-Organizing Map model
Convolutional Autoencoder helper function

@version 1.0
"""

from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Reshape, Flatten, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import math
import numpy as np

def cnn_autoencoder(encoder_dims, act='relu', init='glorot_uniform'):
    """
    Convolutional autoencoder model.

    # Arguments
        encoder_dims: list of number of units in each layer of encoder. encoder_dims[0] is input dim, encoder_dims[-1] is units in hidden layer (latent dim).
        The decoder is symmetric with encoder, so number of layers of the AE is 2*len(encoder_dims)-1
        act: activation of AE intermediate layers, not applied to Input, Hidden and Output layers
        init: initialization of AE layers
    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    n_stacks = len(encoder_dims)
    n_stacks_half = int((n_stacks - 3)/2)
    print ('n_stacks = ', n_stacks)
    print ('n_stacks_half = ', n_stacks_half)

    # Input
    im_size = int(math.sqrt(encoder_dims[0]))
    x = Input(shape=(encoder_dims[0], ), name='input')
    encoded = Reshape((im_size, im_size, 1), name='input_reshape')(x)
    print ('encoded.shape at input =', encoded.shape)

    # Internal layers in encoder
    for i in range(1, n_stacks_half + 1):
       print (i, encoder_dims[i])
       encoded = Conv2D(encoder_dims[i][0], kernel_size=(encoder_dims[i][1], encoder_dims[i][1]), activation=act, padding='same', name='encoder_conv2d_%d' % i)(encoded)
       encoded = MaxPooling2D(pool_size=(encoder_dims[i][2], encoder_dims[i][2]), name='encoder_maxpooling2d_%d' % i)(encoded)

    shape_before_flattening = K.int_shape(encoded)
    encoded = Flatten(name='encoded_flat')(encoded)
    decoded = Reshape(shape_before_flattening[1:], name='decoder_reshape')(encoded)

    # Internal layers in decoder
    print ('decoded.shape on entry = ', decoded.shape)
    for i in range(n_stacks_half + 2, n_stacks - 1):
        print (i, encoder_dims[i])
        decoded = Conv2D(encoder_dims[i][0], kernel_size=(encoder_dims[i][1], encoder_dims[i][1]), activation=act, padding='same', name='decoder_conv2d_%d' % (i-2))(decoded)
        decoded = UpSampling2D(size=(encoder_dims[i][2], encoder_dims[i][2]), name='decoder_upsampling2d_%d' % (i-2))(decoded)
    print (i+1, encoder_dims[i+1])
    decoded = Conv2D(encoder_dims[i+1][0], (encoder_dims[i+1][1], encoder_dims[i+1][1]), activation='sigmoid', padding='same', name='output')(decoded)
    decoded = Flatten(name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')
    autoencoder.summary()

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')
    encoder.summary()

    # Create input for decoder model
    encoded_input = Input(shape=(encoder_dims[n_stacks_half][0] * encoder_dims[n_stacks_half][2] * encoder_dims[n_stacks_half][2], ))
    print ('encoded_input.shape = ', encoded_input.shape)

    # Internal layers in decoder
    decoded = encoded_input
    decoded = autoencoder.get_layer('decoder_reshape')(decoded)
    for i in range(n_stacks_half + 2, n_stacks - 1):
        decoded = autoencoder.get_layer('decoder_conv2d_%d' % (i-2))(decoded)
        decoded = autoencoder.get_layer('decoder_upsampling2d_%d' % (i-2))(decoded)
    decoded = autoencoder.get_layer('output')(decoded)
    decoded = autoencoder.get_layer('decoder_0')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    decoder.summary()

    return (autoencoder, encoder, decoder)
