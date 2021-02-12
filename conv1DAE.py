from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Flatten
from keras.models import Model
from keras import backend as K


def cnn_1dae(input_dims,latent_dims= 512):
    
    #------------------- Encoder--------------
    
    #Input layer
    input_layer = Input(shape=(input_dims,), name='input')
    #Reshape to 3D before Convolution
    reshaped_layer = Reshape((input_dims, 1))(input_layer)

    #Encoder Convolutions and MAX-Pooling Layers
    x = Conv1D(128, 3, activation='relu', padding='same')(reshaped_layer) 
    x = MaxPooling1D(pool_size=2)(x) 
    x = Conv1D(64, 3, activation='relu', padding='same')(x) 
    x = MaxPooling1D(pool_size=2)(x) 
    x = Conv1D(32, 3, activation='relu', padding='same')(x) 
    x = MaxPooling1D(pool_size=2)(x) 
    x = Conv1D(8, 3, activation='relu', padding='same')(x) 
    x = MaxPooling1D(pool_size=2)(x) 
    #----------Store dimensions before we flatten
    shape_before_flattening = K.int_shape(x)[1:]
    #Unpack
    dims, fmaps = shape_before_flattening
    print("Shape before flattening : {}".format(shape_before_flattening))
    #Store number of neurons for Dense layer later
    num_neurons = dims*fmaps 

    #Wrap up encoder
    x = Flatten()(x)
    encoded = Dense(latent_dims, activation='relu', name='z')(x)  #Last layer in Encoder
    #Get the latent dimension
    z_dims =  K.int_shape(encoded)[1] #LATENT using K-backend

    #Define model
    encoder = Model(input_layer, encoded, name='encoder')
    encoder.summary()

    
    
    #------------------ Rest of Autoencoder--------------------
    #Pack with enough Dense neurons
    x = Dense(num_neurons, activation='relu',name='d_in')(encoded)
    #Reshape for Convolutions
    x =  Reshape(shape_before_flattening, name='d_3d')(x)
    x = UpSampling1D(2, name = 'decoder_8')(x) 
    x = Conv1D(32, 3, activation='relu', padding='same', name = 'decoder_7')(x) 
    x = UpSampling1D(2, name = 'decoder_6')(x) 
    x = Conv1D(64, 3, activation='relu', padding='same', name = 'decoder_5')(x) 
    x = UpSampling1D(2, name = 'decoder_4')(x) 
    x = Conv1D(128, 3, activation='relu', padding='same', name = 'decoder_3')(x) 
    x = UpSampling1D(2, name = 'decoder_2')(x) 
    x = Conv1D(1, 3, activation='linear', padding='same', name='decoder_1')(x) 
    
    #Last layer in AE
    decoded = Flatten(name='decoder_0')(x)

    autoencoder = Model(input_layer, decoded, name='autoencoder')
    autoencoder.summary()
    
    #----------------------- Standalone Decoder------------------
    # Create input for decoder model
    encoded_input = Input(shape=(latent_dims,))
    # Internal layers in decoder
    decoded = encoded_input

    #Unpack layers
    decoded = autoencoder.get_layer('d_in')(decoded)
    decoded = autoencoder.get_layer('d_3d')(decoded)
    decoded = autoencoder.get_layer('decoder_8')(decoded)
    decoded = autoencoder.get_layer('decoder_7')(decoded)
    decoded = autoencoder.get_layer('decoder_6')(decoded)
    decoded = autoencoder.get_layer('decoder_5')(decoded)
    decoded = autoencoder.get_layer('decoder_4')(decoded)
    decoded = autoencoder.get_layer('decoder_3')(decoded)
    decoded = autoencoder.get_layer('decoder_2')(decoded)
    decoded = autoencoder.get_layer('decoder_1')(decoded)

    #Final Flattened output
    decoded = autoencoder.get_layer('decoder_0')(decoded)

    #Done connecting...
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    decoder.summary()

    return (autoencoder, encoder, decoder)