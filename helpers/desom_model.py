import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#-------------------------------------------Imports

#Import a class
from DESOM1 import DESOM
#from src.DESOM.SOM import SOMLayer

# #Load model
from keras.models import load_model
from keras.models import Model
# from timeit import default_timer as timer
import os
import numpy as np
from glob import glob
import pathlib
import sys
import matplotlib.pyplot as plt


# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# we can explicitly make assignments on it 
#this.desom = None

#------------------------------------------Load model
flag = False
desom = None  #Initialize as none

#Distance_map
#distance_map



#--------------------------------------------Specify paths
current_path = os.getcwd()
#Where are Images Stored?
data_path =   os.path.join(current_path, 'data/im')







'''
Loads a pre-trained DESOM model
PARAMETERS:
    - Map Size (Tuple)
    - AE_type (String) , eg. cnn1D, cnn2D, etc.
    - input_dims
    - Paths to AE weights and Model weights
RETURNS:
    - Loaded DESOM model
'''
def load_desom_trained(map_size, ae_type, input_dims, ae_path, model_path):

    encoder_dims = [input_dims]
    som = DESOM(encoder_dims= encoder_dims, ae_type = ae_type, map_size = map_size )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'
    
    #Initialize DESOM
    som.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    som.compile(gamma, optimizer)
    
    #-----------------Load AE Weights and Trained Model weights
    som.load_ae_weights(ae_path)
    som.load_weights(model_path)
    
    return som


'''
Predicts and Returns the appropriate BMU for a given sample
'''
def predict_bmu(x, som, decoded_prototypes):
    #Ensure correct input format
    x = x.reshape(1, -1)
    print("x(in) shape: {}".format(x.shape))
    #Find spot and BMU together
    k = som.predict(x)[0]  #Predicted SPOT on SOM
    predicted_bmu = decoded_prototypes[k]
    
    return k, predicted_bmu



'''
Reconstructs a  specified row in the dataset using the AE
PARAMETERS:
    - Data
    - SOM
    - idx to reconstruct
RETURNS:
    - Original, Reconstructed
'''
def reconstruct_sample_from_data(X, som, idx):
    
    x = X[idx].reshape(1,-1)
    #Reconstructed
    x_rec  = som.autoencoder.predict(x)
    
    return x, x_rec



'''
Reconstructs a given sample
PARAMETERS:
    - Data
    - SOM
RETURNS:
    - Reconstructed
'''
def reconstruct_sample(x, som):
    
    x = x.reshape(1,-1)
    #Reconstructed
    x_rec  = som.autoencoder.predict(x)
    
    return x, x_rec



'''
PLots the given spectra on top of each other
PARAMETERS:
    -  Spectra
RETURNS:
    - Original, Reconstructed
'''
def plot_reconstruction(x, x_rec):
    
    fig, axes = plt.subplots(1,1, figsize = (10,5), squeeze = False)

    #Plot each line
    axes[0][0].plot(x[0], color='green', label='Original', linewidth=1)
    axes[0][0].plot(x_rec[0], color='red', linestyle='--',  label='Reconstructed')

    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend()
    
    fname = "reconstruction.png"
    print("Saved reconstruction at :{}".format(fname))
    plt.savefig(fname)
    
    plt.show()
    
    return 



'''
PLots the given spectra on top of each other
PARAMETERS:
    -  Spectra
RETURNS:
    - Original, Reconstructed
'''
def plot_reconstructed_image(x, x_rec):
    
    fig, axes = plt.subplots(1,2, figsize = (10,5), squeeze = False)
    
    #Plot each line
    axes[0][0].imshow(x.reshape(32,32), cmap='gray')
    axes[0][0].set_title("Original")
    
    axes[0][1].imshow(x_rec.reshape(32,32), cmap='gray')
    axes[0][1].set_title("Reconstructed")

    
    fname = "reconstruction.png"
    print("Saved reconstruction at :{}".format(fname))
    plt.savefig(fname)
    plt.show()
    
    return 



# ---------------------------------------------------------------------------------
























'''
- Defines Architecture
- loads  Pretrained Model and AE
- Compiles model
- Returns model
'''
def load_trained_model(filters, map_w, map_h, pretrained_ae, pretrained_model): 
    #Access global
    global flag
    #Refer to Global Model
    global desom 

    #------------------------------------Good model (DESOM 1)--------------------------
    #filters = 32 
    
    #i = pool_size
    i = 2
    ks = 3
    #desom is 15x15 with 8 filters
    desom = DESOM(encoder_dims= [np.power(32,2), 
                                [filters, ks, i],
                                [filters, ks, i], 
                                [filters, ks, i],
                                [filters,ks, i], 
                                                    [],
                                 [filters, ks, i],
                                 [filters, ks, i],
                                 [filters, ks, i], 
                                 [filters, 5, i],

                                 [1, ks, 0]],
                        ae_type = 'cnn',
                        map_size = (map_w, map_h) )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    
    #-----------------Load AE Weights and Trained Model weights
    desom.load_ae_weights(pretrained_ae)
    desom.load_weights(pretrained_model)
    
    flag = "Model Compiled" #Update
    
    return desom





'''
-----------------------LARGER MAPS-----------------
- Defines Architecture
- loads  Pretrained Model and AE FROM SPECIFIED Directories
- Compiles model
- Returns model
'''
def load_som1(pretrained_ae, pretrained_model): 
    #Access global
    global flag
    #Refer to Global Model
    global desom 

    #------------------------------------Good model (DESOM 1)--------------------------
    filters = 16 
    #i = pool_size
    i = 2
    ks = 3
    #desom is 15x15 with 8 filters
    desom = DESOM(encoder_dims= [np.power(32,2), 
                                [filters, ks, i],
                                [filters, ks, i], 
                                [filters, ks, i],
                                [filters,ks, i], 
                                                    [],
                                 [filters, ks, i],
                                 [filters, ks, i],
                                 [filters, ks, i], 
                                 [filters, 5, i],

                                 [1, ks, 0]],
                        ae_type = 'cnn',
                        map_size = (25,25) )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    
#     #Specify directories
#     pretrained_autoencoder_ping = os.path.join(current_path,'ping_work/ae_weights-epoch100.h5')
#     pretrained_model_ping = os.path.join(current_path,'ping_work/DESOM_model_final.h5')
   
    
    #-----------------Load AE Weights and Trained Model weights
    desom.load_ae_weights(pretrained_ae)
    desom.load_weights(pretrained_model)
    
    flag = "Model Compiled" #Update
    
    return desom




'''
PARAMETERS:
    - DATA (X, y)
    - # Filters
    - Map Size
    - AE Epochs
    - SOM/DESOM Iterations
    - Save directories
- Defines Architecture
- Trains AE according to specified pre_train epochs
- Trains Model and SOM according to specified epochs
- SAVES to specified files
- Compiles model
- Returns model
'''
def train_specific_model(X_train, y_train,
                          #Architecture
                          num_filters, map_size,
                          #Epochs
                          ae_epochs, som_iters, model_iters,
                          #Save directories
                          ae_savepath,
                          model_savepath): 
    
    #Refer to Global Model
    global desom 

    #------------------------------------Good model (DESOM 1)--------------------------
    filters = num_filters #flat-32
    #i = pool_size
    i = 2
    ks = 3
    desom = DESOM(encoder_dims= [np.power(32,2), 
                                [filters, ks, i],
                                [filters, ks, i], 
                                [filters, ks, i],
                                [filters,ks, i], 
                                                    [],
                                 [filters, ks, i],
                                 [filters, ks, i],
                                 [filters, ks, i], 
                                 [filters, 5, i],

                                 [1, ks, 0]],
                        ae_type = 'cnn',
                        map_size = map_size )
    #Specify Gamma and Optimizer
    gamma = 0.001
    optimizer = 'adam'

    #Initialize DESOM
    desom.initialize(ae_act ='relu', ae_init ='glorot_uniform')
    #Compile
    desom.compile(gamma, optimizer)
    
   
    
    #-----------------Pretrain AE and Train Model
    #Pretrain AE
    desom.pretrain(X_train, 
                  optimizer='adam',
                     #Epochs
                     epochs = ae_epochs,
                     batch_size = 256,
                     save_dir= ae_savepath)
    
    
    #Train model
    desom.init_som_weights(X_train)
    desom.fit(X_train, 
              y_train,              
              Tmax = 10.0,
              #Iterations
              iterations = model_iters,
              som_iterations= som_iters,
              save_epochs = 100,
              save_dir = model_savepath)
           
    flag = "Model Compiled" #Update
    
    #Find distance_map and save it
    distance_map  = get_distance_map(X_train)
    save_distance_map(distance_map, model_savepath)
    
    return desom
    
    
    
'''
Plot DESOM Map and Save it to directory
Optional Parameter: Filename
If multiple, save all epochs in different folder
'''
def save_grid_plot(multiple = False, fname = 'DESOM_Grid'):
    #Access global 
    global desom
    
    #Find Decoded prototypes
    decoded_prototypes = desom.decode(desom.prototypes)

    #Setup Map Size
    map_size = desom.map_size

    #Set up 32x32 size
    img_size = 32
    #img_size = int(np.sqrt(X_train.shape[1]))

    #Setup plot
    fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10, 10))
    #Iterate over each and every prototype
    for k in range(map_size[0]*map_size[1]):
        ax[k // map_size[1]][k % map_size[1]].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap='gray')
        ax[k // map_size[1]][k % map_size[1]].axis('off')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    #Filename
    filename = fname +'.png'
    img_savepath = os.path.join(current_path,'results/plots/',filename)
    
    if(multiple):
        #Filename
        filename = fname +'.png'
        img_savepath = os.path.join(current_path,'results/plots/multiple_epochs/',filename)
        
    plt.savefig(img_savepath)
    print("DESOM Map saved as {} at {}.".format(filename, img_savepath))
    
    return
    
    
    
    
'''
This function uses the LOADED DESOM to generate a Distance Map
saved_X specifies location
RETURNS the Distance map
'''    
def get_distance_map(desom, X):

    #Get Predicted Labels
    y_pred = desom.predict(X)

    #Distance map
    # i - point in X(data)
    # j - Assigned CELL on SOM
    distance_map = desom.map_dist(y_pred)
    
    return distance_map



'''
Saves the given distance_map NumPy ARRAY at given savepath folder
'''
def save_distance_map(distance_map, savepath):
    #Save as NumPy array
    name = "distmap.npy"
    filename = os.path.join(savepath, name)
    np.save(filename,distance_map)
    print("Saved at {}".format(filename))

    return 




'''
Saves the distance_map NumPy ARRAY from the given savepath folder
'''
def load_distmap(savepath):
    name = "distmap.npy"
    filename = os.path.join(savepath, name)
    
    distance_map = np.load(filename)
    print("Loaded from {}".format(filename))
    
    return distance_map



'''
Simply returns map size
VERY HELPFUL
'''
def get_map_size():
    global desom
    return desom.map_size


'''
Find DECODED Prototypes
'''
def get_decoded_prototypes():
    global desom
    decoded_prototypes = desom.decode(desom.prototypes)
    
    return decoded_prototypes
    




def print_save_directories():
    #Trained Model directories
    print("Pretrained AE at: " + pretrained_autoencoder)
    print("Pretrained DESOM (15x15) at: " + pretrained_model)

    
    
'''
Check if model is not None.
If it has been defined, print map size and architecture.
'''
def print_model_summary():
    #Refer to global
    global desom
    
    if(desom == None):
        print("Model has not been defined!")
        return
    else:
        print("Model has been defined...")
        print("Map Size: {}".format(get_map_size()))
        print(model.summary())
        
        
        
    
















