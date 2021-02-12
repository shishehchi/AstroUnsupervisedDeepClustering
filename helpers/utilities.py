# -*- coding: utf-8 -*-



from timeit import default_timer as timer
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#Random sampling
from random import sample


#-----------------------------Global Variables(used in plenty of places throughout code)
idx_map = {}
node_map = {}

#Maybe? distance_map = None



#--------------------------------------------Specify paths
current_path = os.getcwd()
img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/closest_samples/')





'''
Takes in Map size

RETURNS a list of cords for whole map
'''
def generate_list_of_coords(map_size): 
    #Find map size
    #map_size = desom.map_size
    
    coords = [] #List of Grid Coords
    for k in range(map_size[0] * map_size[1]):
        x = k // map_size[1]
        y = k % map_size[1]

        coords.append((x,y))
    return coords



'''
Requires Grid Coordinates(From generate_list_of_coords)
and map_size



Create a Mapping Dict for (Node => IDX)   CALLED idx_map
'''
def get_idx_map(grid_coordinates, map_size):
    #Refer to global
    global idx_map 
    #Populate
    for k in grid_coordinates:
            w = map_size[0] #Width of MAP
            #Convert Grid NODE to IDX
            arr_i = k[0] + w * k[1]

            #Initialize
            idx_map[k] = arr_i
    print("idx_map: maps from NODE to IDX")
    return idx_map
     
    
    
    
'''
REQUIRES map size


Creates and RETURNS a Mapping Dict for (IDX => Node)   CALLED node_map

also SETS global ref
'''
def get_node_map(map_size): 
    global node_map 
    for k in range(map_size[0] * map_size[1]): 
             #Convert to grid NODE
            x = k // map_size[1]
            y = k % map_size[1]
            #Form coordinate
            node = (x,y)
            #IDX -> Node
            node_map[k] = node
    print("node_map: maps from IDX to NODE")
    
    return node_map



#--------------------------------------------   Distance map Utility functions ------------------------

'''
Generates density map for given SOM, using the given DISTANCE_MAP
Assumption: Distance_map is for the given SOM-1

PARAMETERS:
    - SOM
    - Distance Map
RETURNS:
    - Log10-transformed Density matrix
'''
def generate_density_matrix(som, distance_map):
    #Initialize
    M = np.zeros(shape = (som.map_size[0] * som.map_size[1]))
    print(M.shape)

    #Find NEAREST samples for given BMU
    for bmu in range(distance_map.shape[1]):   
        distances = distance_map[:, bmu]
        #Minimum distance value
        min_dist = np.min(distances)
        #Specify indices of data points
        closest_idx = np.where(distances == min_dist)[0]
        #Heatmap value
        M[bmu] = len(closest_idx)

    #Log10-transform
    M = M + 1
    M = np.log10(M)
    
    #Bin 
    print("Binning M with 3 bins")
    counts, bin_edges = np.histogram(M, bins = 3)
    print("Counts: {} \n ".format(counts))
    print("Bin Edges: {} \n ".format(bin_edges))
    
    
    return M, bin_edges



'''
Generates appropriate colormap value for index(k) depending
    on its location on M (based on it's value)

PARAMETERS:
    - k
    - M
    - bin_edges
RETURNS:
    - cmap (String)
'''
def find_cmap(k, M, bin_edges):
    
    #Default
    cmap = "Greys"
    
#     #Conditions
#     c1 = k in np.where(np.logical_and(M > bin_edges[0], M < bin_edges[1]))[0]
#     c2 = k in np.where(np.logical_and(M > bin_edges[1], M < bin_edges[2]))[0]
#     c3 = k in np.where(np.logical_and(M > bin_edges[2], M < bin_edges[3]))[0]
    
    if(k in np.where(np.logical_and(M > bin_edges[0], M < bin_edges[1]))[0]):
        cmap ="Greens"
    elif(k in np.where(np.logical_and(M > bin_edges[1], M < bin_edges[2]))[0]):
        cmap = "Blues"
    elif(k in np.where(np.logical_and(M > bin_edges[2], M < bin_edges[3]))[0]):
        cmap = "Reds"
                
    return cmap



'''
Generates colors and labels based on bin edges

PARAMETERS:
    - bin_edges
RETURNS:
    - colors
    - labels
'''
def get_colors_and_legend(bin_edges):
    
    #Initialize default
    labels = ["density < {}".format(np.around(bin_edges[0], 3))]
    
    for i in range(len(bin_edges) - 1):
        
        low = np.around(bin_edges[i], 3)
        hi = np.around(bin_edges[i+1], 3)
        c = "{} <= density < {}".format(low,hi)
        
        labels.append(c)
    
    colors = ['grey','green', 'blue','red'] #Order matters!
    
    print("Labels: {}".format(labels))
    print("Colors: {}".format(colors))

    
    return colors,labels





'''
*** MAIN PLOTTING FUNCTION ***

Plots color-coded SOM-1

PARAMETERS:
    - SOM
    - Node_map
    - Decoded Prototypes
RETURNS:
    -
'''
def plot_color_coded_SOM1(som, node_map, decoded_prototypes, distance_map):
    
    #Find the density matrix
    M, bin_edges = generate_density_matrix(som, distance_map)
    map_size = som.map_size

    #Set up 32x32 size
    img_size = 32

    #Setup plot
    fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10, 10))
    #Iterate over each and every prototype
    for k in range(map_size[0] * map_size[1]):
        #Find appropriate CMAP
        cmap = find_cmap(k, M, bin_edges)      
        #Quick lookup
        bmu_node = node_map[k]
        x = bmu_node[0]
        y = bmu_node[1]       
        
        ax[x][y].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap = cmap)
        ax[x][y].axis('off')

    #Use helper to generate 
    colors, labels = get_colors_and_legend(bin_edges)

    plt.subplots_adjust(hspace = 0.05, wspace = 0.05)  
    patches =[mpatches.Patch(color = colors[i],
                             label = labels[i]) for i in range(len(colors))]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.5),  borderaxespad=0.1)
   
    #Filename
    filename = 'density' +'.png'
    img_savepath = os.path.join(current_path,'results/plots/',filename)
    plt.savefig(img_savepath)

    
    return 






























    
'''
Requires Distance-Map Result

Higlight NODE for a given COORDINATE PAIR

#PASS IN : DECODED PROTOTYPES
            map_size

            NODE_MAP dict and IDX_MAP for fast lookup

'''
def highlight_node(grid_coords, map_size ,
                   decoded_prototypes,
                   idx_map, node_map):  
    
    #Get width
    w = map_size[0]
    
    #Array index
    #arr_i = grid_coords[0] + w * grid_coords[1]
    arr_i = idx_map[grid_coords]
    
    #---------------------Plot
    #Set up 32x32 size
    img_size = 32
    #Setup plot
    fig, ax = plt.subplots(map_size[0], map_size[1], figsize=(10, 10))

    #Iterate over each and every prototype
    for k in range(map_size[0]*map_size[1]):
        
        #Extract coordinates
        coords = node_map[k]
        #Find coordinates
        x = coords[0]
        y = coords[1]
   
        ax[x][y].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap='gray')
        ax[x][y].axis('off')

        #Highlight the one we need
        if(k==arr_i):
            ax[x][y].imshow(decoded_prototypes[k].reshape(img_size, 
                                                                img_size),
                                                         cmap='inferno')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    #---------Filename
    filename = 'highlighted_grid.png'
    #Slightly alter
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/')
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, 
                               filename)
    
    plt.savefig(im_savepath)
    print("Highlighted Grid saved as {} at {}".format(filename, im_savepath))
    
    
    
  


    
'''
Takes in BMU Coords (0-indexed)
and Distance Map
and MAP_SIZE for formatting
and X_train

Returns ----------------closest IMAGES (32x32 each)

'''
def find_closest_samples(grid_coords,
                         distance_map,
                         map_size,
                         X_train,
                        verbose = True):
    #Setup Map Size
    #map_size = desom.map_size
    
    #Get width
    w = map_size[0]
    
    #Array index(USE DICT LATER!)
    arr_i = grid_coords[0] + w * grid_coords[1]
    
    
    #Access
    A = distance_map[:, arr_i]

    #Indices of location with closest nodes
    closest_idx = np.asarray((np.where(A == np.min(A)))).flatten()
    
    #Collect samples from original data
    closest_samples = []
    for idx in closest_idx:
        #Extract sample from data and reshape
        closest_samples.append(X_train[idx].reshape(32,32))
        
    if(verbose):
        print("{} matching samples found.".format(len(closest_samples)))
    
    return closest_samples





'''
REQUIRES:  Samples with Minimum Manhattan Distance
         Coordinates (for filename)

Plot them

'''
    
def plot_closest_samples_and_save(closest_samples, coords):
    #How many nearby samples?
    num_samples = len(closest_samples)

    if(num_samples > 20):
        #Select only 20 randomly
        closest_img_list = sample(closest_samples, 20)
        num_samples = 20
    else:
        closest_img_list = closest_samples
    #Setup plot
    plt.figure(figsize=(20, 4))

    for i in range( 1, num_samples):
        ax = plt.subplot(1, num_samples, i)
        plt.imshow(closest_img_list[i].reshape(32,32), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #Filename
    filename = 'closest_'+ str(coords[0]) + '_' + str(coords[1]) +'.png'
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, filename)   
    plt.savefig(im_savepath)
    
    print("Nearest samples saved as {} at {}.".format(filename, im_savepath))



    
    
    
    
    
    
    
    
    








































