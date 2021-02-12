#Module contains all methods for analyzing Images

import os
import numpy as np
import pandas as pd

from helpers import preprocessor as prep
#from helpers import astro_utilities  as astroutils


import seaborn as sn
from timeit import default_timer as timer
import matplotlib.pyplot as plt
#from astropy.visualization import (SqrtStretch,ImageNormalize, ZScaleInterval)



#Figure sizes
fig_size = (10,10)
sn_fig_size = (12,8)  #For heatmaps



#--------------------------------------------Specify paths
current_path = os.getcwd()
img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/heatmaps/')






'''
Takes in a Vector of Predicted Indices corresponding to MAP
-- and a list of grid NODE coordinates
-- and map-size tuple
-- and idx_map, node map for fast lookup

For N-Cutouts, we have a prediction Vector that we pass in.

RETURNS  a Vector of BMU COUNTS for a given Prediction vector
'''
def get_bmu_counts(y_pred, grid_coordinates, map_size, node_map, idx_map):
    
    #Initialize Dict with zeros
    grid_dict ={}  #Key-Node, Value = Counts
    #Initialize counts
    for k in grid_coordinates:
        #Initialize
        grid_dict[k] = 0
        
    #Iterate over predictions
    for k in range(len(y_pred)):
        #Extract k-th element of prediction (== GRID-IDX)
        pred_idx = y_pred[k]   
        
        #Use dict to lookup corresponding NODE
        node = node_map[pred_idx]

        #Update Node COUNT in dict
        grid_dict[node] += 1
        
    #Form Return Vector of correct size
    data_vec = list(np.zeros(map_size[0]*map_size[1])) #Will be length 225
    #Populate
    for k,v in grid_dict.items():       
        #Use Dict to lookup corresponding IDX given node
        arr_i = idx_map[k]
        #Modify in-place
        data_vec[arr_i] = v #the stored Count
              
    return np.array(data_vec)


# '''
# Takes in a particular 


# '''
# def plot_CCD(data):
    
#     #Specify save directory for CCD Image
#     filename = "ccd"
#     im_savepath = os.path.join(img_savepath,filename)
#     plt.savefig(im_savepath)
    








'''

this function returns 2 things: 
--- Given 1 specific CCD number, 
find it's Counts-Vector & RESHAPED MATRIX OF CUTOUTS

REQUIRES:
- HDU List
- CCD number
- trained DESOM model to make predictions


-- FOR HELPER: grid_coordinates, map_size, node_map
'''
def count_vector_CCD(hdu_list, idx, desom,
                    grid_coordinates, map_size, node_map, idx_map):
    #Extract data
    X = astroutils.extract_CCD_data(hdu_list, idx)
    
    #Subtract background now
    print("Subtracting background...")
    X_bkg_subbed = prep.sub_background(X)
      
    #Find Deblended Sources using Photutils
    segm_deblended = astroutils.get_non_deblended_sources(X_bkg_subbed,
                                                      nsigma = 3, 
                                                      connected_points=5)
    
    #----------Form Container of Cutouts for this CCD
    #Find cutouts from original image
    cutouts = astroutils.get_cutouts_list(X, segm_deblended)
    
    #Preprocess, IN-PLACE 
    #-- 1) deal with NANS from making Cutouts (in prep script)
    #---2) Use Preprocessing template 
    #Read individual functions for details
    
    #Preprocess using appropriate function
    prep.nlog_preprocess(cutouts)
     
    #Form proper input shape: (num_cutouts x 1024)
    X = astroutils.create_matrix(cutouts)
    print("X dimensions: {}".format(X.shape))
    
    print("Missing Values? : {}".format(np.isnan(X).any()))

    
    #Use trained DESOM model (desom) to get Vector of Predictions
    #Predictions
    y_pred = desom.predict(X)
    
    #count_vec stores the counts for each BMU in range (0,224)
    count_vec = get_bmu_counts(y_pred, grid_coordinates, map_size, node_map, idx_map)
   
    #Recall that X=cutouts  
    return count_vec , X







'''
Takes in an OBS ID.
Downloads file,
obtains HDU list and Launches Algorithm


RETURNS:
1) Matrix of Cutouts (each row flattened)
2) Vector of BMU Counts
'''
def summarize_file(obs_ID,
                   desom, grid_coordinates,
                   node_map, idx_map,
                   num_CCDs,
                   visualize):
    
    
    #Try to download file
    astroutils.download_fits_file(obs_ID)
    
    #Extract file from list and get HDUList
    hdu_list = astroutils.load_hdu_list_for_ID(obs_ID)
    
    size = len(hdu_list) #Size of file
    print(hdu_list.info())
      

    #Extract map size from given model
    map_size = desom.map_size
    
    #----------------pass in this HDUList
    #Create Vector of Correct Size
    ccd_counts = np.zeros(map_size[0]*map_size[1])
    #WILL STORE Cutouts
    main_bin  = np.zeros((1,1024))

    #Start MAIN TIMER
    s1 = timer()
    #Iterate over ALL CCDs  [1----Size]
#     extraction_idx = [x for x in range(1, 2)]
    extraction_idx = list(np.random.choice(np.arange(1,len(hdu_list)), 
                                                           num_CCDs,replace=False)) #How many CCDs to sample
    
    print("Sampled CCDS: ==> {}".format(extraction_idx))
    
    if(visualize):
        print("Visualizing CCDS...")
        #Visualize sampled CCDS
        visualize_CCDS(hdu_list, extraction_idx, obs_ID)
        
    
    #Run Algorithm for each CCD
    for ccd in extraction_idx:
        try:
            #Check for troublesome ones
#             if(ccd == 22):
#                 continue #Move on                            
            #Time 1 CCD
            start = timer()        
            
            #Find Vec for given CCD & Matrix (2D) of cutout data
            counts, cutouts_matrix = count_vector_CCD(hdu_list, ccd, desom,
                                                      #More params
                                                     grid_coordinates, map_size, node_map, idx_map) 
            
            #This is same size as our  final output, so update it
            ccd_counts = np.add(ccd_counts, counts)

            #Vertically stack Master Bin so far ONTO Cutouts Matrix
            main_bin  = np.vstack([main_bin, cutouts_matrix])

            end = timer()
            #Time ONE CCD
            astroutils.time_logger(start,end)
            
            
        #Deal with Error trivially
        except ValueError:
            print("CCD-{} has an issue.".format(ccd))


    #Remove first row
    main_bin = main_bin[1:, :] 
    #Print
    print("BIN DIMENSIONS: {}".format(main_bin.shape))
    #End main timer
    e1 = timer()
    #Total Time taken:
    print("DONE: \n")
    astroutils.time_logger(s1, e1)
    
    return main_bin, ccd_counts, obs_ID


# --------------------------------------------------  Data generation(store cutouts)------------

'''

--- Given 1 specific CCD number, 
find it's  RESHAPED MATRIX OF CUTOUTS

REQUIRES:
- HDU List
- CCD number
'''
def ccd_to_sources(hdu_list, idx):
    #Extract data
    X = astroutils.extract_CCD_data(hdu_list, idx)
    
    #Subtract background now
    print("Subtracting background...")
    X_bkg_subbed = prep.sub_background(X)
      
    #Find Deblended Sources using Photutils
    segm_deblended = astroutils.get_non_deblended_sources(X_bkg_subbed,
                                                      nsigma = 3, 
                                                      connected_points=5)
    
    #----------Form Container of Cutouts for this CCD
    #Find cutouts from original image
    cutouts = astroutils.get_cutouts_list(X, segm_deblended)
    
    #Preprocess, IN-PLACE 
    #-- 1) deal with NANS from making Cutouts (in prep script)
    #---2) Use Preprocessing template 
    #Read individual functions for details
    
    #Preprocess using appropriate function
    prep.nlog_preprocess(cutouts)
     
    #Form proper input shape: (num_cutouts x 1024)
    X = astroutils.create_matrix(cutouts)
    print("X dimensions: {}".format(X.shape))
    
    print("Missing Values? : {}".format(np.isnan(X).any()))

   
    #Recall that X=cutouts  
    return X




'''
Takes in an OBS ID.
-Downloads file,
-obtains HDU list and Launches Algorithm for number of CCDs required

Algorithm:
    - BKG Subtraction/Source Detection
    - Get Bin of processed cutouts
    - c_name --- FOLDER NAME (like c1,c2,...) 
    
RETURNS:
1) Matrix of Cutouts (each row flattened)
'''
def id_to_cutouts_bin(obs_ID, num_CCDs, c_name, visualize = False):
    
    
    #Try to download file
    astroutils.download_single_obs(obs_ID, c_name)
    
    
    #Extract file from list and get HDUList
    hdu_list = astroutils.load_hdu_list_for_ID_validation(obs_ID, c_name)
    
    size = len(hdu_list) #Size of file
    print(hdu_list.info())
        
    #WILL STORE Cutouts
    main_bin  = np.zeros((1,1024))

    #Start MAIN TIMER
    s1 = timer()
    #Iterate over ALL CCDs  [1----Size]
    extraction_idx = list(np.random.choice(np.arange(1,len(hdu_list)), 
                                                           num_CCDs,replace=False)) #How many CCDs to sample
    
    print("Sampled CCDS: ==> {}".format(extraction_idx))
    
    if(visualize):
        print("Visualizing CCDS...")
        #Visualize sampled CCDS
        visualize_CCDS(hdu_list, extraction_idx, obs_ID)
        
    
    #Run Algorithm for each CCD
    for ccd in extraction_idx:
        try:                   
            #Time 1 CCD
            start = timer()        
            
            #Find processed cutouts for CCD
            cutouts_matrix = ccd_to_sources(hdu_list, ccd)  #TODO
            
            #Vertically stack Master Bin so far ONTO Cutouts Matrix
            main_bin  = np.vstack([main_bin, cutouts_matrix])

            end = timer()
            #Time ONE CCD
            print("CCD {} took time: \n".format(ccd))
            astroutils.time_logger(start,end)
            
            
        #Deal with Error trivially
        except ValueError:
            print("CCD-{} has an issue.".format(ccd))


    #Remove first row
    main_bin = main_bin[1:, :] 
    #Print
    print("BIN DIMENSIONS: {}".format(main_bin.shape))
    #End main timer
    e1 = timer()
    #Total Time taken:
    print("DONE: \n")
    astroutils.time_logger(s1, e1)
    
    #Close HDU List
    hdu_list.close()
    #Clean up
    astroutils.remove_single_obs(obs_ID, c_name)
    
    return main_bin




'''
Forms Vector of BMU Counts

REQUIRES:
- X (cutouts from N-CCDs)
- trained DESOM model to make predictions


-- FOR HELPER: grid_coordinates, map_size, node_map
'''
def count_vector_ALL_CCD(X, desom,
                    grid_coordinates, map_size, node_map, idx_map):
    
    
    #Use trained DESOM model (desom) to get Vector of Predictions
    #Predictions
    y_pred = desom.predict(X)
    
    #count_vec stores the counts for each BMU in range (0,625)
    count_vec = get_bmu_counts(y_pred, grid_coordinates, map_size, node_map, idx_map)
   
    #Recall that X=cutouts  
    return count_vec








'''
Plots each CCD from given HDU List

PARAMETERS: 
    - HDU List to extract data from
    - List of CCDS to visualize
    - Observation ID for filename
RETURNS:
    -

'''
def visualize_CCDS(hdu_list, extraction_idx, obs_ID):
    
    
    #Form container of data
    extracted_ims = []
    
    #Extract data for each CCD
    for ccd in extraction_idx:
        X = astroutils.extract_CCD_data(hdu_list, ccd)
        extracted_ims.append(X)
        
    #How many images?
    num_images = len(extracted_ims)
    
    #----------------------------------------------------Setup plot
    fig, ax = plt.subplots(2, num_images, 
                              figsize=(30, 15),
                              squeeze=False)
    
    #For each cutout, do background subtraction appropriately and plot
    for k in range(num_images):
       
        #Extract an image
        X = extracted_ims[k]

        # Create an ImageNormalize object
        norm = ImageNormalize(X, interval= ZScaleInterval(),stretch=SqrtStretch())

        #-----------------------------------------Plot---------------------------
        ax[0][k].imshow(X, cmap='gray', norm = norm)
        ax[0][k].axis('off')
        ax[0][k].set_title('CCD: {}'.format(extraction_idx[k]))
        
        #Plot cropped version
        #Initial cooords
        crop_size = 150
        r = np.random.randint(0, X.shape[0]-crop_size )
        c = np.random.randint(0, X.shape[1]-crop_size)

        
        X_cropped = X[r:r+crop_size,  c:c+crop_size]
        ax[1][k].imshow(X_cropped, cmap='gray', norm = norm)
        ax[1][k].axis('off')
        ax[1][k].set_title('O:({},{})'.format(r,c))
        
        
        
        
    img_savepath = os.path.join(os.getcwd(),'static/plots/')
    print("Saving sampled CCD-images at {}".format(img_savepath))
   
    #Filename
    filename = str(obs_ID) + '_ccds.png'
    #Refer to LOCAL
    im_savepath = os.path.join(img_savepath, filename)   
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    plt.savefig(im_savepath)












# '''
# PARAMETERS:
#     -List of Bins (Fake IDs)
#     - MORE...
# RETURNS:
#     - Matrix of dimensions:   (num_IDs ,(map_size[0]*map_size[1]))


# Basically takes each Bin, finds a Vector of Counts for it,
#         stacks on main_vector_collection
        
# '''
# def create_bin_counts_matrix(bins, desom,
#                              grid_coordinates,
#                            map_size, 
#                            node_map, idx_map):
    
#     #Create Vector of Correct Size
#     predicted_bmu_counts = np.zeros(map_size[0]*map_size[1])
    
#     #Iterate over each bin
#     for X in bins:
#         print("Loaded bin has shape: {}".format(X.shape))
        
#         #Bin is an X-matrix that we can pass into our trained model
#         #Use trained DESOM model (desom) to get Vector of Predictions
#         #Predictions
#         y_pred = desom.predict(X)

#         #count_vec stores the counts for each BMU in range (0, n_nodes-1)
#         count_vec = get_bmu_counts(y_pred, grid_coordinates,
#                                    map_size, 
#                                    node_map, idx_map)
        
#         #print("Formed vector of length: {}".format(count_vec.shape))
#         #print(count_vec)
        
#         #Stack
#         predicted_bmu_counts = np.vstack([predicted_bmu_counts, count_vec])
        
#     #Remove first row
#     predicted_bmu_counts = predicted_bmu_counts[1:, :] 

#     print("Final Matrix of Predicted counts has dimensions: {}".format(predicted_bmu_counts.shape))
    
#     return predicted_bmu_counts




'''
Takes in a list of files, 
reads appropriate file, 
obtains HDU list and Launches Algorithm

RETURNS:
1) Matrix of Cutouts (each row flattened)
2) Vector of BMU Counts
3) Observation ID Number
'''
def summarize_file_old(filenames, idx, desom,
                  grid_coordinates, map_size, node_map, idx_map):
    
    #Extract file from list and get HDUList
    # idx refers to file number in list
    hdu_list = astroutils.load_hdu_list(filenames, idx)
    print(hdu_list.info())
    
    #Get appropriate ID
    loaded_filename   = filenames[idx]
    #Extract ID
    obs_ID  = loaded_filename.split('/')[-1][:len('fits.fz')] #ID number

    #Extract map size from given model
    map_size = desom.map_size
    
    #----------------pass in this HDUList
    #Create Vector of Correct Size
    ccd_counts = np.zeros(map_size[0]*map_size[1])
    #WILL STORE Cutouts
    main_bin  = np.zeros((1,1024))

    #Start MAIN TIMER
    s1 = timer()
    #Iterate over ALL CCDs  [1----36]
    extraction_idx = [x for x in range(1,36)]
    
    #Run Algorithm for each CCD
    for ccd in extraction_idx:
        try:
            #Check for troublesome ones
            if(ccd == 22):
                continue #Move on                            
            #Time 1 CCD
            start = timer()        
            
            #Find Vec for given CCD & Matrix (2D) of cutout data
            counts, cutouts_matrix = count_vector_CCD(hdu_list, ccd, desom,
                                                      #More params
                                                     grid_coordinates, map_size, node_map, idx_map) 
            
            #This is same size as our  final output, so update it
            ccd_counts = np.add(ccd_counts, counts)

            #Vertically stack Master Bin so far ONTO Cutouts Matrix
            main_bin  = np.vstack([main_bin, cutouts_matrix])

            end = timer()
            #Time ONE CCD
            astroutils.time_logger(start,end)
            
            
        #Deal with Error trivially
        except ValueError:
            print("CCD-{} has an issue.".format(ccd))


    #Remove first row
    main_bin = main_bin[1:, :] 
    #Print
    print("BIN DIMENSIONS: {}".format(main_bin.shape))
    #End main timer
    e1 = timer()
    #Total Time taken:
    print("DONE: \n")
    astroutils.time_logger(s1, e1)
    
    return main_bin, ccd_counts, obs_ID

    
'''
Takes an array of BMU counts
Returns indices of locations where count is GREATER THAN Threshold (consider on a Normalized Scale)
'''
def find_max_activated_idx(bmu_counts_vec, threshold):
    
    #Normalize
    bmu_counts_vec = prep.normalize(bmu_counts_vec)
    
    #Find relevant indices of 'most populated' NODES
    activated_idx = np.where(bmu_counts_vec > threshold)[0]
    #Collect
    max_activated_idx = []
    for idx in activated_idx:
        max_activated_idx.append(idx)
        
    print("locations where BMU Count i > {}: ".format(threshold))
    print(max_activated_idx)
    
    return max_activated_idx


#----------------------------------------------------Preprocessing job--------------------------------------
'''
RETURNS : RAW, Processed (Order!!!!)


this function: 
--- Given 1 specific CCD number(idx), 
RETURNS:            it's RESHAPED MATRIX OF CUTOUTS (Raw),
                         RESHAPED MATRIX OF CUTOUTS (Processed)

REQUIRES:
- HDU List
- CCD number
- sub_bkg - if we should subtract background before detecting sources or not
'''
def get_cutout_matrix_from_CCD(hdu_list, idx, 
                               sub_bkg):
       
    #Extract data
    X = astroutils.extract_CCD_data(hdu_list, idx)
    
    #Check if we should subtract background now
    if(sub_bkg):
        print("Subtracting background...")
        X_bkg_subbed = prep.sub_background(X)       
        #Reshaped!
#         X_bkg_subbed = X_bkg_subbed.reshape(w, h)
    #Else, just keep as is
    else:
        X_bkg_subbed = X
        
       
    #Find Deblended Sources using Photutils
    segm_deblended = astroutils.get_deblended_sources(X_bkg_subbed,
                                                      nsigma = 3, 
                                                      connected_points=5)
    
    #----------Form Container of Cutouts for this CCD
    
    #Find cutouts from ORIGINAL X
    cutouts = astroutils.get_cutouts_list(X, segm_deblended)
    
        
    #Form RAW (un-processed copy of data)
    #Form proper input shape: (num_cutouts x 1024)
    X_RAW = astroutils.create_matrix(cutouts)
    print("Formed bin of RAW, Unprocessed Cutouts")
    print("X-RAW dimensions: {}".format(X_RAW.shape))
    
    
    #Preprocess, IN-PLACE 
    #-- 1) deal with NANS from making Cutouts (in prep script)
    #---2) Use Preprocessing template 
    #Read individual functions for details 
    #---------------Preprocess using appropriate function
    prep.nlog_preprocess(cutouts)
              
    #Form X from processed cutouts
    X = astroutils.create_matrix(cutouts)
    print("X dimensions: {}".format(X.shape))
    
    #Recall that X=cutouts which have been PROCESSED  
    return X_RAW, X




'''
Takes in an OBS ID 
reads appropriate file, 
obtains HDU list and Launches Algorithm

RETURNS:
1) Matrix of Cutouts RAW(each row flattened)
1) Matrix of Cutouts PROCESSED (each row flattened)

'''
def cutouts_from_all_CCD(obs_ID, sub_bkg = False):
    
    #Load appropriate file's HDU List
    hdu_list = astroutils.load_hdu_list_for_ID(obs_ID)
    print(hdu_list.info())
        
    #----------------pass in this HDUList
    #WILL STORE Cutouts
    main_bin_RAW  = np.zeros((1,1024))  #Stores RAW
    main_bin  = np.zeros((1,1024))

    #Start MAIN TIMER
    s1 = timer()
    #Iterate over ALL CCDs  [1----36]
    extraction_idx = [x for x in range(1,36)]
    
    #Run Algorithm for each CCD
    for ccd in extraction_idx:
        try:
            #Check for troublesome ones
            problem1 = (ccd == 22) & (obs_ID == "1736831")
            
            if(problem1):
                print("CCD {} of Observation {} has a problem. Skipping... \n".format(ccd, 
                                                                                      obs_ID))
                continue #Move on  
                
                            
            #Time 1 CCD
            start = timer()        
            
            #Find Matrices(2D) of cutout data
            raw_cutouts, cutouts_matrix = get_cutout_matrix_from_CCD(hdu_list, 
                                                                     ccd,
                                                                     sub_bkg)
                              
            #Vertically stack Master Bin so far ONTO Cutouts Matrix
            main_bin_RAW  = np.vstack([main_bin_RAW, raw_cutouts])
            main_bin  = np.vstack([main_bin, cutouts_matrix])

            end = timer()
            #Time ONE CCD
            astroutils.time_logger(start,end)
            
            
        #Deal with Error trivially
        except ValueError:
            print("CCD-{} has an issue.".format(ccd))


    #Remove first row
    main_bin_RAW = main_bin_RAW[1:, :] 
    main_bin = main_bin[1:, :] 
    
    #Print
    print("Raw BIN DIMENSIONS: {}".format(main_bin_RAW.shape))
    print("Processed BIN DIMENSIONS: {}".format(main_bin.shape))
    #End main timer
    e1 = timer()
    #Total Time taken:
    print("DONE: \n")
    astroutils.time_logger(s1, e1)
    
    return main_bin_RAW, main_bin










#---------------------------------------------------Plotting functions---------------------
'''
Expects (map-size x map_size) reshaped array
and filename (ie. 1786883) or sth---STRING
and MAP SIZE
and obs_ID

//Grid elements are NORMALIZED Counts

'''
def plot_counts_heatmap(bmu_counts_vec, map_size, obs_ID):
    global sn_fig_size
    
    #Normalize
    bmu_counts_vec = prep.normalize(bmu_counts_vec)
    
    #Reshape
    counts_grid = np.array(bmu_counts_vec).reshape(map_size[0],map_size[1])
    #Resize into Grid
    M = counts_grid

    plt.figure(figsize = sn_fig_size)
    ax = sn.heatmap(M , linewidths = 0.3, annot =True)

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.title("Distribution of BMU Counts(Normalized) for ID: "+ obs_ID)
    plt.ylabel("X-Node index")
    plt.xlabel("Y-Node index")


    filename = "{}_normalized".format(obs_ID)
    im_savepath = os.path.join(img_savepath,filename)
    plt.savefig(im_savepath)

    
    
    

'''
Expects (map-size x map_size) reshaped array
and filename (ie. 1786883) or sth---STRING
and MAP SIZE
and obs_ID

//Grid elements are PERCENTAGE VALUES
'''
def plot_composition_heatmap(bmu_counts_vec, map_size , obs_ID):
    global sn_fig_size

        
    #Grid of Percentages
    M = ((bmu_counts_vec/np.sum(bmu_counts_vec))*100).reshape(map_size[0],map_size[1])

    plt.figure(figsize = sn_fig_size)
    ax = sn.heatmap(M , linewidths = 0.3, annot =True)

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("Distribution of BMU Counts(Percentage) for ID: "+ obs_ID)
    plt.ylabel("X-Node index")
    plt.xlabel("Y-Node index")

    filename = "{}_percentage".format(obs_ID)
    im_savepath = os.path.join(img_savepath, filename)
    plt.savefig(im_savepath)



   
'''
Given a flat array of SOM idx,
highlight the ones with the HIGHEST counts
obs_ID -- File Observation ID
map_size for formatting
node_map for aiding FAST Lookup
decoded_prototypes for plotting
'''    
    
    
def highlight_most_activated(A,#Array of counts 
                             max_activated_idx, #Indices with most activations
                             obs_ID,
                             map_size,
                             node_map,
                            decoded_prototypes):
    #---------------------Plot
    global fig_size


    #Set up 32x32 size
    img_size =  32    
    #Setup plot
    fig, ax = plt.subplots(map_size[0], map_size[1], figsize= fig_size)  


    #Iterate over each and every prototype
    for k in range(map_size[0]*map_size[1]):      
        #Use dict to find appropriate Node
        node = node_map[k]
        #Extract components
        x = node[0]
        y = node[1]
        
        #Highlight the one we need
        if(k in max_activated_idx):           
            ax[x][y].imshow(decoded_prototypes[k].reshape(img_size, img_size))             
            #---Uncomment if you wanna highlight count
            #count = np.around(A[k],2)
            #ax[x][y].text(0.25,0.25,   str(count),color='red')
            
            ax[x][y].axis('off')            
            continue  #Move on
        
        #Otherwise plot normally
        ax[x][y].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap= 'gray')

        #---Uncomment if you wanna plot counts
        #count = np.around(A[k],2)
        #ax[x][y].text(0.25,0.25, str(count))
        ax[x][y].axis('off')
      
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    #---------Filename
    filename = obs_ID +'_grid.png'
    
    im_savepath = os.path.join(img_savepath,filename)
    plt.savefig(im_savepath)
 
#---------------------------------------------------Background Analysis----------------------------

'''
Loads each and every CCD, measures % difference between MODE and Median for each

Returns a DataFrame with percentage difference values.
'''
def get_background_stats_df(obs_ID):
    
    #Use Astro Utilities to load HDU List
    hdu_list = astroutils.load_hdu_list_for_ID(obs_ID)
    
    #Collection containers
    mode_median = []
    sep_median = []
    
    #Iterate over all CCDS
    for ccd in range(1,36):
        #Extract CCD data for each CCD
        X = astroutils.extract_CCD_data(hdu_list, ccd)
        
        #Obtain percentage difference values
        diff_1 , diff_2 = prep.bkg_diff_values(X)
        
        #Append % difference
        mode_median.append(diff_1)
        sep_median.append(diff_2)
        
    #Create DataFrame
    data = np.array([mode_median, sep_median])
    df = pd.DataFrame({'median_mode' : data[0,:],
                      'sep_median' : data[1,:]}, 
                      index=[x for x in range(1,36)])
    
    #Set index
    print(df.shape)
    
    return df

        
        
    
















'''
Takes in two LOADED bins of cutouts(matrices)
- randomly samples 10 rows from each
- plots a row of images

bin1 --------- No background subtraction
bin2 --------- WITH background subtraction

num_rows---
'''
def plot_original_bkgsub(num_rows, bin1, bin2):
    
    
    #Print shapes
    print(bin1.shape)
    print(bin2.shape)
    
    #Setup plot
    fig, ax = plt.subplots(num_rows, 4, figsize=(25, 20))

    #---Random sample
    #Generate random IDX for sampling from BIN 1
    sample_idx = np.random.randint(bin1.shape[0], 
                                   size = num_rows)
    #Sample 1 
    sample1 = bin1[sample_idx,:]
    #Sample 2
    sample2 = bin2[sample_idx,:]
    
    #Generate random IDX for sampling from BIN 2
#     sample_idx = np.random.randint(bin2.shape[0], 
#                                    size = num_rows)
    
    
    for k in range(num_rows):
        #Extract an image for each (each is a FLAT array)
        X1 = sample1[k]
        X2 = sample2[k]
        
        lwd = 1.2

        #-----------------------------------------Plot---------------------------
        ax[k][0].imshow(X1.reshape(32,32), cmap='gray')
        ax[k][0].axis('off')
        ax[k][0].set_title('NO background subtraction')


        ax[k][1].hist(X1, bins =10,color='green',edgecolor='black', linewidth= lwd)
        ax[k][1].set_title('Histogram(NO background subtraction)')


        ax[k][2].imshow(X2.reshape(32,32),cmap='gray')
        ax[k][2].set_title('Background-subtracted')
        
        ax[k][3].hist(X2, bins =10,color='yellow',edgecolor='black', linewidth= lwd)
        ax[k][3].set_title('Histogram(background subtraction)')
        
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        #Filename
        filename = 'sources.png'
        img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/histograms/')
        
        #Refer to LOCAL but just use local im_savepath
        im_savepath = os.path.join(img_savepath, filename)   
        plt.savefig(im_savepath)

        

'''
Takes in an ID (it should've been downloaded as a fits file already in download directory

- loads full Image
- Tries different Background estimation methods
-plots results side by side
'''
def background_estimation_plots(obs_ID, ccd):
    
    #Use Astro Utilities to load
    hdu_list = astroutils.load_hdu_list_for_ID(obs_ID)
    #Extract CCD data for CCD 3
    X = astroutils.extract_CCD_data(hdu_list, ccd)
    
    #Store  dimensions for  reshaping later
    w = X.shape[0]
    h = X.shape[1]

    
    #Setup plot
    fig, ax = plt.subplots(2, 4, figsize=(25, 20))

    #Different background estimation methods
    
    #1) Median as BKG
    X_med = prep.sub_background(X)
    
    #2) Equation
    X_eqn = prep.sub_background_eqn(X)
    
    #3) SEP
    X_sep = prep.sub_background_sep(X)
 
    lwd = 1.2
    
    #Setup image normalizers
    # Create an ImageNormalize object
    norm = ImageNormalize(X, interval= ZScaleInterval(),stretch=SqrtStretch())
    
    norm_med = ImageNormalize(X_med, interval= ZScaleInterval(),stretch=SqrtStretch())
    norm_eqn = ImageNormalize(X_eqn, interval= ZScaleInterval(),stretch=SqrtStretch())
    norm_sep = ImageNormalize(X_sep, interval= ZScaleInterval(),stretch=SqrtStretch())

    

    #-----------------------------------------Plot---------------------------
    
    k = 0 #For images
    
    ax[k][0].imshow(X, cmap='gray', norm = norm)
    ax[k][0].axis('off')
    ax[k][0].set_title('Original Image')

    ax[k][1].imshow(X_med.reshape(w,h),cmap='gray',norm = norm)
    ax[k][1].set_title('Median Subtraction')


    ax[k][2].imshow(X_eqn.reshape(w,h),cmap='gray',norm = norm)
    ax[k][2].set_title('Equation Estimation')

    ax[k][3].imshow(X_sep,cmap='gray',norm = norm) #No reshape necessary
    ax[k][3].set_title('SEP Estimation')
    
    #--------------------------------Histograms---------------
    k = 1 #For Histograms
    #Flatten as you pass in!
    
#     ax[k][0].hist(X.flatten(), bins =10,color='gray',edgecolor='black', linewidth= lwd)
#     ax[k][0].axis('off')
#     ax[k][0].set_title('Original Image')

#     ax[k][1].hist(X_med, bins =10,edgecolor='black', linewidth= lwd)
#     ax[k][1].set_title('Median Subtraction')


#     ax[k][2].hist(X_eqn, bins =10,edgecolor='black', linewidth= lwd)
#     ax[k][2].set_title('Equation Estimation')

#     ax[k][3].hist(X_sep.flatten(), bins =10,color='blue',edgecolor='black', linewidth= lwd)
#     ax[k][3].set_title('SEP Estimation')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    #Filename
    filename = 'bkg_estimation.png'
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/histograms/')

    #Refer to LOCAL but just use local im_savepath
    im_savepath = os.path.join(img_savepath, filename)   
    plt.savefig(im_savepath)



'''
Takes in an ID (it should've been downloaded as a fits file already in download directory)
--and takes CCD to load


-- num_images means number of columns


- loads full Image
- Tries different Background estimation methods on randomly sampled cutouts
- plots results side by side
'''
def background_estimation_plots_rows(obs_ID, ccd,  num_images):
    
    #Use Astro Utilities to load
    hdu_list = astroutils.load_hdu_list_for_ID(obs_ID)
    #Extract CCD data for CCD 3
    X = astroutils.extract_CCD_data(hdu_list, ccd)
    
    #Form cutouts Matrix
    #Find Deblended Sources using Photutils
    segm_deblended = astroutils.get_deblended_sources(X, nsigma = 3, 
                                                       connected_points=5)
    
    #----------Form Container of Cutouts for this CCD
    #Find cutouts
    cutouts = astroutils.get_cutouts_list(X, segm_deblended)
        
    #Form proper input shape: (num_cutouts x 1024)
    X = astroutils.create_matrix(cutouts)
    print("BIN dimensions: {}".format(X.shape))
    
    #---Randomly sample cutouts from BIN
    #Generate random IDX for sampling from BIN 1
    sample_idx = np.random.randint(X.shape[0], 
                                   size = num_images)
    #Sample of num images 
    sample = X[sample_idx,:]
  
    #Store  dimensions for  reshaping later
    w = 32
    h = 32
  
    #----------------------------------------------------Setup plot
    fig, ax = plt.subplots(4, num_images, figsize=(30, 15))
    
    #For each cutout, do background subtraction appropriately and plot
    for k in range(num_images):
    
    
        #Extract an image
        X = sample[k].reshape(32,32)

        #Different background estimation methods
        #1) Median as BKG
        X_med = prep.sub_background(X)

        #2) Equation
        X_eqn = prep.sub_background_eqn(X)

        #3) SEP
        X_sep = prep.sub_background_sep(X)

        lwd = 1.2
        #Setup image normalizers
        # Create an ImageNormalize object
        norm = ImageNormalize(X, interval= ZScaleInterval(),stretch=SqrtStretch())

        #-----------------------------------------Plot---------------------------
        ax[0][k].imshow(X, cmap='gray', norm = norm)
        ax[0][k].axis('off')
        ax[0][k].set_title('Original Image')

        ax[1][k].imshow(X_med.reshape(w,h),cmap='gray',norm = norm)
        ax[1][k].set_title('Median Subtraction')


        ax[2][k].imshow(X_eqn.reshape(w,h),cmap='gray',norm = norm)
        ax[2][k].set_title('Equation Estimation')

        ax[3][k].imshow(X_sep,cmap='gray',norm = norm) #No reshape necessary
        ax[3][k].set_title('SEP Estimation')


    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    #Filename
    filename = 'cutouts_bkg.png'
    img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/histograms/')

    #Refer to LOCAL but just use local im_savepath
    im_savepath = os.path.join(img_savepath, filename)   
    plt.savefig(im_savepath)





'''
TOY Data Creation Script


Takes in a list of files, 
Downloads appropriate file,
reads appropriate file, 
obtains HDU list and Launches Algorithm

RETURNS:

--Accumulated BMU Count Vectors for each CCD, in each file as VALUES of dict
-- KEYS are File IDS


RETURNED ARRAY has shape: (#IDs, #CCDs, 225)

    
'''
def accumulate_CCD_vecs(files_to_download, 
                        desom,
                        grid_coordinates, map_size, node_map, idx_map):
    
    main_bin_collection = {}
    
    #Download each file
    for obs_ID in files_to_download:
        #OBS ID?
        obs_ID = str(obs_ID)
        
        #Setup download URL
        url = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/"+obs_ID+"p.fits.fz"
        print(url)
        
        #--Run download
        os.system(('cd CADC_downloads/ && '+ 'curl -O -J -L '+url))
    
    
    
        #Extract file for this ID from list and get HDUList
        # idx refers to file number in list
        hdu_list = astroutils.load_hdu_list_for_ID(obs_ID)
        print(hdu_list.info())
        
    
        #Extract map size from given model
        map_size = desom.map_size
        
        #----------------pass in this HDUList
        #Create Vector of Correct Size
        ccd_counts = np.zeros(map_size[0]*map_size[1])
        #WILL STORE Vectors
        main_bin  = np.zeros((1,map_size[0]*map_size[1]))
    
        #Start MAIN TIMER
        s1 = timer()
        #Iterate over ALL CCDs  [1----36]
        extraction_idx = [x for x in range(1, 6)]
        
        #Run Algorithm for each CCD
        for ccd in extraction_idx:
            try:
                #Check for troublesome ones
                if(ccd == 22):
                    continue #Move on                            
                #Time 1 CCD
                start = timer()        
                
                #Find Vec for given CCD & Matrix (2D) of cutout data
                counts, cutouts_matrix = count_vector_CCD(hdu_list, ccd, desom,
                                                          #More params
                                                         grid_coordinates, map_size, node_map, idx_map) 
                
                #This is same size as our  final output, so update it
                #ccd_counts = np.add(ccd_counts, counts)
    
                #Vertically stack Master Bin so far ONTO Cutouts Matrix
                main_bin  = np.vstack([main_bin, counts])
                print("Added new row of Vector for ccd {}".format(ccd))
    
                end = timer()
                #Time ONE CCD
                astroutils.time_logger(start,end)
                
                
            #Deal with Error trivially
            except ValueError:
                print("CCD-{} has an issue.".format(ccd))
    
    
        #Remove first row
        main_bin = main_bin[1:, :] 
        #Print
        print("Main Bin shape: {}".format(main_bin.shape))
        
        
        #Add to collection as the value for this KEY (OBS-ID)
        main_bin_collection[obs_ID] =  main_bin
        print("Added to collection...")
        
        #End main timer
        e1 = timer()
        #Total Time taken:
        print("DONE: \n")
        astroutils.time_logger(s1, e1)
    
    return main_bin_collection



































