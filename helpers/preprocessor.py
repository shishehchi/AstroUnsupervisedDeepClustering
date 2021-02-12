#Preprocessing Utility Functions

#Imports
import numpy as np
import os
import matplotlib.pyplot as plt

#SEP
import sep



#--------------------------------------------Specify paths
current_path = os.getcwd()
img_savepath = os.path.join(current_path,'results/tmp/pipeline_plots/histograms/')







#Subtract Mean
def zero_mean_data(dat):
    return dat - np.mean(dat, axis= 0)

#Standardize data
def standardize(dat):
    #Subtract mean of each row
    mean_p , std_p = np.mean(dat, axis = 0), np.std(dat)
    #Standardize
    return (dat - mean_p)/std_p

#Normalize data [0,1]
def normalize(dat):
    dat_min , dat_max = np.min(dat), np.max(dat)
    return (dat -dat_min)/(dat_max - dat_min)

#Log-Transform Data
def log_trans(dat):
    dat = np.log(dat + 1e-5) #Add small positive value
    return dat


'''
Pass in a CUTOUT DATA MATRIX(shape: numCutouts x 1024)
and impute nans with MEDIAN

RETURNS Matrix Data
'''
def impute_with_median(X):
    #Find median
    median = np.nanmedian(X)
    #Locate nan
    X[np.where(np.isnan(X))] = median
    
    return X




'''
Takes in a list of cutouts and processes EACH IN-PLACE

--IMPUTE WITH MEDIAN if NaNs present

'''
def preprocess(cutouts):
    #Iterate over Cutouts and:
    # - Log transform first
    # - Normalize
    for cutout in cutouts:
        
        #Impute with MEDIAN, for non-overlapping values(aka.nans)
        cutout.data = impute_with_median(cutout.data)
        
        #Log transform
        cutout.data = log_trans(cutout.data)
        #Normalize
        cutout.data = normalize(cutout.data)

        
'''
MAIN PREPROCESSING Function

Takes in a list of cutouts and processes EACH IN-PLACE

-- Impute NaNs with MEDIAN
-- Convert negatives to 1
-- Take Natural Logarithm
--  NORMALIZE

'''
def nlog_preprocess(cutouts):
    print("Preprocessing all given cutouts: (NaN imputation)=>(Background-subtraction)=>(Log Transform)=>(Normalization)")
    #Iterate over Cutouts and:
    # - Subtract Median
    # - Scale up
    # - Log 10
    # - Normalize
    for cutout in cutouts:
        
        #Impute with MEDIAN, for non-overlapping values(aka.nans)
        #print("Imputing NaNs with Median...")
        cutout.data = impute_with_median(cutout.data)
        
        #Subtract background again from cutout!
        #print("Subtracting background again from cutout...")        
        cutout.data = sub_background(cutout.data)
            
        #Log Transform
        #Must turn any negative values to 1 within this function!
        #print("Appling Natural Log Transformation...")        
        cutout.data = nlog_transform(cutout.data)
        
        #Normalize
        #print("Normalizing...")        
        cutout.data = normalize(cutout.data)
        



'''
Takes in a list of cutouts and processes EACH IN-PLACE, with rough Background estimation
'''
def preprocess_bkg(cutouts):
    #Iterate over Cutouts and:
    # - Subtract Median
    # - Scale up
    #-  Log 10
    for cutout in cutouts:
        
        #Impute with MEDIAN, for non-overlapping values(aka.nans)
        cutout.data = impute_with_median(cutout.data)
        
        #Background Normalize
        cutout.data = background_normalize(cutout.data)
        
        
        
'''
Pass in Matrix (2D)

-Imputes data with 1 where it is negative

'''
def background_normalize(dat):    
#     #find Median
#     median_val = np.median(dat.flatten())
    #Subtract median
    dat = sub_background(dat)

    #Check if we need to scale
    dat[np.where(dat < 1)] = 1

    #Log Transform
    dat = np.log10(dat)
    
    return dat


#-----------------------------------------------Background Estimation----------------------------
'''
-- Main Background Estimation function
--Pass in entire image as 2D(dat), before running Photutils Source Detection
-- Compute two estimates of BKG (Median and Mode)
-- Checks if difference is large enough to disregard MODE



Simply return the background subtracted image (RESHAPED as 2D)

'''

def sub_background(dat):
    
    #Store original dimensions
    w = dat.shape[0]
    h = dat.shape[1]
    
    #Flatten dat
    dat = dat.flatten()
    #Store stats
    mean, median = np.mean(dat), np.median(dat)
    
    #Background Estimates
    mode = 3 * median - 2 * mean     
    #Find percentage difference
    percent_diff = 100 * np.absolute(mode-median)/(median)
    
    #Decide on appropriate bkg_estimate
    if(percent_diff > 30):
        #print("Using Median as BKG Estimate.")
        #Simply use Median
        bkg_estimate = median
    else:
        #print("Using MODE as BKG Estimate.")
        bkg_estimate = mode

    #Subtract from Image, in-place
    dat = dat - bkg_estimate  
    
    #Reshape into original 2D form
    dat = dat.reshape(w,h)
    
    return dat




'''
Pass in entire image as 2D(dat), before running Photutils Source Detection

--NOT USING Sep

--Background Estimate: MEDIAN


Simply return the background subtracted image FLAT
'''

def sub_background_eqn(dat): 
    #Flatten
    dat = dat.flatten()
    
    #Store stats
    mean, median = np.mean(dat), np.median(dat)
    #Background Estimate
    bkg_estimate = 3*median - 2*mean
     
    #Subtract from Image, in-place
    dat = dat - bkg_estimate 
    
    return dat



'''
Pass in entire image as 2D(dat), before running Photutils Source Detection

--USING Sep

--Background Estimate: SEP


Simply return the background subtracted image FLAT
'''

def sub_background_sep(dat): 
   
    #Use 2D Data without flattening
    #Find background using SEP
    bkg = sep.Background(dat)       
    # evaluate background as 2-d array, same size as original image
    bkg_estimate = bkg.back()
     
    #Subtract from Image, in-place
    dat = dat - bkg_estimate 
    
    return dat



'''
Pass in entire image as 2D(dat), before running Photutils Source Detection
-- Find 2 estimates for  background
-- Compare and find percentage difference
-- Return TRUE if percentage_diff > 30 %

'''
def is_diff_large(dat):
    
    #Flatten dat
    dat = dat.flatten()
    #Store stats
    mean, median = np.mean(dat), np.median(dat)
    #Background Estimates
    mode = 3 * median - 2 * mean 
    
    #Find percentage difference
    percent_diff = 100* np.absolute(mode-median)/(median)
    
    print("Percentage difference between {} and {}: {} %".format(mode,median,np.around(percent_diff,2)))
    print("Difference greater than 30%?-- {}".format(percent_diff > 30))
    
    return percent_diff > 30


'''
Pass in entire image as 2D(dat), before running Photutils Source Detection
-- Find 3 estimates for  background
-- Find % differences
-- Return both values

'''
def bkg_diff_values(dat):
    
    #Original Dimensions
    w = dat.shape[0]
    h = dat.shape[1]
 
    #Flatten dat
    dat = dat.flatten()
    #Store stats
    mean, median = np.mean(dat), np.median(dat)
    
    #Background Estimates
    mode = 3 * median - 2 * mean  #(Equation)
    
    bkg_sep = sep.Background(dat.reshape(w, h)).globalback #(SEP BACK MEAN back directly)
    
    #Find percentage difference
    diff_1 = np.around(100* np.absolute(mode-median)/(median), 2)
    #Find % diff between SEP and Median
    diff_2 = np.around(100* np.absolute(bkg_sep-median)/(median), 2)
  
    return diff_1, diff_2












#-------------------------------------------------- Transformations-------------------------------------





'''
Pass in flat array of floats (1024)

-Impute negatives with 1
-Apply log10 
'''

def log10_transform(dat):
    
    #Check if we need to impute
    dat[np.where(dat < 1)] = 1    
    #Apply log10 transform
    dat = np.log10(dat)
    
    return dat


'''
Pass in flat array of floats (1024)

-Impute negatives with 1
-Apply Natural log 
'''

def nlog_transform(dat):
    
    #Check if we need to impute
    dat[np.where(dat < 1)] = 1    
    #Apply log transform
    dat = np.log(dat)
    
    return dat



'''
Specify number of Rows you want.
-- pass in a MATRIX (Bin) of cutouts (each is flattened as it comes in)
-- by this point, cutouts have been background-subtracted

-- simply preprocess them as you see fit, and observe the resulting distributions
'''
def plot_distributions(num_rows, cutouts):

    #Setup plot
    fig, ax = plt.subplots(num_rows, 6, figsize=(20, 20))

    #---Random sample
    #Generate random IDX for sampling
    sample_idx = np.random.randint(cutouts.shape[0], 
                                   size = num_rows)
    #Sample
    sample = cutouts[sample_idx,:]

    #Iterate over each and every prototype
    for k in range(num_rows):
        #Extract image
        X = sample[k]
        X_0 = X #Original

        
       #Log-trans
        Xn = nlog_transform(X)
        #Log-trans
        X10 = log10_transform(X)
        
        
       #Normalize
        #X = normalize(X)
        lwd = 1.2


        #-----------------------------------------Plot---------------------------
        ax[k][0].imshow(X_0.reshape(32,32), cmap='gray')
        ax[k][0].axis('off')
        ax[k][0].set_title('Original Image')


        ax[k][1].hist(X_0, bins =10,color='gray',edgecolor='black', linewidth= lwd)
        ax[k][1].set_title('Original Histogram')


        ax[k][2].hist(X10, bins =10, color='red',edgecolor='black', linewidth= lwd)
        ax[k][2].set_title('Log-10')
        
        ax[k][3].hist(normalize(X10), bins =10, color='red',edgecolor='black', linewidth= lwd)
        ax[k][3].set_title('NORMALIZE[Log-10]')


        ax[k][4].hist(Xn, bins =10, color='green',edgecolor='black', linewidth= lwd)
        ax[k][4].set_title('Log-e')
        
        ax[k][5].hist(normalize(Xn), bins =10, color='green',edgecolor='black', linewidth= lwd)
        ax[k][5].set_title('NORMALIZE[Log-e]')


    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    #Filename
    filename = 'transformations.png'
    #Refer to global but just use local im_savepath
    im_savepath = os.path.join(img_savepath, filename)   
    plt.savefig(im_savepath)
  
    

    
 










    
    
    
    
    
    
    
    
    
    
    
    