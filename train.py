#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:18:11 2017

@author: kyleguan
"""

import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

from features import *
        

    
if __name__ =='__main__':
   
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True# Histogram features on or off
    hog_feat = True # HOG features on or off
    
    
    # Read in cars and notcars
    images = glob.glob('data/*/*/*.*')
    cars = []
    notcars = []
    # For example, car images are stored in data/vehicels/GTI_LEFT/*.png
    for image in images:
        if 'non-vehicles' in image: # differentiate car and non-car by file name
            notcars.append(image)
        else:
            cars.append(image)
    
    # Extract features for cars and non-cars
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # Use MLPClassifier
    mlp=MLPClassifier(random_state=54321)
    t=time.time()
    
    mlp.fit(X_train, y_train)
    
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train MLP ...')
    # Check the score of the MLPClassifier
    print('Test Accuracy of MLP Classifier = ', round(mlp.score(X_test, y_test), 4))
    
    
    # Save the 
    file_name_mlp = 'mlp_'+'spa_'+str(spatial_size[0])+'_hist'+str(hist_bins)\
                          +'_'+color_space+'_o'+str(orient)+'_p'+str(pix_per_cell)\
                          +'_c'+str(cell_per_block)\
                          +'_ch'+str(hog_channel)+'.p' 
    file_name_scaler = 'mlp_scaler_'+'spa_'+str(spatial_size[0])+'_hist'+str(hist_bins)\
                          +'_'+color_space+'_o'+str(orient)+'_p'+str(pix_per_cell)\
                          +'_c'+str(cell_per_block)\
                          +'_ch'+str(hog_channel)+'.p' 
    
    with open(file_name_mlp, mode='wb') as f:
         pickle.dump(mlp, f)
         
         
    with open(file_name_scaler, mode='wb') as f:
         pickle.dump(X_scaler, f)
    

     


