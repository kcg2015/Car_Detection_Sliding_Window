#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:18:11 2017

@author: kyleguan
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
#from load_and_classify_heat_v1 import *
from features import *
from moviepy.editor import VideoFileClip

QUEUE_LEN=10
from collections import deque

class Box:
    def __init__(self):
        # No. of cars detected in the last frame
        self.last_num_detected = None
        # Keep track no.of cars detected in the most recent QUEUE_LEN frames
        self.num_detected = deque(maxlen = QUEUE_LEN)
        
        # Remember boxes
        self.boxes = deque(maxlen = QUEUE_LEN)
        
        self.labels =deque(maxlen = QUEUE_LEN)
        
        self.last_labels=None
        
        
        # Count the number of frames processed
        self.count = 0


def pipeline(image):
    
    
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)
    hist_bins = 32   
    spatial_feat = True # Spatial features on or off
    hist_feat = True# Histogram features on or off
    hog_feat = True # HOG features on or off
    #mlp=MLPClassifier(random_state=54321)
    every_no_frames=1 # Have a full detection every few frames
                      # Adjust this parameter to reduce runtime
    file_name_clf='mlp_spa_32_hist32_YCrCb_o9_p8_c2_chALL.p'
    with open(file_name_clf, 'rb') as f:
         mlp=pickle.load(f)
    
    file_name_scaler='mlp_scaler_spa_32_hist32_YCrCb_o9_p8_c2_chALL.p'
    with open(file_name_scaler, 'rb') as f:
         X_scaler=pickle.load(f)     
  
    window_img=np.copy(image)
    draw_img=np.copy(image)
    
    #Python dictionary data structure to store the specifications for 
    #each window (size, search range in x an y directions, and
    #colors of the boxes):
    multi_window={}
    multi_window['max']=[(256, 256),[200, None],[400,720],(255,0,0)]
    multi_window['min']=[(48, 48),[200, None],[400,500],  (0, 255,0)]
    multi_window['mid1']=[(128, 128),[200, None],[400,720],(0,0,255)]
    multi_window['mid2']=[(96, 96),[200, None],[400,720],(255,255,0)]
   
    image = image.astype(np.float32)/255
    
    if car.count%every_no_frames==0: # Full processing
        bbox_list=[]
        for key in multi_window: # Loop through different window sizes
           xy_window=multi_window[key][0]
           x_start_stop = multi_window[key][1]
           y_start_stop = multi_window[key][2]
           color=multi_window[key][3]
           windows = slide_window(image, x_start_stop= x_start_stop, 
                                  y_start_stop=y_start_stop, 
                                  xy_window=xy_window, 
                                  xy_overlap=(0.8, 0.65))
          
           hot_windows = search_windows(image, windows, mlp, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)                       
          
           bbox_list.append(hot_windows)
           window_img = draw_boxes(window_img, hot_windows, color=color, thick=3)                    
           
        bbox_list=filter(None, bbox_list)
        
        if len(bbox_list)>0:
            bbox_list = np.concatenate(bbox_list)
            heatmap = np.zeros_like(window_img[:,:,0]).astype(np.float)
            heatmap = add_heat(heatmap, bbox_list)
            heatmap = apply_threshold(heatmap, 2)
            labels = label(heatmap)
            draw_img, draw_box_list = draw_labeled_bboxes(draw_img, labels)
       
            car.boxes.append(draw_box_list)
            car.labels.append(labels)
            car.last_labels=labels
            car.last_num_detected = labels[1]
            car.num_detected.append(labels[1])
        
        else:
            
            car.last_num_detected = 0
            car.num_detected.append(0)
            
    else:  
        if (car.last_num_detected >0) & (car.last_labels is not None):
           labels = car.last_labels
           draw_img, bbox_list = draw_labeled_bboxes(draw_img, labels)
    
    car.count+=1
    return draw_img
    
    
if __name__=='__main__':    
    
    car=Box()    
    
    images = glob.glob(test_images/test*.jpg')  
    #images = glob.glob('start/frame*.jpg')  
    
    t=time.time()
    # Uncomment this part to test on a sequence of images.
    #for file_name in images[2:12]:
#        print file_name
#        image = mpimg.imread(file_name)
#        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#    
#        overlay = pipeline(image)
#        f, (ax1, ax2)=plt.subplots(1,2,figsize=(10,4))
#        ax1.imshow(image)
#        ax2.imshow(overlay)
    #t2 = time.time()
    #
    #print(round(t2-t, 2), 'Seconds to finish')
#
    output = 'test_full_sca5_b.mp4'
    clip1 = VideoFileClip("project_video.mp4")#.subclip(0,6) # The first 8 seconds doesn't have any cars...
    clip = clip1.fl_image(pipeline)
    clip.write_videofile(output, audio=False)
    
    t2 = time.time()
    
    print(round(t2-t, 2), 'Seconds to finish')