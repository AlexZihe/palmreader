import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import skvideo.io
import gc
import h5py
from collections import defaultdict
import cv2
from scipy.ndimage import gaussian_filter1d

def cal_distance_(label, bodypart = 'centroid'):
    '''helper function for "calculate distance traveled'''
    x = gaussian_filter1d(label[bodypart]['x'].values, 3)
    y = gaussian_filter1d(label[bodypart]['y'].values, 3)
    d_x = np.diff(x)
    d_y = np.diff(y)
    d_location = np.sqrt(d_x**2 + d_y**2)
    return d_location

def four_point_transform(image, tx,ty, cx,cy, wid, length):
    '''
    helper function for center and align a single video frame
    input:
        T, coord of tailbase, which is used to center the mouse
        TN, vector from tailbase to centroid
        wid, the width of the to-be-cropped portion
        length, the length of the to-be-cropped portion

    output:
        warped: the cropped portion in the size of (wid, length), mouse will be centered by tailbase, aligned by the direction from tailbase to centroid

    '''
    T = np.array([tx,ty])
    N = np.array([cx,cy])
    TN = N - T

    uTN = TN / np.linalg.norm(TN) # calculate the unit vector for TN

    # calculate the unit vector perpendicular to uTN
    uAB = np.zeros((1,2),dtype = "float32")
    uAB[0][0] = uTN[1]
    uAB[0][1] = -uTN[0]

    # calculate four corners of the to-be-cropped portion of the image
    #   use centroid to center the mouse
    A = N + uAB * (wid/2) + uTN * (length/2)
    B = N - uAB * (wid/2) + uTN * (length/2)
    C = N - uAB * (wid/2) - uTN * (length/2)
    D = N + uAB * (wid/2) - uTN * (length/2)

    # concatenate four corners into a np.array
    pts = np.concatenate((A,B,C,D))
    pts = pts.astype('float32')

    # generate the corresponding four corners in the cropped image
    dst = np.float32([[0,0],[wid,0],[wid,length],[0,length]])

    # generate transform matrix
    M = cv2.getPerspectiveTransform(pts,dst)

    # rotate and crop image
    warped = cv2.warpPerspective(image,M,(wid,length))

    return warped

def cal_paw_luminance(label, ftir_video, size = 22):
    '''
    helper function for extracting the paw luminance signals of both hind paws from the ftir video

    input:
    label: DLC tracking of the recording
    ftir_video: ftir video of the recording
    size: size of the cropping window centered on a paw
    output:
    hind_left: paw luminance of the left hind paw
    hind_right: paw luminance of the right hind paw
    '''

    num_of_frames = ftir_video.shape[0]
    # print(f'video length is {num_of_frames/25/60} mins')

    # right hind paw
    hind_right = []
    for i in tqdm(range(num_of_frames)):
        x,y = (int(label['hrpaw'][['x']].values[i]),int(label['hrpaw'][['y']].values[i]))
        hind_right.append(np.mean(ftir_video[i][y-size:y+size,x-size:x+size]))

    # left hind paw
    hind_left = []
    for i in tqdm(range(num_of_frames)):
        x,y = (int(label['hlpaw'][['x']].values[i]),int(label['hlpaw'][['y']].values[i]))
        hind_left.append(np.mean(ftir_video[i][y-size:y+size,x-size:x+size]))
    
    # release space from RAM after the analysis has finished
    del ftir_video
    gc.collect()

    return hind_left, hind_right

def scale_ftir(hind_left, hind_right):
    '''helper function for doing min 95-quntile scaler
       for individual recording, pool left paw and right paw ftir readings and find min and 95 percentile; then use those values to scale the readings'''


    left_paw = np.array(hind_left)
    right_paw = np.array(hind_right)

    min_ = min(np.nanmin(left_paw), np.nanmin(right_paw))
    max_ = max(np.nanmax(left_paw), np.nanmax(right_paw))
    quantile_ = np.nanquantile(np.concatenate([left_paw,right_paw]),.95)

    left_paw = (left_paw - min_) / (quantile_- min_)
    right_paw = (right_paw - min_) / (quantile_- min_)

    # replace all nan values with the mean, the nan values comes from DLC not tracking properly for those timepoints
    left_paw_mean = np.nanmean(left_paw)
    right_paw_mean = np.nanmean(right_paw)
    left_paw = np.nan_to_num(left_paw, nan = left_paw_mean)
    right_paw = np.nan_to_num(right_paw, nan = right_paw_mean)

    return (left_paw,right_paw)
