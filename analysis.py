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

from utils import *

def main():
    features = {}

    # read DLC tracking
    df = pd.read_hdf('A1_bodyDLC_resnet50_palmreader-500Mar25shuffle1_500000.h5')
    label = df['DLC_resnet50_palmreader-500Mar25shuffle1_500000']

    # calculate distance traveled
    features['distance_traveled'] = np.nansum(cal_distance_(label)).reshape(-1,1)

    #----center and align body_pose video----

    # load the body_pose video
    # note that this step will need 32GiB of RAM for a 30min recording
    body_video = skvideo.io.vread('A1_body.avi')[:,:,:,0]
    frame_num = body_video.shape[0]
    frame_ht = body_video.shape[1]
    frame_wd = body_video.shape[2]

    # load the DLC tracking of tailbase and centroid
    tailbase_coords = label['tailbase'][['x','y']].values
    centroid_coords = label['centroid'][['x','y']].values

    # smoothening the tracking of tailbase and centorid with 1d gaussian filter
    tailbase_coords_smooth = np.zeros_like(tailbase_coords)
    centroid_coords_smooth = np.zeros_like(centroid_coords)

    sig = 3

    tailbase_coords_smooth[:,0] = gaussian_filter1d(tailbase_coords[:,0], sig, mode='nearest')
    tailbase_coords_smooth[:,1] = gaussian_filter1d(tailbase_coords[:,1], sig, mode='nearest')

    centroid_coords_smooth[:,0] = gaussian_filter1d(centroid_coords[:,0], sig, mode='nearest')
    centroid_coords_smooth[:,1] = gaussian_filter1d(centroid_coords[:,1], sig, mode='nearest')

    # center and align the body video
    for i in tqdm(range(frame_num)):
        body_video[i] = four_point_transform(body_video[i],
                                        tailbase_coords_smooth[i,0],
                                        tailbase_coords_smooth[i,1],
                                        centroid_coords_smooth[i,0],
                                        centroid_coords_smooth[i,1],
                                        frame_ht,
                                        frame_wd)

    # save the processed body video
    writer = skvideo.io.FFmpegWriter('centered_aligned_body.avi', outputdict={
                '-vcodec': 'mjpeg','-qscale': '1', '-b': '300000000', '-pix_fmt': 'yuv420p','-r':'25' })

    for i in tqdm(range(np.asarray(body_video).shape[0])):
        writer.writeFrame(body_video[i])
    writer.close()

    # release space from RAM after the center_align_video has been saved to hard drive
    del body_video
    gc.collect()
    #-------------------------------------------------------------

    #----calculate paw luminance, average paw luminance ratio, and paw luminance log-ratio----
    # read ftir video
    ftir_video = skvideo.io.vread('A1_ftir.avi')[:,:,:,0]

    # calculate paw luminance
    hind_left, hind_right = cal_paw_luminance(label, ftir_video, size = 22)

    features['hind_left_luminance'] = hind_left
    features['hind_right_luminance'] = hind_right

    hind_left_scaled, hind_right_scaled = scale_ftir(hind_left, hind_right)
    features['hind_left_luminance_scaled'] = hind_left_scaled
    features['hind_right_luminance_scaled'] = hind_right_scaled

    # calculate luminance logratio
    features['average_luminance_ratio'] = np.nansum(features['hind_left_luminance']) / np.nansum(features['hind_right_luminance']).reshape(-1,1)
    features['luminance_logratio'] = np.log((features['hind_left_luminance_scaled']+1e-4)/(features['hind_right_luminance_scaled']+1e-4))
    #-------------------------------------------------------------

    # save extracted features
    with h5py.File('features.h5', 'w') as hdf:
        video = hdf.create_group('A1')
        for key in features.keys():
            video.create_dataset(key, data = features[key])

    return

if __name__ == "__main__":
    main()
