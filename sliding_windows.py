from __future__ import division

import sys
sys.path.append('../imtools/')
from imtools import tools

import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as skiexp
import skimage.feature as skifea
import scipy.stats as scista
import skimage.transform as skitra

import cv2

import copy
import math
import os


def run(image, mask, pyr_scale=2, min_pyr_size=20):
    for i, (res_gaus, res_lap) in enumerate(zip(skitra.pyramid_gaussian(image, downscale=pyr_scale), skitra.pyramid_laplacian(image, downscale=pyr_scale))):
        # if the image is too small, break from the loop
        if res_gaus.shape[0] < min_pyr_size or res_gaus.shape[1] < min_pyr_size:
            break

        plt.figure()
        plt.subplot(121), plt.imshow(res_gaus, 'gray', interpolation='nearest'), plt.title('layer %i' % (i + 1))
        plt.subplot(122), plt.imshow(res_lap, 'gray', interpolation='nearest')

    plt.show()

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    run(data_s, mask_s)