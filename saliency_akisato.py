from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append('../imtools/')
from imtools import tools
import skimage.exposure as skiexp

import os
import sys

sys.path.append('../pySaliencyMap/')
import pySaliencyMap


def run(image, mask=None, smoothing=False, show=False, show_now=True):
    if mask is None:
        mask = np.ones_like(image)
        im_orig = image.copy()
    else:
        image, mask = tools.crop_to_bbox(image, mask)
        im_orig = image.copy()
        mean_v = int(image[np.nonzero(mask)].mean())
        image = np.where(mask, image, mean_v)

    if smoothing:
        image = tools.smoothing(image)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR).astype(np.float32)
    imgsize = rgb_image.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    saliency_map = sm.SMGetSM(rgb_image)

    saliency_map *= mask
    im_orig *= mask

    saliency_map = skiexp.rescale_intensity(saliency_map, out_range=(0, 1))

    if show:
        plt.figure(figsize=(24, 14))
        if smoothing:
            plt.subplot(131), plt.imshow(im_orig, 'gray', interpolation='nearest'), plt.title('input')
            plt.subplot(132), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('smoothed')
            plt.subplot(133), plt.imshow(saliency_map, 'gray', interpolation='nearest'), plt.title('saliency')
        else:
            plt.subplot(121), plt.imshow(im_orig, 'gray', interpolation='nearest'), plt.title('input')
            plt.subplot(122), plt.imshow(saliency_map, 'gray', interpolation='nearest'), plt.title('saliency')
        if show_now:
            plt.show()

    return im_orig, image, saliency_map


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    im_o, img, saliency = run(data_s, mask_s, show=False)
    im_o_s, img_s, saliency_s = run(data_s, mask_s, smoothing=True, show=False)

    plt.figure()
    plt.subplot(231), plt.imshow(im_o, 'gray', interpolation='nearest'), plt.title('input')
    plt.subplot(233), plt.imshow(saliency, 'gray', interpolation='nearest'), plt.title('saliency map')

    plt.subplot(234), plt.imshow(im_o_s, 'gray', interpolation='nearest'), plt.title('input')
    plt.subplot(235), plt.imshow(img_s, 'gray', interpolation='nearest'), plt.title('smoothed')
    plt.subplot(236), plt.imshow(saliency_s, 'gray', interpolation='nearest'), plt.title('saliency')
    plt.show()