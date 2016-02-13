from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('../imtools/')
from imtools import tools

import sys
import os

import skimage.feature as skifea

FAST = 'FAST'


def detect_keypoints(image, mask, method, layer_id, smooth=True, show=False, show_now=True):
    ns = [6, 8, 10, 12]
    ts = [0.02, 0.05, 0.08, 0.15]
    resps = []
    for n, t in zip(ns, ts):
        resp = skifea.corner_fast(image, n=n, threshold=t)
        resp *= mask
        resps.append(resp)

    return resps


def run(image, mask, pyr_scale, method, smooth=True, show=False, show_now=True, save_fig=False):
    fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/saliency/%s/' % method
    if save_fig:
        dirs = fig_dir.split('/')
        for i in range(2, len(dirs)):
            subdir = '/'.join(dirs[:i])
            if not os.path.exists(subdir):
                os.mkdir(subdir)

    image, mask = tools.crop_to_bbox(image, mask)
    if smooth:
        image = tools.smoothing(image, sliceId=0)

    pyr_saliencies = []
    # pyr_survs = []
    # pyr_titles = []
    pyr_imgs = []
    pyr_masks = []
    for layer_id, (im_pyr, mask_pyr) in enumerate(zip(tools.pyramid(image, scale=pyr_scale, inter=cv2.INTER_NEAREST),
                                                      tools.pyramid(mask, scale=pyr_scale, inter=cv2.INTER_NEAREST))):
        resp = detect_keypoints(im_pyr, mask_pyr, method, layer_id=layer_id, smooth=smooth, show=show, show_now=show_now)

        pyr_saliencies.append(saliency)
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

    n_layers = len(pyr_imgs)



#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    show = True
    show_now = False
    save_fig = False
    smooth = True
    pyr_scale = 1.5
    method = FAST

    run(data_s, mask_s, pyr_scale=pyr_scale, method=method, smooth=True, show=show, show_now=show_now, save_fig=save_fig)
    # run_all(data_s, mask_s, pyr_scale=pyr_scale, smooth=smooth, show=show, show_now=show_now, save_fig=save_fig)

    if show:
        plt.show()