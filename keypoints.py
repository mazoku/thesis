from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

import sys
sys.path.append('../imtools/')
from imtools import tools

import sys
import os

import skimage.feature as skifea

BRISK = 'BRISK'
MSER = 'MSER'


def detect_brisk(image, mask, layer_id, smooth=True, show=False, show_now=True, save_fig=False):
    ns = [6, 9, 12]
    # ts = [0.02, 0.05, 0.08, 0.15]
    ts = [0.02, 0.07, 0.15]
    params = list(itertools.product(ns, ts))
    resps = []
    corners = []
    for n, t in params:
        resp = skifea.corner_fast(image, n=n, threshold=t)
        resp *= mask
        resps.append(resp)
        corners.append(skifea.corner_peaks(resp, min_distance=1))

    if show or save_fig:
        r = 2
        fig = plt.figure(figsize=(24, 14))
        n_rows = len(ns)
        n_cols = len(ts)
        for id, c_resp in enumerate(corners):
            plt.subplot(n_rows, n_cols, id + 1)
            plt.imshow(image, 'gray', interpolation='nearest')
            plt.title('layer=%i, ns=%i, ts=%.2f' % (layer_id + 1, params[id][0], params[id][1]))
            for y, x in c_resp:
                circ = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
                plt.gca().add_patch(circ)
        if save_fig:
            fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/keypoints/brisk/'
            dirs = fig_dir.split('/')
            for i in range(2, len(dirs)):
                subdir = '/'.join(dirs[:i])
                if not os.path.exists(subdir):
                    os.mkdir(subdir)
            fig.savefig(os.path.join(fig_dir, 'brisk_layer_%i.png' % (layer_id + 1)), dpi=100, bbox_inches='tight', pad_inches=0)
        if show_now:
            plt.show()

    return corners, resps


def detect_mser(image, mask, layer_id, smooth=True, show=False, show_now=True, save_fig=False):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.FeatureDetector_create('MSER')
    keypts = detector.detect(image)
    keypts = [kp for kp in keypts if mask[kp.pt[1], kp.pt[0]]]

    if show or save_fig:
        fig = plt.figure(figsize=(24, 14))
        for kp in keypts:
            plt.imshow(image, 'gray', interpolation='nearest')
            plt.title('layer=%i' % (layer_id + 1))
            r = int(0.5 * kp.size)
            x, y = kp.pt
            circ = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
            plt.gca().add_patch(circ)
        if save_fig:
            fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/keypoints/mser/'
            dirs = fig_dir.split('/')
            for i in range(2, len(dirs)):
                subdir = '/'.join(dirs[:i])
                if not os.path.exists(subdir):
                    os.mkdir(subdir)
            fig.savefig(os.path.join(fig_dir, 'mser_layer_%i.png' % (layer_id + 1)), dpi=100, bbox_inches='tight', pad_inches=0)
        if show_now:
            plt.show()


def detect_keypoints(image, mask, method, layer_id, smooth=True, show=False, show_now=True, save_fig=False):
    if method in ['brisk', 'BRISK', 'Brisk']:
        keypts, resps = detect_brisk(image, mask, layer_id, smooth=smooth, show=show, show_now=show_now, save_fig=save_fig)
    elif method in ['mser', 'MSER', 'Mser']:
        keypts = detect_mser(image, mask, layer_id, smooth=smooth, show=show, show_now=show_now, save_fig=save_fig)
        resps = []

    return keypts, resps


def run(image, mask, pyr_scale, method, smooth=True, show=False, show_now=True, save_fig=False):
    # fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/saliency/%s/' % method
    # if save_fig:
    #     dirs = fig_dir.split('/')
    #     for i in range(2, len(dirs)):
    #         subdir = '/'.join(dirs[:i])
    #         if not os.path.exists(subdir):
    #             os.mkdir(subdir)

    image, mask = tools.crop_to_bbox(image, mask)
    if smooth:
        image = tools.smoothing(image, sliceId=0)

    pyr_keypts = []
    # pyr_survs = []
    # pyr_titles = []
    pyr_imgs = []
    pyr_masks = []
    for layer_id, (im_pyr, mask_pyr) in enumerate(zip(tools.pyramid(image, scale=pyr_scale, inter=cv2.INTER_NEAREST),
                                                      tools.pyramid(mask, scale=pyr_scale, inter=cv2.INTER_NEAREST))):
        resp = detect_keypoints(im_pyr, mask_pyr, method, layer_id=layer_id, smooth=smooth,
                                show=show, show_now=show_now, save_fig=save_fig)

        pyr_keypts.append(resp)
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

    n_layers = len(pyr_imgs)

    # if show or save_fig:
    #     for layer_id, layer_keypts in enumerate(pyr_keypts):
    #         fig_keypts_layer = plt.figure(figsize=(24, 14))
    #         n_imgs = len(layer_keypts)
    #         for im_id, im in enumerate(layer_keypts):
    #             plt.subplot(1, n_imgs, im_id + 1)
    #             plt.imshow



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
    save_fig = True
    smooth = True
    pyr_scale = 1.5
    # method = BRISK
    method = MSER

    run(data_s, mask_s, pyr_scale=pyr_scale, method=method, smooth=True, show=show, show_now=show_now, save_fig=save_fig)
    # run_all(data_s, mask_s, pyr_scale=pyr_scale, smooth=smooth, show=show, show_now=show_now, save_fig=save_fig)

    if show:
        plt.show()