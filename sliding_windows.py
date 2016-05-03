from __future__ import division
#TODO: sepsat do experimentu

import sys
sys.path.append('../imtools/')
from imtools import tools

import matplotlib.pyplot as plt
import matplotlib.patches as matpat

import numpy as np
import skimage.exposure as skiexp
import skimage.feature as skifea
import scipy.stats as scista
import skimage.transform as skitra
import skimage.filters as skifil
import skimage.morphology as skimor

import cv2

import copy
import math
import os
import time


def calc_survival_fcn(imgs, masks, show=False, show_now=True):
    shape = imgs[0].shape
    surv_im = np.zeros(shape)
    n_imgs = len(imgs)
    surv_k = 1. / n_imgs
    for i in imgs:
        surv_im += cv2.resize(i, shape[::-1])

        if show:
            plt.figure()
            plt.subplot(121), plt.imshow(surv_im, 'gray', interpolation='nearest'), plt.title('surv_im')
            plt.subplot(122), plt.imshow(i, 'gray', interpolation='nearest'), plt.title('current')
            if show_now:
                plt.show()

    return surv_im# * mask


def run(image, mask, pyr_scale=2., min_pyr_size=20, show=False, show_now=True, verbose=True):
    image = tools.smoothing(image)

    pyr_imgs = []
    outs = []
    pyr_masks = []
    for im_pyr, mask_pyr in zip(tools.pyramid(image, scale=pyr_scale, min_size=(min_pyr_size, min_pyr_size), inter=cv2.INTER_NEAREST),
                                tools.pyramid(mask, scale=pyr_scale, min_size=(min_pyr_size, min_pyr_size), inter=cv2.INTER_NEAREST)):
        im_pyr, mask_pyr = tools.crop_to_bbox(im_pyr, mask_pyr)

        hist, bins = skiexp.histogram(im_pyr[np.nonzero(mask_pyr)])
        liver_peak = bins[np.argmax(hist)]

        # if show:
        #     plt.figure()
        #     plt.plot(bins, hist, 'b-', linewidth=3)
        #     plt.xlim(0, 255)
        #     plt.plot([bins[np.argmax(hist)],] * 2, [0, hist.max()], 'r-')
        #     plt.plot(bins[np.argmax(hist)], hist.max(), 'ro')
        #     plt.gca().annotate('peak', xy=(bins[np.argmax(hist)], hist.max()), xytext=(bins[np.argmax(hist)] + 20, hist.max() + 20),
        #                        arrowprops=dict(facecolor='black', shrink=0.05))
        #     if show_now:
        #         plt.show()

        out = np.zeros(im_pyr.shape)
        win_w = 10
        win_h = 10
        win_size = (win_w, win_h)
        step_size = 4
        for (x, y, win_mask, win) in tools.sliding_window(im_pyr, window_size=win_size, step_size=step_size):
            win_mean = win.mean()

            # suptitle = 'abs of diff'
            # out_val = abs(liver_peak - win_mean)

            # suptitle = 'entropy'
            # out_val = skifil.rank.entropy(win, skimor.disk(3)).mean()

            # suptitle = 'procent. zastoupeni hypo'
            # out_val = (win < liver_peak).sum() / (win_w * win_h)

            suptitle = 'procent. zastoupeni hypo + bin. open'
            m = win < liver_peak
            m = skimor.binary_opening(m, skimor.disk(3))
            out_val = m.sum() / (win_w * win_h)

            out[np.nonzero(win_mask)] = out_val

        out *= mask_pyr
        outs.append(out)
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

    # survival image acros pyramid layers
    surv_im = calc_survival_fcn(outs, pyr_masks, show=False)

    if show:
        n_layers = len(outs)
        plt.figure()
        plt.suptitle(suptitle)
        for i, (im, out) in enumerate(zip(outs, pyr_imgs)):
            plt.subplot(2, n_layers, i + 1), plt.imshow(im, 'gray', interpolation='nearest'), plt.title('inut at pyr. layer %i' % (i + 1))
            plt.subplot(2, n_layers, i + 1 + n_layers), plt.imshow(out, 'gray', interpolation='nearest'), plt.title('output at pyr. layer %i' % (i + 1))

        plt.figure()
        plt.subplot(121), plt.imshow(image, 'gray', interpolation='nearest')
        plt.subplot(122), plt.imshow(surv_im, 'jet', interpolation='nearest')

        if show_now:
            plt.show()

    return surv_im, pyr_imgs

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    pyr_scale = 1.5
    run(data_s, mask_s, pyr_scale=pyr_scale)