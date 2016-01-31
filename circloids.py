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


def create_masks(shapes, border_width=1, show=False, show_now=True):
    masks = []
    for sh in shapes:
        masks_sh = []
        axes_x, axes_y = (int(sh[0] / 2), int(sh[1] / 2))
        c_x = int(sh[0] / 2) + border_width
        c_y = int(sh[1] / 2) + border_width
        n_rows = sh[0] + 2 * border_width
        n_cols = sh[1] + 2 * border_width
        mask = np.zeros([x + 2 * border_width for x in sh], dtype="uint8")

        ellip_m = mask.copy()
        cv2.ellipse(ellip_m, (c_x, c_y), (axes_x, axes_y), 0, 0, 360, 1, -1)

        masks_sh.append(ellip_m)

        corners = [(c_x, n_rows, 0, c_y), (c_x, n_rows, c_y, n_cols), (0, c_x, c_y, n_cols), (0, c_x, 0, c_y)]
        for (startX, endX, startY, endY) in corners:
            # construct a mask for each corner of the image, subtracting the elliptical center from it
            corner_m = mask.copy()
            cv2.rectangle(corner_m, (startX, startY), (endX, endY), 1, -1)
            corner_m = cv2.subtract(corner_m, ellip_m)
            masks_sh.append(corner_m)

        masks.append(masks_sh)

    if show:
        for masks_sh in masks:
            mask = np.zeros(masks_sh[0].shape, dtype=np.byte)
            for i, m in enumerate(masks_sh):
                mask += (i + 1) * m
            plt.figure()
            plt.subplot(231)
            plt.imshow(mask, interpolation='nearest'), plt.colorbar()
            for i, m in enumerate(masks_sh):
                plt.subplot(2, 3, i + 2)
                plt.imshow(m, 'gray', interpolation='nearest')
            if show_now:
                plt.show()


def run(image, mask, pyr_scale=2., min_pyr_size=20, show=False, show_now=True):
    image = tools.smoothing(image)

    masks = create_masks([(5, 5)], show=True)

    pyr_imgs = []
    outs = []
    pyr_masks = []
    for im_pyr, mask_pyr in zip(tools.pyramid(image, scale=pyr_scale, inter=cv2.INTER_NEAREST),
                                tools.pyramid(mask, scale=pyr_scale, inter=cv2.INTER_NEAREST)):
        im_pyr, mask_pyr = tools.crop_to_bbox(im_pyr, mask_pyr)

        hist, bins = skiexp.histogram(im_pyr[np.nonzero(mask_pyr)])
        liver_peak = bins[np.argmax(hist)]

        # show=True
        if show:
            plt.figure()
            plt.plot(bins, hist, 'b-', linewidth=3)
            plt.xlim(0, 255)
            plt.plot([bins[np.argmax(hist)],] * 2, [0, hist.max()], 'r-')
            plt.plot(bins[np.argmax(hist)], hist.max(), 'ro')
            plt.gca().annotate('peak', xy=(bins[np.argmax(hist)], hist.max()), xytext=(bins[np.argmax(hist)] + 20, hist.max() + 20),
                               arrowprops=dict(facecolor='black', shrink=0.05))
            if show_now:
                plt.show()

        out = np.zeros(im_pyr.shape)
        win_w = 10
        win_h = 10
        win_size = (win_w, win_h)
        step_size = 4
        for (x, y, win_mask, win) in tools.sliding_window(im_pyr, window_size=win_size, step_size=step_size):
            #TODO: iterovat pres masky a pocitat odezvy
            resps = []

            # win_mean = win.mean()
            #
            # # suptitle = 'abs of diff'
            # # out_val = abs(liver_peak - win_mean)
            #
            # # suptitle = 'entropy'
            # # out_val = skifil.rank.entropy(win, skimor.disk(3)).mean()
            #
            # # suptitle = 'procent. zastoupeni hypo'
            # # out_val = (win < liver_peak).sum() / (win_w * win_h)
            #
            # suptitle = 'procent. zastoupeni hypo + bin. open'
            # m = win < liver_peak
            # m = skimor.binary_opening(m, skimor.disk(3))
            # out_val = m.sum() / (win_w * win_h)
            #
            # out[np.nonzero(win_mask)] = out_val

        out *= mask_pyr
        outs.append(out)
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

    # survival image acros pyramid layers
    surv_im = calc_survival_fcn(outs, pyr_masks, show=False)

        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(image, 'gray', interpolation='nearest')
        # sw = matpat.Rectangle((x, y), win.shape[1], win.shape[0], fill=False, edgecolor='m', linewidth=3)
        # plt.gca().add_patch(sw)
        #
        # plt.subplot(132)
        # plt.imshow(out, 'gray', interpolation='nearest')
        # sw = matpat.Rectangle((x, y), win.shape[1], win.shape[0], fill=False, edgecolor='m', linewidth=3)
        # plt.gca().add_patch(sw)
        #
        # plt.subplot(133)
        # ax = plt.gca()
        # ax.text(0.1, 0.9, 'liver_peak = %i' % liver_peak)
        # ax.text(0.1, 0.8, 'win_mean = %.1f' % win_mean)
        # ax.text(0.1, 0.7, 'out_val = %.1f' % out_val)
        # plt.axis('off')
        # plt.show()

    # for i, (res_gaus, res_lap) in enumerate(zip(skitra.pyramid_gaussian(image, downscale=pyr_scale), skitra.pyramid_laplacian(image, downscale=pyr_scale))):
    #     # if the image is too small, break from the loop
    #     if res_gaus.shape[0] < min_pyr_size or res_gaus.shape[1] < min_pyr_size:
    #         break
    #
    #     plt.figure()
    #     plt.subplot(121), plt.imshow(res_gaus, 'gray', interpolation='nearest'), plt.title('layer %i' % (i + 1))
    #     plt.subplot(122), plt.imshow(res_lap, 'gray', interpolation='nearest')

    # plt.figure()
    # plt.subplot(121), plt.imshow(image, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(out, 'gray', interpolation='nearest')
    # plt.show()
    n_layers = len(outs)
    plt.figure()
    plt.suptitle(suptitle)
    for i, (im, out) in enumerate(zip(outs, pyr_imgs)):
        plt.subplot(2, n_layers, i + 1), plt.imshow(im, 'gray', interpolation='nearest'), plt.title('inut at pyr. layer %i' % (i + 1))
        plt.subplot(2, n_layers, i + 1 + n_layers), plt.imshow(out, 'gray', interpolation='nearest'), plt.title('output at pyr. layer %i' % (i + 1))

    plt.figure()
    plt.subplot(121), plt.imshow(image, 'gray', interpolation='nearest')
    plt.subplot(122), plt.imshow(surv_im, 'gray', interpolation='nearest')
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

    pyr_scale = 1.5
    run(data_s, mask_s, pyr_scale=pyr_scale)