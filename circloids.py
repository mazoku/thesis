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
import skimage.measure as skimea

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
        # surv_im += surv_k * cv2.resize(i, shape[::-1])
        surv_im += cv2.resize(i, shape[::-1])

        if show:
            plt.figure()
            plt.subplot(121), plt.imshow(surv_im, 'gray', interpolation='nearest'), plt.title('surv_im')
            plt.subplot(122), plt.imshow(i, 'gray', interpolation='nearest'), plt.title('current')
            if show_now:
                plt.show()

    return surv_im# * mask


def save_figure(image, mask, blobs, fname, blob_name, param_name, param_vals, show=False, show_now=True, max_rows=3, max_cols=5):
    n_images = len(blobs)
    if max_cols < n_images:
        n_cols = max_cols
        n_rows = math.ceil(n_images / max_cols)
    else:
        n_cols = n_images
        n_rows = 1

    dirs = fname.split('/')
    dir_name = os.path.curdir
    for i in range(len(dirs)):
        dir_name = os.path.join(dir_name, dirs[i])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    fig = plt.figure(figsize=(20, n_rows * 5))
    # fig = plt.figure()
    for i, p_blobs in enumerate(blobs):
        # plt.subplot(1, len(blobs), i + 1)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image, 'gray', vmin=0, vmax=255, interpolation='nearest')
        plt.title('%s, %s=%.2f' % (blob_name, param_name, param_vals[i]))
        for blob in p_blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='m', linewidth=2, fill=False)
            plt.gca().add_patch(c)
    fig.savefig(os.path.join(fname, '%s_%s.png' % (blob_name, param_name)))
    if show:
        if show_now:
            plt.show()
    else:
        plt.close(fig)


def create_masks(shapes, border_width=1, show=False, show_masks=False, show_now=True):
    """

    :param shapes: shapes of ellipses in the form [(width, height), (width, hwight), ...]
    :param border_width: width of bordering frame
    :param show: whether to visualize results
    :param show_now: whether to show the visualization now
    :return:
    """
    masks = []
    elipses = []
    if show_masks:
        plt.figure()  # to visualize mask
    i = 1
    for sh in shapes:
        masks_sh = []
        axes_x, axes_y = (int(sh[0] / 2), int(sh[1] / 2))
        # c_x = int(sh[0] / 2) + border_width
        # c_y = int(sh[1] / 2) + border_width
        # n_rows = sh[1] + 2 * border_width
        # n_cols = sh[0] + 2 * border_width
        # mask = np.zeros([x + 2 * border_width for x in (sh[1], sh[0])], dtype="uint8")
        maxi = max(sh[:2])
        c_x = int(maxi / 2) + border_width
        c_y = int(maxi / 2) + border_width
        mask = np.zeros((maxi + 2 * border_width, maxi + 2 * border_width), dtype="uint8")

        ellip_m = mask.copy()
        cv2.ellipse(ellip_m, (c_x, c_y), (axes_x, axes_y), sh[2], 0, 360, 1, -1)
        # contours, hierarchy = cv2.findContours(ellip_m.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt=contours[0]
        # x, y, w, h = cv2.boundingRect(cnt)
        min_r, min_c, max_r, max_c = skimea.regionprops(ellip_m)[0].bbox
        # ellip_m = ellip_m[y - border_width: y + h + 2 * border_width + 1, x - border_width: x + w + 2 * border_width + 1]
        ellip_m = ellip_m[min_r - border_width: max_r + border_width, min_c - border_width: max_c + border_width]
        mask = np.zeros(ellip_m.shape, dtype=np.uint8)
        n_rows, n_cols = mask.shape
        c_x = int(n_cols / 2) + border_width
        c_y = int(n_rows / 2) + border_width

        # masks visualization --
        if show_masks:
            plt.subplot(1,len(shapes),i), plt.imshow(ellip_m, 'gray', interpolation='nearest')
            i += 1

        masks_sh.append(ellip_m)

        corners = [(c_x, n_cols, 0, c_y), (c_x, n_cols, c_y, n_rows), (0, c_x, c_y, n_rows), (0, c_x, 0, c_y)]
        for (startX, endX, startY, endY) in corners:
            # construct a mask for each corner of the image, subtracting the elliptical center from it
            corner_m = mask.copy()
            cv2.rectangle(corner_m, (startX, startY), (endX, endY), 1, -1)
            corner_m = cv2.subtract(corner_m, ellip_m)
            masks_sh.append(corner_m)

        masks.append(masks_sh)
        elipses.append(((c_x, c_y), (axes_x, axes_y), sh[2]))
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
    if (show or show_masks) and show_now:
        plt.show()

    return masks, elipses


def masks_response(image, masks, type='dark', mean_val=None, offset=10, show=False, show_now=True):
    resps = []
    # resps = [image[np.nonzero(m)].mean() for m in masks]
    for mask in masks:
        pts = image[np.nonzero(mask)]
        resp = (pts.mean(), pts.std())
        resps.append(resp)

    center = resps[0]
    corners = np.array(resps[1:])

    #TODO: zakodovat ala LBP
    if type in ['dark', 'black']:
        final_resp = (center[0] < corners[:, 0]).all()
        if mean_val is not None:
            final_resp = final_resp and center[0] < (mean_val - offset)
    elif type in ['bright', 'white']:
        final_resp = (center[0] > corners[:, 0]).all()
        if mean_val is not None:
            final_resp = final_resp and center[0] > (mean_val + offset)
    else:
        raise ValueError('Unknown type provided. Possible types are: dark, black, bright, white.')

    if show:
        plt.figure()
        plt.suptitle('Circloids responses')
        vmin = 0
        vmax = image.max()
        plt.subplot(231), plt.imshow(image, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.title('input')
        for i, (resp, mask) in enumerate(zip(resps, masks)):
            plt.subplot(2, 3, i + 2)
            plt.imshow(image * mask, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax)
            plt.title('u=%.1f, std=%.2f' % (resp[0], resp[1]))
        if show_now:
            plt.show()

    return final_resp, resps


def run(image, mask, pyr_scale=2., min_pyr_size=20, show=False, show_now=True, save_fig=False):
    fig_dir = ''
    image = tools.smoothing(image)

    mask_shapes = [(5, 5, 0), (9, 5, 0), (9, 5, 45), (9, 5, 90), (9, 5, 135)]
    border_width = 1

    masks, elipses = create_masks(mask_shapes, border_width=border_width, show=False, show_masks=False)

    pyr_imgs = []
    pyr_masks = []
    pyr_positives = []
    pyr_outs = []
    for layer_id, (im_pyr, mask_pyr) in enumerate(zip(tools.pyramid(image, scale=pyr_scale, inter=cv2.INTER_NEAREST),
                                                   tools.pyramid(mask, scale=pyr_scale, inter=cv2.INTER_NEAREST))):
        im_pyr, mask_pyr = tools.crop_to_bbox(im_pyr, mask_pyr)
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

        hist, bins = skiexp.histogram(im_pyr[np.nonzero(mask_pyr)])
        liver_peak = bins[np.argmax(hist)]

        layer_positives = []  # hold candidates for current layer and all masks
        layer_outs = []

        if show:
            layer_fig = plt.figure(figsize=(24, 14))
            plt.suptitle('layer #%i' % layer_id)
            plt.subplot(2, len(mask_shapes), 1), plt.imshow(im_pyr, 'gray', interpolation='nearest')
            plt.title('input')

        for mask_id, (m, e) in enumerate(zip(masks, elipses)):
            mask_positives = []
            win_size = m[0].shape[::-1]
            step_size = 4
            for (x, y, win_mask, win) in tools.sliding_window(im_pyr, window_size=win_size, step_size=step_size, mask=mask_pyr):
                resp, m_resps = masks_response(win, m, mean_val=liver_peak, show=False)
                if resp:
                    # positives.append(((x, y), (x + win_size[0], y + win_size[1])))
                    elip = ((e[0][0] + x, e[0][1] + y), e[1], e[2])
                    mask_positives.append(elip)

            layer_positives.append(mask_positives)

            # creating out image
            mask_outs = np.zeros(im_pyr.shape)
            for i in mask_positives:
                tmp = np.zeros(im_pyr.shape)
                cv2.ellipse(tmp, i[0], i[1], i[2], 0, 360, 1, thickness=-1)
                mask_outs += tmp
            layer_outs.append(mask_outs)


            # visualization
            if show:
                im_vis = cv2.cvtColor(im_pyr.copy(), cv2.COLOR_GRAY2BGR)
                for i in mask_positives:
                    # cv2.rectangle(im_vis, i[0], i[1], (255, 0, 255), thickness=1)
                    cv2.ellipse(im_vis, i[0], i[1], i[2], 0, 360, (255, 0, 255), thickness=1)

                plt.subplot(2, len(mask_shapes), mask_id + len(mask_shapes) + 1)
                plt.imshow(im_vis, interpolation='nearest'), plt.title('%i positives' % len(mask_positives))
                plt.title('mask #%i' % mask_id)

        if show:
            # visualizing layer survival function
            plt.subplot(2, len(mask_shapes), 2)
            layer_surv = np.zeros_like(layer_outs[0])
            for im in layer_outs:
                layer_surv += im
            plt.imshow(layer_surv, interpolation='nearest'), plt.colorbar()
            plt.title('surv. fcn')

        if save_fig:
            fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/circloids/'
            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)
            layer_fig.savefig(os.path.join(fig_dir, 'layer_%i_responses.png' % layer_id), dpi=100)

            # plt.figure()
            # plt.suptitle('layer #%i, mask #%i' % (layer_id, mask_id))
            # plt.subplot(121), plt.imshow(im_pyr, 'gray', interpolation='nearest'), plt.title('input')
            # plt.subplot(122), plt.imshow(mask_outs, 'jet', interpolation='nearest'), plt.title('surv func')
        # plt.show()

        pyr_positives.append(layer_positives)
        pyr_outs.append(layer_outs)

    all_outs = [im for layer in pyr_outs for im in layer]
    # survival image acros pyramid layers
    surv_im = calc_survival_fcn(all_outs, pyr_masks, show=False)

    if show:
        surv_fig = plt.figure(figsize=(24, 14))
        plt.subplot(121), plt.imshow(pyr_imgs[0], 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(122), plt.imshow(surv_im, interpolation='nearest'), plt.colorbar(), plt.title('complete survival fcn')
        if save_fig:
            surv_fig.savefig(os.path.join(fig_dir, 'complete_survival_fcn.png'), dpi=100)

    if show_now:
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
    run(data_s, mask_s, pyr_scale=pyr_scale, show=True)