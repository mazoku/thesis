from __future__ import division
#TODO: pri testovani parametru urcit survival rate - kolik iteraci zmen parametru blob prezije / celkovy pocet iteraci

import sys
sys.path.append('../imtools/')
from imtools import tools

import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as skiexp
import skimage.feature as skifea
import scipy.stats as scista

import copy
import math
import os


def check_blob_intensity(image, intensity, mask=None, show=False, show_now=True):
    if mask is None:
        mask = np.ones_like(image)
    if intensity in ['bright', 'light', 'white']:
        im = image.copy() * mask
    elif intensity in ['dark', 'black']:
        im = (image.copy().max() - image) * mask
    else:
        raise ValueError('Wrong blob intensity.')
    if show:
        plt.figure()
        plt.subplot(121), plt.imshow(image, 'gray'), plt.title('input')
        plt.subplot(122), plt.imshow(im, 'gray'), plt.title('output')
        if show_now:
            plt.show()
    return im


def dog(image, mask=None, intensity='bright', min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.1, overlap=0.5):
    if mask is None:
        mask = np.ones_like(image)
    im = check_blob_intensity(image, intensity, show=False)
    try:
        blobs = skifea.blob_dog(im, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio,
                                threshold=threshold, overlap=overlap)
    except:
        return []
    blobs = [x for x in blobs if mask[x[0], x[1]]]
    blobs = np.array(blobs)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    return blobs


def log(image, intensity='bright', min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.1, overlap=0.5, log_scale=False):
    im = check_blob_intensity(image, intensity)
    blobs = skifea.blob_log(im, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap, log_scale=log_scale)
    blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    return blobs


def doh(image, intensity='bright', min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False):
    im = check_blob_intensity(image, intensity)
    blobs = skifea.blob_doh(im, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap, log_scale=log_scale)
    blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    return blobs


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
        plt.title('%s, %s=%.1f' % (blob_name, param_name, param_vals[i]))
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


def run(image, mask):
    image, mask = tools.crop_to_bbox(image, mask)
    image = tools.smoothing(image, sliceId=0)

    # DOG testing -----------------
    sigma_ratios = np.arange(0.6, 2, 0.2)
    thresholds = np.arange(0, 1, 0.1)
    overlaps = np.arange(0, 1, 0.2)
    dogs = []
    for i in sigma_ratios:
        blobs = dog(image, mask=mask, intensity='dark', sigma_ratio=i)
        dogs.append(blobs)
    save_figure(image, mask, dogs, 'blobs/dogs', 'DOG', 'sigma_ratio', sigma_ratios, show=True, show_now=False)

    dogs = []
    for i in thresholds:
        blobs = dog(image, mask=mask, intensity='dark', threshold=i)
        dogs.append(blobs)
    save_figure(image, mask, dogs, 'blobs/dogs', 'DOG', 'threshold', thresholds, show=True, show_now=False)

    dogs = []
    for i in overlaps:
        blobs = dog(image, mask=mask, intensity='dark', overlap=i)
        dogs.append(blobs)
    save_figure(image, mask, dogs, 'blobs/dogs', 'DOG', 'overlap', overlaps, show=True)


    # im_orig = image.copy()
    # image = check_blob_intensity(image, 'dark', show=False)
    # blobs_log = skifea.blob_log(image, max_sigma=30, num_sigma=10, threshold=.1)
    # if len(blobs_log) > 0:
    #     blobs_log[:, 2] = blobs_log[:, 2] * math.sqrt(2)
    #
    # blobs_dog = skifea.blob_dog(image, max_sigma=30, threshold=.1)
    # if len(blobs_log) > 0:
    #     blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)
    #
    # blobs_doh = skifea.blob_doh(image, max_sigma=30, threshold=.01)
    # if len(blobs_log) > 0:
    #     blobs_doh[:, 2] = blobs_doh[:, 2] * math.sqrt(2)
    #
    # blobs_list = [blobs_log, blobs_dog, blobs_doh]
    # colors = ['yellow', 'lime', 'red']
    # titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
    #           'Determinant of Hessian']
    # sequence = zip(blobs_list, colors, titles)
    #
    # blobs_list = [blobs_log, blobs_dog, blobs_doh]
    # colors = ['yellow', 'lime', 'red']
    # titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
    #           'Determinant of Hessian']
    # sequence = zip(blobs_list, colors, titles)
    #
    # fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True,
    #                          subplot_kw={'adjustable': 'box-forced'})
    # plt.tight_layout()
    #
    # axes = axes.ravel()
    # for blobs, color, title in sequence:
    #     ax = axes[0]
    #     axes = axes[1:]
    #     ax.set_title(title)
    #     ax.imshow(im_orig, 'gray', interpolation='nearest')
    #     ax.set_axis_off()
    #     for blob in blobs:
    #         y, x, r = blob
    #         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
    #         ax.add_patch(c)
    #
    # plt.show()

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

    # im_o, img, saliency = run(data_s, mask_s, show=False)
    # im_o_s, img_s, saliency_s = run(data_s, mask_s, smoothing=True, show=False)
    #
    # plt.figure()
    # plt.subplot(231), plt.imshow(im_o, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(233), plt.imshow(saliency, 'gray', interpolation='nearest'), plt.title('saliency map')
    #
    # plt.subplot(234), plt.imshow(im_o_s, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(235), plt.imshow(img_s, 'gray', interpolation='nearest'), plt.title('smoothed')
    # plt.subplot(236), plt.imshow(saliency_s, 'gray', interpolation='nearest'), plt.title('saliency')
    # plt.show()