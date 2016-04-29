from __future__ import division

from skimage import feature
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

import matplotlib.pyplot as plt

import numpy as np
import cv2

import datetime

import sys
import os
if os.path.exists('../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


verbose = True


def set_verbose(val):
    global verbose
    verbose = val


def _debug(msg, msgType="[INFO]"):
    if verbose:
        print '{} {} | {}'.format(msgType, msg, datetime.datetime.now())


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')


def find_prototypes(im, mask, lbp, lbp_type, blob_type, w, n_points, k=1):
    if lbp_type == 'edge':
        labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    elif lbp_type == 'flat':
        labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    elif lbp_type == 'corner':
        i_14 = n_points // 4  # 1/4th of the histogram
        i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
        labels = (list(range(i_14 - w, i_14 + w + 1)) + list(range(i_34 - w, i_34 + w + 1)))
    else:
        raise IOError('Wrong prototype value.')

    t = k * im[np.nonzero(mask)].mean()
    if blob_type == 'hypo':
        blob_m = (im < t) * mask
    elif blob_type == 'hyper':
        blob_m = (im > t) * mask
    else:
        raise IOError('Wrong blobtype value.')
    lbp_m = np.logical_or.reduce([lbp == each for each in labels])
    lbp_prototype = np.logical_and(blob_m, lbp_m)

    return lbp_prototype


def run(image, mask=None, n_points=24, radius=3, lbp_type='flat', blob_type='hypo', pyr_scale=2, show=False,
        show_now=True, save_fig=False, verbose=True):
    set_verbose(verbose)

    if mask is None:
        mask = np.ones_like(image)

    image, mask = tools.crop_to_bbox(image, mask)
    image = tools.smoothing(image, sliceId=0)

    lbp_pyr = []
    pyr_imgs = []
    pyr_masks = []
    for layer_id, (im_pyr, mask_pyr) in enumerate(zip(tools.pyramid(image, scale=pyr_scale, inter=cv2.INTER_NEAREST),
                                                      tools.pyramid(mask, scale=pyr_scale, inter=cv2.INTER_NEAREST))):
        lbp_res = local_binary_pattern(im_pyr, n_points, radius, 'uniform')
        w = radius - 1
        lbp_prototype = find_prototypes(im_pyr, mask_pyr, lbp_res, lbp_type, blob_type, w, n_points, k=1)
        lbp_pyr.append(lbp_prototype.astype(np.float))
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

    # survival overall
    surv_overall = np.zeros(image.shape)
    for surv_l in lbp_pyr:
        surv_overall += cv2.resize(surv_l, image.shape[::-1])

    # visualization
    if show:
        n_layers = len(lbp_pyr)
        imgs = [image, surv_overall] + lbp_pyr
        n_non_layers = len(imgs) - len(lbp_pyr)
        titles = ['input', 'surv'] + ['layer #%i' % (i + 1) for i in range(n_layers)]
        to_rgb = [0, 0,] + n_layers * [1, ]
        cmaps = ['gray', 'jet'] + n_layers * ['jet']
        plt.figure()
        for i, (im, tit, cmap, conv) in enumerate(zip(imgs, titles, cmaps, to_rgb)):
            plt.subplot(1, n_layers + n_non_layers, i + 1)
            if conv:
                plt.imshow(label2rgb(im, image=pyr_imgs[i - n_non_layers], bg_label=0, alpha=0.5), interpolation='nearest')
            else:
                plt.imshow(im, cmap, interpolation='nearest')
                # if i == 1:
                #     plt.colorbar(ticks=np.unique(im.astype(np.int)))
            plt.title(tit)
        # plt.subplot(132), plt.imshow(label2rgb(hypo_m, image=im_bb, bg_label=0, alpha=0.5)), plt.title('input < input.mean()')
        # plt.subplot(133), plt.imshow(label2rgb(hypo_lbp, image=im_bb, bg_label=0, alpha=0.5)), plt.title('hypo lbp')

        # plt.figure()
        # counts, _, bars = plt.hist(lbp.ravel(), normed=True, bins=lbp.max() + 1, range=(0, lbp.max() + 1), facecolor='0.5')
        # for i in flat_labels:
        #     bars[i].set_facecolor('r')
        # plt.gca().set_ylim(ymax=np.max(counts[:-1]))
        # plt.gca().set_xlim(xmax=n_points + 2)
        # plt.title('lbp hist with highlighted flat prototypes')
        if show_now:
            plt.show()

    return surv_overall, lbp_pyr


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    show = True
    pyr_scale = 1.5
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'  # slice_id = 17
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_185_48441644_venous_5.0_B30f-.pklz'  # slice_id = 14
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_186_49290986_venous_5.0_B30f-.pklz'  # slice_id = 5
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    # tools.show_3d(tools.windowing(data))

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    run(data_s, mask_s, pyr_scale=pyr_scale, show=show)
    plt.show()

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