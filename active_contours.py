from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

import scipy.ndimage.filters as scindifil
import skimage.transform as skitra
import skimage.feature as skifea
import skimage.morphology as skimor
import skimage.exposure as skiexp
from skimage import img_as_float

import localized_level_sets as lls

if os.path.exists('../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'Import error in active_contours.py. You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

if os.path.exists('../growcut/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../growcut/')
    from growcut import growcut
else:
    print 'You need to import package growcut: https://github.com/mazoku/growcut'
    sys.exit(0)


def initialize(data, dens_min=0, dens_max=255, prob_c=0.2, prob_c2=0.01, show=False, show_now=True):
    _, liver_rv = tools.dominant_class(data, dens_min=dens_min, dens_max=dens_max, show=show, show_now=show_now)

    liver_prob = liver_rv.pdf(data)

    prob_c = 0.2
    prob_c_2 = 0.01
    seeds1 = liver_prob > (liver_prob.max() * prob_c)
    seeds2 = liver_prob <= (np.median(liver_prob) * prob_c_2)
    seeds = seeds1 + 2 * seeds2

    plt.figure()
    plt.subplot(131), plt.imshow(data, 'gray')
    plt.subplot(132), plt.imshow(liver_prob, 'gray')
    plt.subplot(133), plt.imshow(seeds, 'gray')
    plt.show()

    gc = growcut.GrowCut(data, seeds, smooth_cell=False, enemies_T=0.7)
    gc.run()

    labs = gc.get_labeled_im().astype(np.uint8)
    labs = scindifil.median_filter(labs, size=3)

    return labs


def initialize_graycom(data, distances, scale=0.5, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, c_t=5):
    if scale != 1:
        data = tools.resize3D(data, scale, sliceId=0)

    data = tools.smoothing(data, sigmaSpace=10, sigmaColor=10, sliceId=0)

    print 'Computing gray co-occurence matrix ...',
    gcm = skifea.greycomatrix(data, distances, angles, symmetric=symmetric)
    print 'done'

    gcm = gcm.sum(axis=3).sum(axis=2)

    # plt.figure()
    # thresh = np.mean(gcm)
    # ts = (thresh * np.array([1, 3, 6, 9, 12])).astype(int)
    # for i, t in enumerate(ts):
    #     plt.subplot(100 + 10 * len(ts) + i + 1)
    #     plt.imshow(gcm > t, 'gray')
    #     plt.title('t = %i' % t)
    # plt.show()

    thresh = c_t * np.mean(gcm)
    gcm_t = gcm > thresh

    gcm_t = skimor.binary_opening(gcm_t, selem=skimor.disk(3))

    plt.figure()
    plt.imshow(gcm_t, 'gray')
    plt.show()


def localized_seg(im, mask):
    im = skitra.rescale(im, scale=0.5, preserve_range=True)
    mask = skitra.rescale(mask, scale=0.5, preserve_range=True).astype(np.bool)
    lls.run(im, mask)


def run(im, mask=None, smoothing=False, show=False, show_now=True, save_fig=False, verbose=True):
    if smoothing:
        im = tools.smoothing(im, sigmaSpace=10, sigmaColor=10, sliceId=0)
    init_mask = initialize(im, dens_min=50, dens_max=200, show=True, show_now=False)[0,...]

    plt.figure()
    plt.subplot(121), plt.imshow(im, 'gray')
    plt.subplot(122), plt.imshow(init_mask, 'gray')
    plt.show()

    # init_mask = np.zeros(im.shape, dtype=np.bool)
    # init_mask[37:213, 89:227] = 1
    # localized_seg(im, init_mask)
    # init = initialize(im, dens_min=10, dens_max=245)


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    verbose = True
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    # data_w = tools.windowing(data)
    initialize_graycom(data_s, [1, 2], scale=1)

    #---
    # data = tools.windowing(data, sliceId=0)
    # data = tools.smoothing(data, sigmaSpace=10, sigmaColor=10, sliceId=0)
    # # plt.figure()
    # # plt.subplot(121), plt.imshow(data[slice_ind, ...], 'gray')
    # # plt.subplot(122), plt.imshow(data2[slice_ind, ...], 'gray')
    # # plt.show()
    # _, liver_rv = tools.dominant_class(data, dens_min=50, dens_max=220, show=True, show_now=False)
    # liver_prob = liver_rv.pdf(data)
    #
    # prob_c = 0.2
    # prob_c_2 = 0.01
    # seeds1 = liver_prob > (liver_prob.max() * prob_c)
    # seeds2 = liver_prob <= (np.median(liver_prob) * prob_c_2)
    # seeds = seeds1 + 2 * seeds2
    #
    # plt.figure()
    # plt.subplot(131), plt.imshow(data_s, 'gray')
    # plt.subplot(132), plt.imshow(liver_prob[slice_ind,...], 'gray')
    # plt.subplot(133), plt.imshow(seeds[slice_ind,...], 'gray')
    # plt.show()
    #
    # gc = growcut.GrowCut(data, seeds, smooth_cell=False, enemies_T=0.7, maxits=4)
    # gc.run()
    #
    # plt.figure()
    # plt.subplot(121), plt.imshow(data_s, 'gray')
    # plt.subplot(122), plt.imshow(gc.get_labeled_im().astype(np.uint8)[slice_ind,...], 'gray')
    # plt.show()
    #---

    # data_s, mask_s = tools.crop_to_bbox(data_s, mask_s)
    # mean_v = int(data_s[np.nonzero(mask_s)].mean())
    # data_s = np.where(mask_s, data_s, mean_v)
    # data_s *= mask_s.astype(data_s.dtype)
    run(data_s, mask=mask_s, smoothing=True, save_fig=False, show=True, verbose=verbose)