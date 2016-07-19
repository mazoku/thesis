from __future__ import division

import numpy as np
import scipy.stats as scista
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os

import cv2
import scipy.ndimage.filters as scindifil
import skimage.transform as skitra
import skimage.feature as skifea
import skimage.morphology as skimor
import skimage.measure as skimea
import skimage.segmentation as skiseg
import skimage.exposure as skiexp
from skimage import img_as_float

from collections import namedtuple

# import lankton_lls
import active_contours as ac

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


def seg_stats(mask, mask_res):
    stats = namedtuple('stats', ['tp', 'tn', 'fp', 'fn'])
    # n_pts = np.prod(mask.shape)
    mask = mask > 0
    mask_res = mask_res > 0
    tp = 100 * (mask * mask_res).sum() / mask.sum()
    tn = 100 * (np.logical_not(mask) * np.logical_not(mask_res)).sum() / np.logical_not(mask).sum()
    fp = 100 * (np.logical_not(mask) * mask_res).sum() / mask_res.sum()
    fn = 100 * (mask * np.logical_not(mask_res)).sum() / np.logical_not(mask_res).sum()

    res = stats(tp, tn, fp, fn)

    return res


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    verbose = True
    show = False
    show_now = False
    save_fig = False
    smoothing = True

    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    # 2D
    # slice_ind = 17
    slice_ind = 12
    data = data[slice_ind, :, :]
    data = tools.windowing(data)
    mask = mask[slice_ind, :, :]

    # 3D
    # data = tools.windowing(data)

    # SFM  ----
    method = 'sfm'
    params = {'rad': 10, 'alpha': 0.6, 'scale': 0.5, 'smoothing': smoothing, 'max_iters': 100,
              'save_fig': save_fig, 'show': show, 'show_now': show_now, 'verbose': verbose}
    data_sfm, mask_sfm, sfm = ac.run(data, slice=None, mask=None, method=method, **params)

    # MORPH SNAKES  ----
    method = 'morphsnakes'
    params = {'alpha': 2000, 'sigma': 1, 'smoothing_ls': 1, 'threshold': 0.3, 'balloon': 1, 'max_iters': 50,
              'smoothing': smoothing, 'save_fig': save_fig, 'show': show, 'show_now': show_now}
    data_morph, mask_morph, morph = ac.run(data, slice=None, mask=None, method=method, **params)

    # counting precision
    if mask.shape != mask_sfm.shape:
        mask_res = cv2.resize(mask_sfm.astype(np.uint8), mask.shape[::-1])
    else:
        mask_res = mask_sfm
    stats_sfm = seg_stats(mask, mask_res)

    if mask.shape != mask_morph.shape:
        mask_res = cv2.resize(mask_morph.astype(np.uint8), mask.shape[::-1])
    else:
        mask_res = mask_morph
    stats_morph = seg_stats(mask, mask_res)

    # tp_sfm = 100
    # tp_morph = 100

    ac.visualize_seg(data_sfm, mask_sfm, sfm, 'sfm, TP=%.1f, TN=%.1f, FP=%.1f, FN=%.1f' %
                     (stats_sfm.tp, stats_sfm.tn, stats_sfm.fp, stats_sfm.fn), False)
    ac.visualize_seg(data_morph, mask_morph, morph, 'morph_snakes, TP=%.1f, TN=%.1f, FP=%.1f, FN=%.1f' %
                     (stats_morph.tp, stats_morph.tn, stats_morph.fp, stats_morph.fn), False)
    plt.show()

    # if show_now == False:
    #     plt.show()