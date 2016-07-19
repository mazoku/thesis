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

    # 3D
    # data = tools.windowing(data)

    # SFM  ----
    method = 'sfm'
    params = {'rad': 10, 'alpha': 0.6, 'scale': 0.5, 'smoothing': smoothing, 'max_iters': 100,
              'save_fig': save_fig, 'show': show, 'show_now': show_now, 'verbose': verbose}
    data, mask, sfm = ac.run(data, slice=None, mask=None, method=method, **params)

    # MORPH SNAKES  ----
    method = 'morphsnakes'
    params = {'alpha': 2000, 'sigma': 1, 'smoothing_ls': 1, 'threshold': 0.3, 'balloon': 1, 'max_iters': 50,
              'smoothing': smoothing, 'save_fig': save_fig, 'show': show, 'show_now': show_now}
    data, mask, morph = ac.run(data, slice=None, mask=None, method=method, **params)

    ac.visualize_seg(data, mask, sfm, 'sfm', False)
    ac.visualize_seg(data, mask, morph, 'morph_snakes', False)
    plt.show()

    # if show_now == False:
    #     plt.show()