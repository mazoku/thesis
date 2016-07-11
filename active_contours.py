from __future__ import division

import numpy as np

import sys
import os

import scipy.ndimage.filters as scindifil

if os.path.exists('../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

if os.path.exists('../growcut/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../growcut/')
    from growcut import growcut
else:
    print 'You need to import package growcut: https://github.com/mazoku/growcut'
    sys.exit(0)


def initialize(data, dens_min=0, dens_max=255, prob_c=0.2, prob_c2=0.01):
    _, liver_rv = tools.dominant_class(data, dens_min=dens_min, dens_max=dens_max, show=False, show_now=False)

    liver_prob = liver_rv.pdf(data)

    prob_c = 0.2
    prob_c_2 = 0.01
    seeds1 = liver_prob > (liver_prob.max() * prob_c)
    seeds2 = liver_prob <= (np.median(liver_prob) * prob_c_2)
    seeds = seeds1 + 2 * seeds2

    gc = GrowCut(data, seeds, smooth_cell=False, enemies_T=0.7)
    gc.run()

    labs = gc.get_labeled_im().astype(np.uint8)
    labs = scindifil.median_filter(labs, size=3)

    return labs


def run(im, mask=None, smoothing=False, show=False, show_now=True, verbose=True):
    init = initialize(im, dens_min=10, dens_max=245)

    pass

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

    data_s, mask_s = tools.crop_to_bbox(data_s, mask_s)
    mean_v = int(data_s[np.nonzero(mask_s)].mean())
    data_s = np.where(mask_s, data_s, mean_v)
    data_s *= mask_s.astype(data_s.dtype)
    run(data_s, mask=mask_s, smoothing=True, save_fig=False, show=True, verbose=verbose)