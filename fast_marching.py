__author__ = 'tomas'

import tools

import matplotlib.pyplot as plt
import numpy as np
import skfmm

def run(data, params, mask=None):
    debug = True
    vmin = params['win_level'] - params['win_width'] / 2
    vmax = params['win_level'] + params['win_width'] / 2

    if mask is None:
        mask = np.ones(data.shape, dtype=np.bool)

    mode = tools.get_hist_mode(data, mask)

    hypo_init = initial_region_hypo(data, mask, mode)
    hyper_init = initial_region_hyper(data, mask, mode)
    speed_hypo = speed_function_hypo(data, mask, mode, params['fat_int'], params['hypo_int'], params['alpha'], params['speed_hypo_denom'])
    speed_hyper = speed_function_hypo(data, mask, mode, params['min_speed'], params['high_int_const'], params['alpha'], params['speed_hypo_denom'])

    slice_idx = 14



    plt.figure()
    plt.subplot(231), plt.imshow(data[slice_idx,:,:], 'gray', vmin=vmin, vmax=vmax)
    plt.subplot(232), plt.imshow(hypo_init[slice_idx,:,:], 'gray'), plt.title('hypo initial')
    plt.subplot(235), plt.imshow(hyper_init[slice_idx,:,:], 'gray'), plt.title('hyper initial')
    plt.subplot(233), plt.imshow(speed_hypo[slice_idx,:,:], 'gray'), plt.title('speed hypo')#, plt.colorbar()
    plt.subplot(236), plt.imshow(speed_hyper[slice_idx,:,:], 'gray'), plt.title('speed hyper')#, plt.colorbar()
    plt.show()


def speed_function_hyper(data, mask, mode, min_speed, high_int_const, alpha, denom):
    speed_lower = np.where(data <= mode, 1, 0)
    speed_higher = np.where(data > mode, 1 - alpha * (data - mode) / denom)
    speed_signif_higher = np.where(data > mode + high_int_const, 0.2, 0)

    speed = speed_lower + speed_higher + speed_signif_higher
    speed *= mask

    return speed


def speed_function_hypo(data, mask, mode, fat_int, hypo_int, alpha, denom):
    speed_bellow_fat = np.where(data <= fat_int, 1, 0)
    speed_fat2hypo = np.where((data > fat_int)*(data <= hypo_int), \
                                      1 - alpha * (data + hypo_int) / denom, 0)
    speed_hypo2mode = np.where((data > hypo_int)*(data <= mode), \
                                      1 - alpha * (mode - data) / (mode - hypo_int), 0)
    speed_higher = np.where(data > mode, 1, 0)

    speed = speed_bellow_fat + speed_fat2hypo + speed_hypo2mode + speed_higher
    speed *= mask

    return speed


def initial_region_hyper(data, mask, mode):
    region = np.where(data <= mode, True, False)
    region *= mask

    return region


def initial_region_hypo(data, mask, mode, slice_axis=0):
    selem_size = 5
    region = np.where(data >= mode, True, False)
    border = tools.eroding3D(data, selem_size=selem_size)
    # if len(data.shape) == 3:
    #     border = tools.eroding3D(data, selem_size=selem_size)
    # else:
    #     # setting slicewise parameter to False will perform 2D erosion if the input data are 2D as well
    #     border = tools.eroding3D(data, selem_size=selem_size, dtype=np.uint8), slicewise=False)

    # In order to eliminate low density fat regions (located in the vicinity of the liver)
    # from the abnormality map, voxels having lower intensity than -70 HU are also included
    #  in the initial region - as written in Thesis from Rusko, page 78.

    region += border
    region *= mask

    return region