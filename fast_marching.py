__author__ = 'tomas'

import tools

import matplotlib.pyplot as plt
import numpy as np
import skfmm
import skimage.restoration as skires
import skimage.filter as skifil
import skimage.exposure as skiexp

def run(data, params, slice_idx=0, mask=None, weight=0.0001, show=False):

    vmin = params['win_level'] - params['win_width'] / 2
    vmax = params['win_level'] + params['win_width'] / 2

    if mask is None:
        mask = np.ones(data.shape, dtype=np.bool)

    # print 'min = %i, max = %i' % (data.min(), data.max())
    # weight = 0.001
    # sigma_range = 0.05
    # sigma_spatial = 15
    # win_size = 10
    # x = data[14,:,:]
    # x_min = x.min()
    # x_max = x.max()
    # y = skires.denoise_tv_chambolle(x, weight=weight, multichannel=False)
    # y = skiexp.rescale_intensity(y, (y.min(), y.max()), (x_min, x_max)).astype(x.dtype)
    # # y = skires.denoise_bilateral(x, win_size=win_size, sigma_range=sigma_range, sigma_spatial=sigma_spatial)
    # # y = skifil.gaussian_filter(x, sigma=0.1)
    # print 'min = %.2f, max = %.2f' % (y.min(), y.max())
    # plt.figure()
    # plt.subplot(121), plt.imshow(x, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(y, 'gray', interpolation='nearest')
    # plt.show()
    #
    # data_s = np.zeros(data)
    # for idx in range(data.shape[0]):
    #         data_s[idx, :, :] = skires.denoise_tv_chambolle(data[idx, :, :], weight=weight, multichannel=False)
    # print 'min = %i, max = %i' % (data_s.min(), data_s.max())

    # data = tools.smoothing_float_tv(data, weight=weight, sliceId=0)

    mode = tools.get_hist_mode(data, mask)
    print 'histogram mode = %i' % mode

    hypo_init = initial_region_hypo(data, mask, mode)
    hyper_init = initial_region_hyper(data, mask, mode)
    speed_hypo = speed_function_hypo(data, mask, mode, params['fat_int'], params['hypo_int'], params['alpha'], params['speed_hypo_denom'])
    speed_hyper = speed_function_hyper(data, mask, mode, params['min_speed'], params['high_int_const'], params['alpha'], params['speed_hypo_denom'])

    if show:
        # slice_idx = 14
        plt.figure()
        plt.subplot(231), plt.imshow(data[slice_idx,:,:], 'gray', vmin=vmin, vmax=vmax)
        plt.subplot(232), plt.imshow(hypo_init[slice_idx,:,:], 'gray'), plt.title('hypo initial')
        plt.subplot(235), plt.imshow(hyper_init[slice_idx,:,:], 'gray'), plt.title('hyper initial')
        plt.subplot(233), plt.imshow(speed_hypo[slice_idx,:,:], 'gray'), plt.title('speed hypo')#, plt.colorbar()
        plt.subplot(236), plt.imshow(speed_hyper[slice_idx,:,:], 'gray'), plt.title('speed hyper')#, plt.colorbar()
        plt.show()

    return hypo_init, hyper_init, speed_hypo, speed_hyper


def speed_function_hyper(data, mask, mode, min_speed, high_int_const, alpha, denom):
    speed_lower = np.where(data <= mode, 1, 0)
    speed_higher = np.where(data > mode, 1 - alpha * (data - mode) / denom, 0)
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
    selem_size = 3
    region = np.where(data >= mode, True, False)
    border = mask - tools.eroding3D(mask, selem_size=selem_size)
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