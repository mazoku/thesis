__author__ = 'tomas'

import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as skifil
import skimage.exposure as skiexp
import skimage.restoration as skires

# - - - - - - - - - - - - - - - - - - - - - - - -
def run(img, params):
    slice_idx = 14
    vmin = params['win_level'] - params['win_width'] / 2
    vmax = params['win_level'] + params['win_width'] / 2

    img = img[slice_idx, :, :]
    img_w = windowing(img, params['win_level'], params['win_width'])
    pot = calc_potential(img_w, params)

    # plt.figure()
    # plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax), plt.title('input')
    # plt.subplot(122), plt.imshow(pot, 'gray', interpolation='nearest'), plt.title('smoothed')

    plt.show()


def windowing(img, level=50, width=350):
    minHU = level - width
    maxHU = level + width
    img_w = skiexp.rescale_intensity(img, in_range=(minHU, maxHU), out_range=(0, 255))

    return img_w


def calc_potential(img, params):
    # img_g = skifil.gaussian_filter(img, params['sigma'])
    img_g = skires.denoise_tv_chambolle(img, weight=params['tv_weight'])
    # img_g = skifil.denoise_bilateral(img, sigma_range=params['sigma_range'], sigma_spatial=params['sigma_spatial'])

    grad_1 = skifil.prewitt(img_g)
    grad_2 = skifil.roberts(img_g)
    grad_3 = skifil.scharr(img_g)
    grad_4 = skifil.sobel(img_g)

    # plt.figure()
    # hist, bins = skiexp.histogram(img_g)
    # plt.plot(bins, hist)
    # plt.show()

    plt.figure()
    max_v = 0.0005
    plt.subplot(221), plt.imshow(grad_1, 'gray', interpolation='nearest', vmax=max_v), plt.title('prewitt')
    plt.subplot(222), plt.imshow(grad_2, 'gray', interpolation='nearest', vmax=max_v), plt.title('roberts')
    plt.subplot(223), plt.imshow(grad_3, 'gray', interpolation='nearest', vmax=max_v), plt.title('scharr')
    plt.subplot(224), plt.imshow(grad_4, 'gray', interpolation='nearest', vmax=max_v), plt.title('sobel')
    # plt.show()

    return img_g