__author__ = 'tomas'

import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as skifil
import skimage.exposure as skiexp
import skimage.restoration as skires
import scipy.stats as scista

# - - - - - - - - - - - - - - - - - - - - - - - -
def run(img, params, mask=None):
    slice_idx = 14
    vmin = params['win_level'] - params['win_width'] / 2
    vmax = params['win_level'] + params['win_width'] / 2

    img = img[slice_idx, :, :]
    mask = mask[slice_idx, :, :]
    # img_w = windowing(img, params['win_level'], params['win_width'])
    # pot = calc_potential(img_w, params)

    pot = experimental_potential(img, mask, params)

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
    c = 5
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

    pot = c * np.abs(grad_1)

    plt.figure()
    max_v = 0.0005
    plt.subplot(221), plt.imshow(grad_1, 'gray', interpolation='nearest'), plt.title('grad')
    plt.subplot(222), plt.imshow(grad_2, 'gray', interpolation='nearest', vmax=max_v), plt.title('grad, adjusted vmax')
    plt.subplot(224), plt.imshow(pot, 'gray', interpolation='nearest'), plt.title('potential')
    # plt.show()

    return img_g


def experimental_potential(img, mask, params):
    img_g = skires.denoise_tv_chambolle(img, weight=params['tv_weight'])

    hist, bins = skiexp.histogram(img_g[np.nonzero(mask)])
    peak_idx = hist.argmax()

    # plt.figure()
    # plt.plot(bins, hist)
    # plt.hold(True)
    # plt.plot(bins[peak_idx], hist[peak_idx], 'ro')
    # plt.show()

    # diff = img_g.max() - img_g.min()
    # step = diff / 256
    # c = 5
    # img_t = (img_g > (bins[peak_idx] - c*step)) * (img_g <= (bins[peak_idx] + c*step))

    # plt.figure()
    # plt.imshow(img_t, 'gray', interpolation='nearest')

    sigma = 0.0005
    rv = scista.norm(bins[peak_idx], sigma)

    probs = rv.pdf(img_g) * mask

    plt.figure()
    plt.subplot(121), plt.imshow(img_g, 'gray', interpolation='nearest'), plt.title('input')
    plt.subplot(122), plt.imshow(probs, 'gray', interpolation='nearest'), plt.title('rv from hist')
    # plt.show()
