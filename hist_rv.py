from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import skimage.restoration as skires
import skimage.exposure as skiexp
import scipy.stats as scista


def run(img, mask, params, show=False):
    img_g = skires.denoise_tv_chambolle(img, weight=params['tv_weight'])
    img_g = skiexp.rescale_intensity(img_g, out_range=(img.min(), img.max()))

    hist, bins = skiexp.histogram(img_g[np.nonzero(mask)])
    peak_idx = hist.argmax()

    sigma = (img_g.max() - img_g.min()) / 100
    rv = scista.norm(bins[peak_idx], sigma)

    probs = rv.pdf(img_g) * mask

    if show:
        vmin = params['win_level'] - params['win_width'] / 2
        vmax = params['win_level'] + params['win_width'] / 2
        plt.figure()
        plt.subplot(121), plt.imshow(img_g, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax), plt.title('input')
        plt.subplot(122), plt.imshow(probs, 'gray', interpolation='nearest'), plt.title('peak probs')

    return rv, probs