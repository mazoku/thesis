from __future__ import division

import sys
sys.path.append('../mrf_segmentation/')
from markov_random_field import MarkovRandomField

sys.path.append('../imtools/')
from imtools import tools

import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as skiexp
import scipy.stats as scista


def estimate_healthy_pdf(data, mask, params):
    perc = params['perc']
    k_std_l = params['k_std_h']
    simple_estim = params['healthy_simple_estim']

    ints = data[np.nonzero(mask)]
    hist, bins = skiexp.histogram(ints, nbins=256)
    if simple_estim:
        mu, sigma = scista.norm.fit(ints)
    else:
        ints = data[np.nonzero(mask)]

        n_pts = mask.sum()
        perc_in = n_pts * perc / 100

        peak_idx = np.argmax(hist)
        n_in = hist[peak_idx]
        win_width = 0

        while n_in < perc_in:
            win_width += 1
            n_in = hist[peak_idx - win_width:peak_idx + win_width].sum()

        idx_start = bins[peak_idx - win_width]
        idx_end = bins[peak_idx + win_width]
        inners_m = np.logical_and(ints > idx_start, ints < idx_end)
        inners = ints[np.nonzero(inners_m)]

        # liver pdf -------------
        mu = bins[peak_idx]
        sigma = k_std_l * np.std(inners)

    mu = int(mu)
    sigma = int(sigma)
    rv = scista.norm(mu, sigma)

    return rv


def estimate_outlier_pdf(data, mask, rv_healthy, outlier_type, params):
    prob_w = params['prob_w']

    probs = rv_healthy.pdf(data) * mask

    max_prob = rv_healthy.pdf(rv_healthy.mean())

    prob_t = prob_w * max_prob

    ints_out_m = probs < prob_t * mask

    ints_out = data[np.nonzero(ints_out_m)]

    if outlier_type == 'hypo':
        ints = ints_out[np.nonzero(ints_out < rv_healthy.mean())]
    elif outlier_type == 'hyper':
        ints = ints_out[np.nonzero(ints_out > rv_healthy.mean())]
    else:
        print 'Wrong outlier specification.'
        return

    mu, sigma = scista.norm.fit(ints)

    mu = int(mu)
    sigma = int(sigma)
    rv = scista.norm(mu, sigma)

    return rv


def calculate_intensity_models(data, mask, params):
    print 'calculating intensity models...'
    # liver pdf ------------
    rv_heal = estimate_healthy_pdf(data, mask, params)
    print '\tliver pdf: mu = ', rv_heal.mean(), ', sigma = ', rv_heal.std()
    # hypodense pdf ------------
    rv_hypo = estimate_outlier_pdf(data, mask, rv_heal, 'hypo', params)
    print '\thypodense pdf: mu = ', rv_hypo.mean(), ', sigma = ', rv_hypo.std()
    # hyperdense pdf ------------
    rv_hyper = estimate_outlier_pdf(data, mask, rv_heal, 'hyper', params)
    print '\thyperdense pdf: mu = ', rv_hyper.mean(), ', sigma = ', rv_hyper.std()

    models = dict()
    models['heal'] = rv_heal
    models['hypo'] = rv_hypo
    models['hyper'] = rv_hyper

    return models

def get_unaries(data, mask, models, params):
    rv_heal = models['heal']
    rv_hyper = models['hyper']
    rv_hypo = models['hypo']
    # mu_heal = models['mu_heal']
    mu_heal = rv_heal.mean()

    # if params['erode_mask']:
    #     if data.ndim == 3:
    #         mask = tools.eroding3D(mask, skimor.disk(5), slicewise=True)
    #     else:
    #         mask = skimor.binary_erosion(mask, np.ones((5, 5)))

    unaries_healthy = - rv_heal.logpdf(data) * mask
    if params['unaries_as_cdf']:
        unaries_hyper = - np.log(rv_hyper.cdf(data) * rv_heal.pdf(mu_heal)) * mask
        # removing zeros with second lowest value so the log(0) wouldn't throw a warning -
        tmp = 1 - rv_hypo.cdf(data)
        values = np.unique(tmp)
        tmp = np.where(tmp == 0, values[1], tmp)
        #-
        unaries_hypo = - np.log(tmp * rv_heal.pdf(mu_heal)) * mask
        unaries_hypo = np.where(np.isnan(unaries_hypo), 0, unaries_hypo)
    else:
        unaries_hyper = - rv_hyper.logpdf(data) * mask
        unaries_hypo = - rv_hypo.logpdf(data) * mask

    unaries = np.dstack((unaries_hypo.reshape(-1, 1), unaries_healthy.reshape(-1, 1), unaries_hyper.reshape(-1, 1)))
    unaries = unaries.astype(np.int32)

    return unaries

def run(im, mask, alpha=1, beta=1, scale=0.5, unaries=None, show=False, show_now=True):
    scale = 0.5  # scaling parameter for resizing the image
    alpha = 1  # parameter for weighting the smoothness term (pairwise potentials)
    beta = 1  # parameter for weighting the data term (unary potentials)
    print 'I\'m in'

    mrf = MarkovRandomField(im, mask=mask, alpha=alpha, beta=beta, scale=scale)
    mrf.set_unaries(unaries)
    unaries =
    mrf.run()
    return mrf.labels_orig

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

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