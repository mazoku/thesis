from __future__ import division

import sys
sys.path.append('../imtools/')
from imtools import tools

sys.path.append('../mrf_segmentation/')
from mrfsegmentation.markov_random_field import MarkovRandomField


import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as skiexp
import scipy.stats as scista

import saliency_akisato as salaki
import saliency_google as salgoo
import saliency_ik as salik
import saliency_mayo as salmay
import copy


def run(im, mask, alpha=1, beta=1, scale=0.5, show=False, show_now=True):
    scale = 0.5  # scaling parameter for resizing the image
    alpha = 1  # parameter for weighting the smoothness term (pairwise potentials)
    beta = 1  # parameter for weighting the data term (unary potentials)

    im_orig, image, salaki_map = salaki.run(im, mask)
    im_orig, image, salgoo_map = salgoo.run(im, mask=mask)
    im_orig, _, salik_map = salik.run(im, mask=mask, smoothing=True)
    im_orig, image, salmay_map = salmay.run(im, mask=mask, smoothing=True)

    # inverting intensity so that the tumor has high saliency
    salgoo_map = np.where(salgoo_map, 1 - salgoo_map, 0)
    # salik_map = skiexp.rescale_intensity(salik_map, out_range=(0, 1))

    saliencies = [salaki_map, salgoo_map, salik_map, salmay_map]
    tits = ['akisato', 'google', 'ik', 'mayo']
    # saliencies = [salaki_map, salgoo_map, salmay_map]
    # tits = ['akisato', 'google', 'mayo']

    # plt.figure()
    # for i, (im, tit) in enumerate(zip(saliencies, tits)):
    #     # plt.subplot(2, 2, i+1), plt.imshow(skiexp.rescale_intensity(im, out_range=(0, )), 'gray', interpolation='nearest')
    #     plt.subplot(2, 2, i+1), plt.imshow(im, 'gray', interpolation='nearest')
    #     plt.colorbar()
    #     plt.title(tit)
    # plt.show()


    mrf = MarkovRandomField(im, models_estim='hydohy', mask=mask, alpha=alpha, beta=beta, scale=scale)
    unary_domin = mrf.get_unaries()[:, :, 1]
    max_prob = unary_domin.max()

    # rescaling intensities
    # max_int = 0.5
    max_int = max_prob
    for i, im in enumerate(saliencies):
        saliencies[i] = skiexp.rescale_intensity(im, out_range=(0, max_int))

    unaries_domin_sal = [np.dstack((unary_domin, x.reshape(-1, 1))) for x in saliencies]

    results = []
    for unary in unaries_domin_sal:
        mrf.set_unaries(unary)
        res = mrf.run().copy()
        results.append(res)

    # mrf.set_unaries(unaries)
    # unaries =
    # mrf.run()
    # return mrf.labels_orig

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    run(data_s, mask_s)

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