import os
import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import skimage.exposure as skiexp
import skimage.filters as skifil
import skimage.morphology as skimor
import skimage.measure as skimea
import skimage.feature as skifea

from sklearn import mixture

import cv2

if os.path.exists('../imtools/'):
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


def deriving_seeds_for_growcut(img):
    tools.seeds_from_hist(img, show=True, show_now=False, verbose=True, min_int=5, max_int=250)
    tools.seeds_from_glcm(img, show=True, show_now=False, verbose=True)

    plt.show()


def data_from_glcm(glcm):
    pts = np.argwhere(glcm)
    counts = [glcm[tuple(i)] for i in pts]
    data = []
    for p, c in zip(pts, counts):
        for i in range(c):
            data.append(p)
    data = np.array(data)
    return data


def fit_mixture(glcm, n_components, max_components=4, type='gmm'):
    print 'preparing data ...',
    data = data_from_glcm(glcm)
    print 'done'

    print 'fitting %s ...' % type,
    if type == 'dpgmm':
        gmm = mixture.DPGMM(n_components=n_components, covariance_type='spherical', alpha=0.1)
    elif type == 'gmm':
        if n_components == 0:
            aics = []
            n_comps = range(1, max_components + 1)
            print 'searching for optimal number of components: ',
            for i in n_comps:
                gmm = mixture.GMM(n_components=i, covariance_type='spherical')
                gmm.fit(data)
                aic = gmm.aic(data)
                print '(%i, %.2f)' % (i, aic),
                aics.append(aic)
            best = n_comps[np.argmin(np.array(aics))]
            print ' -> %i' % best
            gmm = mixture.GMM(n_components=best, covariance_type='spherical')
        else:
            gmm = mixture.GMM(n_components=n_components, covariance_type='spherical')
    else:
        raise ValueError('Wrong micture type. Allowed values: dpgmm, gmm')

    gmm.fit(data)
    print 'done'
    print 'means:'
    print gmm.means_

    print 'predicting %s ...' % type,
    y_pred = gmm.predict(data)
    glcm_labs = np.zeros(glcm.shape)
    for x, y in zip(data, y_pred):
        glcm_labs[tuple(x)] = y + 1
    print 'done'

    plt.figure()
    plt.subplot(121), plt.imshow(glcm, 'gray', interpolation='nearest')
    plt.subplot(122), plt.imshow(glcm_labs, 'jet', interpolation='nearest')
    plt.show()


def glcm_mesh_anal(glcm):
    inds = skifea.peak_local_max(np.triu(glcm), min_distance=20, exclude_border=False, indices=True, num_peaks=6)
    inds = np.array([int(round(np.mean(x))) for x in inds])
    print inds

    glcm_c = skimor.closing(glcm, selem=skimor.disk(3))
    min_coo = 10
    peaks_str = np.array([glcm[x, x] for x in inds])
    sorted_idxs = np.argsort(peaks_str)[::-1]
    inds = inds[sorted_idxs]
    class_vals = [[] for i in range(len(inds))]
    labels = []
    for x in range(256):
        if glcm_c[x, x] > min_coo:
            dists = abs(inds - x)
            idx = np.argmin(dists)
            labels.append(idx)
            class_vals[idx].append(x)
        else:
            labels.append(-1)
    # print class_vals
    ellipses = []
    for i, c in zip(inds, class_vals):
        if c:
            c = np.array(c)
            # cent = int(round((c.max() + c.min()) / 2.))
            cent = [i, i]
            major_axis = (c.max() - c.min() )/ 2
            ellipses.append((cent, major_axis))

    # while True:


    plt.figure()
    plt.subplot(121), plt.imshow(glcm)
    plt.subplot(122), plt.imshow(glcm_c)

    plt.figure()
    plt.subplot(121), plt.imshow(glcm, 'jet')
    plt.axis('off')
    plt.subplot(122), plt.imshow(glcm, 'jet')
    plt.hold(True)
    for i in inds:
        plt.plot(i, i, 'ko')  # , markersize=14)
    plt.axis('image')
    plt.axis('off')
    #
    # plt.figure()
    # plt.imshow(glcm_c)
    # plt.hold(True)
    # colors = 5 * 'kwgcmy'
    # for x in range(256):
    #     if labels[x] != -1:
    #         plt.plot(x, x, colors[labels[x]] + 'o')
    # plt.axis('image')
    # plt.axis('off')

    # plt.figure()
    # tmp = cv2.cvtColor(glcm_c.copy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # colors = 5 * 'kwgcmy'
    # for e in ellipses:
    #     cv2.ellipse(tmp, (e[0], e[0]), (e[1], 5), 45, 0, 360, (255, 0, 255), thickness=2)
    # plt.imshow(tmp)
    plt.figure()
    plt.imshow(glcm)
    ax = plt.gca()
    for e in ellipses:
        ell = Ellipse(xy=e[0], width=e[1], height=5, angle=45)
        ax.add_artist(ell)
    plt.show()


def glcm_dpgmm(img):
    # deriving glcm
    mask = (img > 0) * (img < 255)
    glcm = tools.graycomatrix_3D(img, mask=mask)

    # inds = tuple(inds.flatten())

    # processing glcm
    # glcm_gc = skimor.closing(glcm, selem=skimor.disk(1))
    # glcm_go = skimor.opening(glcm, selem=skimor.disk(1))

    glcm_mesh_anal(glcm)

    # plt.figure()
    # plt.subplot(131), plt.imshow(glcm, 'gray', interpolation='nearest'), plt.title('glcm')
    # plt.subplot(132), plt.imshow(glcm_gc, 'gray', interpolation='nearest'), plt.title('glcm_gc')
    # plt.subplot(133), plt.imshow(glcm_go, 'gray', interpolation='nearest'), plt.title('glcm_go')

    # thresholding glcm
    # c_t = 4
    # thresh = c_t * np.mean(glcm)
    # glcm_t = glcm > thresh
    # glcm_to = skimor.binary_closing(glcm_t, selem=skimor.disk(3))
    # glcm_to = skimor.binary_opening(glcm_to, selem=skimor.disk(3))
    # tools.blob_from_gcm(glcm_to, img, return_rvs=True, show=True, show_now=False)
    #
    # labs_im, num = skimea.label(glcm_to, return_num=True)
    #
    # labels = np.unique(labs_im)[1:]
    # for l in labels:
    #     tmp = glcm * (labs_im == l)
    #     fit_mixture(tmp, n_components=0, type='gmm')

    # syntetic glcm
    # glcm = np.array([[0,1,1,2,0,0,0,0],
    #                  [1,2,2,3,1,0,0,1],
    #                  [1,3,4,2,1,0,0,0],
    #                  [0,1,3,1,0,0,0,1],
    #                  [1,0,0,0,0,0,1,3],
    #                  [0,2,0,0,0,2,2,1],
    #                  [0,0,0,0,1,3,4,0],
    #                  [0,0,0,0,1,2,0,0]])

    # dpgmm
    # glcm = glcm_go * glcm_to
    # glcm = glcm_go
    # glcm = glcm_gc
    glcm = glcm * glcm_to
    data = data_from_glcm(glcm)

    # fitting DPGMM
    # print 'fitting DPGMM ...'
    # types = ['spherical', 'tied', 'diag', 'full']
    # n_comps = range(2, 11)
    # # n_comps = range(2, 4)
    # aics = np.zeros((len(types), len(n_comps)))
    # bics = np.zeros((len(types), len(n_comps)))
    # scores = np.zeros((len(types), len(n_comps)))
    # for i, type in enumerate(types):
    #     print '\nTYPE:', type
    #     for j, n in enumerate(n_comps):
    #         # dpgmm = mixture.DPGMM(n_components=6, covariance_type='tied', alpha=0.100)
    #         dpgmm = mixture.GMM(n_components=n, covariance_type=type)
    #         dpgmm.fit(data)
    #         aic = dpgmm.aic(data)
    #         bic = dpgmm.bic(data)
    #         score = dpgmm.score(data).mean()
    #         # aics.append(aic)
    #         # bics.append(bic)
    #         # scores.append(score)
    #         aics[i, j] = aic
    #         bics[i, j] = bic
    #         scores[i, j] = score
    #         print 'n_comps=%i, score=%.2f, aic=%.2f, bic=%.2f' % (n, score, aic, bic)
    #
    # plt.figure()
    # color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    # for aic, color in zip(aics, color_iter):
    #     plt.plot(n_comps, aic, color + '-')
    # plt.legend(types)
    # plt.title('aic')
    #
    # plt.figure()
    # color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    # for bic, color in zip(bics, color_iter):
    #     plt.plot(n_comps, bic, color + '-')
    # plt.legend(types)
    # plt.title('bic')
    #
    # plt.figure()
    # color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    # for score, color in zip(scores, color_iter):
    #     plt.plot(n_comps, score, color + '-')
    # plt.legend(types)
    # plt.title('scores')

    print 'fitting DPGMM ...',
    # dpgmm = mixture.GMM(n_components=5, covariance_type='tied')
    dpgmm = mixture.DPGMM(n_components=6, covariance_type='tied', alpha=1.)
    dpgmm.fit(data)
    print 'done'
    print 'means:'
    print dpgmm.means_

    # dpgmm = mixture.GMM(n_components=3, covariance_type='tied')
    # dpgmm.fit(data)
    # print 'n_comps=3, score=%.2f, aic=%.2f, bic=%.2f' % (dpgmm.score(data).mean(), dpgmm.aic(data), dpgmm.bic(data))
    # dpgmm = mixture.GMM(n_components=4, covariance_type='tied')
    # dpgmm.fit(data)
    # print 'n_comps=4, score=%.2f, aic=%.2f, bic=%.2f' % (dpgmm.score(data).mean(), dpgmm.aic(data), dpgmm.bic(data))
    # dpgmm = mixture.GMM(n_components=5, covariance_type='tied')
    # dpgmm.fit(data)
    # print 'n_comps=5, score=%.2f, aic=%.2f, bic=%.2f' % (dpgmm.score(data).mean(), dpgmm.aic(data), dpgmm.bic(data))

    # predicting DPGMM
    print 'predicting DPGMM ...',
    y_pred = dpgmm.predict(data)
    glcm_labs = np.zeros(glcm.shape)
    for x, y in zip(data, y_pred):
        glcm_labs[tuple(x)] = y + 1
    print 'done'

    plt.figure()
    plt.subplot(121), plt.imshow(glcm, 'gray', interpolation='nearest')
    plt.subplot(122), plt.imshow(glcm_labs, 'jet', interpolation='nearest')
    plt.show()

    plt.show()

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fname = '/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/input.png'
    img = cv2.imread(fname, 0)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # deriving_seeds_for_growcut(img)

    glcm_dpgmm(img)