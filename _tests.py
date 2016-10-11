from __future__ import division

import os
import sys
import itertools

import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import skimage.exposure as skiexp
import skimage.filters as skifil
import skimage.morphology as skimor
import skimage.segmentation as skiseg
import skimage.measure as skimea
import skimage.feature as skifea
import skimage.color as skicol
import scipy.ndimage.filters as scindifil

from sklearn import mixture
from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2

if os.path.exists('../imtools/'):
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

if os.path.exists('/home/tomas/projects/ShoPaBas/'):
    sys.path.insert(0, '/home/tomas/projects/ShoPaBas/')
    import ShoPaBas as spb
else:
    print 'Import error in %s. You need to import package ShoPaBas: https://github.com/mazoku/ShoPaBas'\
          % os.path.basename(__file__)
    sys.exit(0)


def deriving_seeds_for_growcut(img):
    # tools.seeds_from_hist(img, show=True, show_now=False, verbose=True, min_int=5, max_int=250, seed_area_width=10)
    # tools.seeds_from_glcm(img, show=True, show_now=False, verbose=True)
    tools.seeds_from_glcm_mesh(img, show=True, show_now=False, verbose=True)

    plt.show()


def glcm_meanshift(glcm):
    print 'calculating glcm ...',
    mask = (img > 0) * (img < 255)
    glcm = tools.graycomatrix_3D(img, mask=mask)
    print 'done'

    print 'preparing data ...',
    data = data_from_glcm(glcm)
    print 'done'

    print 'estimating bandwidth ...',
    bandwidth = estimate_bandwidth(data, quantile=0.08, n_samples=2000)
    print 'done'

    print 'fitting mean shift ...',
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms = MeanShift()
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    print 'done'

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print 'number of estimated clusters : %d' % n_clusters_
    print 'cluster centers :', cluster_centers

    y_pred = ms.predict(data)
    glcm_labs = np.zeros(glcm.shape, dtype=np.uint8)
    for x, y in zip(data, y_pred):
        glcm_labs[tuple(x)] = int(y + 1)

    inds = np.argsort(ms.cluster_centers_.mean(axis=1))
    glcm_labs2 = glcm_labs.copy()
    for i, l in enumerate(inds):
        glcm_labs2 = np.where(glcm_labs == l + 1, i + 1, glcm_labs2)

    # plt.figure()
    # plt.subplot(121), plt.imshow(glcm_labs)
    # plt.subplot(122), plt.imshow(glcm_labs2)

    glcm_labs = glcm_labs2

    labint = ms.predict(np.vstack((range(0, 256), range(0, 256))).T)
    labim = labint[img.flatten()].reshape(img.shape)
    # labim += 10
    labim2 = labim.copy()
    for i, l in enumerate(inds):
        labim2 = np.where(labim == l, i + 1, labim2)
    labim = labim2

    labim_f = scindifil.median_filter(labim, size=3)
    # plt.figure()
    # plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
    # plt.subplot(132), plt.imshow(labim, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.subplot(133), plt.imshow(labim_f, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.show()

    # deriving seeds
    # int_labels = []
    # for x in range(256):
    #     int_labels.append(ms.predict(np.array([[x, x]])))
    # seeds = np.array(int_labels)[img.flatten()].reshape(img.shape)
    # seeds_f = scindifil.median_filter(seeds, size=3)

    plt.figure()
    plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
    plt.subplot(132), plt.imshow(labim, 'jet', interpolation='nearest', vmin=0), plt.axis('off')
    plt.subplot(133), plt.imshow(labim_f, 'jet', interpolation='nearest', vmin=0), plt.axis('off')

    plt.figure()
    plt.subplot(121), plt.imshow(glcm, 'jet', interpolation='nearest', vmin=0), plt.axis('off')
    for c in ms.cluster_centers_:
        plt.plot(c[0], c[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=12)
    plt.subplot(122), plt.imshow(glcm_labs, 'jet', interpolation='nearest', vmin=0), plt.axis('off')
    for c in ms.cluster_centers_:
        plt.plot(c[0], c[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=12)

    # visualization
    # plt.figure()
    # plt.subplot(131), plt.imshow(img, 'gray'), plt.axis('off')
    # plt.subplot(132), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.subplot(133), plt.imshow(seeds_f, 'jet', interpolation='nearest'), plt.axis('off')
    #
    # plt.figure()
    # plt.subplot(121), plt.imshow(glcm, 'jet')
    # for c in cluster_centers:
    #     plt.plot(c[0], c[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=8)
    # plt.axis('image')
    # plt.axis('off')
    # plt.subplot(122), plt.imshow(glcm, 'jet')
    # colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     my_members = labels == k
    #     cluster_center = cluster_centers[k]
    #     plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
    #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=8)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.axis('image')
    # plt.axis('off')

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


def glcm_dpgmm(img):
    # deriving glcm
    mask = (img > 0) * (img < 255)
    glcm = tools.graycomatrix_3D(img, mask=mask)

    # inds = tuple(inds.flatten())

    # processing glcm
    # glcm_gc = skimor.closing(glcm, selem=skimor.disk(1))
    # glcm_go = skimor.opening(glcm, selem=skimor.disk(1))

    # plt.figure()
    # plt.subplot(131), plt.imshow(glcm, 'gray', interpolation='nearest'), plt.title('glcm')
    # plt.subplot(132), plt.imshow(glcm_gc, 'gray', interpolation='nearest'), plt.title('glcm_gc')
    # plt.subplot(133), plt.imshow(glcm_go, 'gray', interpolation='nearest'), plt.title('glcm_go')

    # thresholding glcm
    c_t = 4
    thresh = c_t * np.mean(glcm)
    glcm_t = glcm > thresh
    glcm_to = skimor.binary_closing(glcm_t, selem=skimor.disk(3))
    glcm_to = skimor.binary_opening(glcm_to, selem=skimor.disk(3))
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
    glcm_o = glcm.copy()
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
    # dpgmm = mixture.GMM(n_components=3, covariance_type='tied')
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
    data = data_from_glcm(glcm_o)
    y_pred = dpgmm.predict(data)
    glcm_labs = np.zeros(glcm.shape, dtype=np.uint8)
    for x, y in zip(data, y_pred):
        glcm_labs[tuple(x)] = int(y + 1)
    print 'done'

    # glcm_labs += 10
    inds = np.argsort(dpgmm.means_.mean(axis=1))
    glcm_labs2 = glcm_labs.copy()
    for i, l in enumerate(inds):
        glcm_labs2 = np.where(glcm_labs == l + 1, i + 1, glcm_labs2)
    glcm_labs = glcm_labs2
    # glcm_labs3 = inds[glcm_labs.flatten()].reshape(glcm_labs.shape)

    # plt.figure()
    # plt.subplot(121), plt.imshow(glcm_labs)
    # plt.subplot(122), plt.imshow(glcm_labs2)
    # plt.show()

    labint = dpgmm.predict(np.vstack((range(0, 256), range(0,256))).T)
    labim = labint[img.flatten()].reshape(img.shape)
    # labim += 10
    labim2 = labim.copy()
    for i, l in enumerate(inds):
        labim2 = np.where(labim == l, i + 1, labim2)
    labim = labim2
    # labim3 = inds[labim.flatten()].reshape(labim.shape)
    # plt.figure()
    # plt.subplot(121), plt.imshow(labim)
    # plt.subplot(122), plt.imshow(labim2)
    # # plt.subplot(133), plt.imshow(labim3)
    # plt.show()

    labim_f = scindifil.median_filter(labim, size=3)
    plt.figure()
    plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
    plt.subplot(132), plt.imshow(labim, 'jet', interpolation='nearest', vmin=0), plt.axis('off')
    plt.subplot(133), plt.imshow(labim_f, 'jet', interpolation='nearest', vmin=0), plt.axis('off')

    plt.figure()
    plt.subplot(121), plt.imshow(glcm_o, 'jet', interpolation='nearest', vmin=0), plt.axis('off')
    for c in dpgmm.means_:
        plt.plot(c[0], c[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=12)
    plt.subplot(122), plt.imshow(glcm_labs, 'jet', interpolation='nearest', vmin=0), plt.axis('off')
    for c in dpgmm.means_:
        plt.plot(c[0], c[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=12)

    plt.show()


def add_mask():
    dirpath = '/home/tomas/Dropbox/Data/medical/dataset/gt/'
    data_struc = [('180_venous-GT.pklz', '180_arterial-GT.pklz', [(x, y) for x, y in zip(range(5, 16), range(5, 16))]),
                  ('183a_venous-GT.pklz', '183a_arterial-GT.pklz', [(x, y) for x, y in zip(range(9, 23), range(7, 21))]),
                  ('185a_venous-GT.pklz', '185a_arterial-GT.pklz', [(x, y) for x, y in zip([9, ] + range(10, 24), [10, ] + range(10, 24))]),
                  ('186a_venous-GT.pklz', '186a_arterial-GT.pklz', [(x, y) for x, y in zip(range(1, 8), range(1, 8))]),
                  ('189a_venous-GT.pklz', '189a_arterial-GT.pklz', [(x, y) for x, y in zip(range(12, 26), range(11, 25))]),
                  ('221_venous-GT.pklz', '221_arterial-GT.pklz', [(x, y) for x, y in zip(range(17, 22), range(17, 22))]),
                  ('222a_venous-GT.pklz', '222a_arterial-GT.pklz', [(x, y) for x, y in zip(range(2, 7) + range(24, 30), range(1, 7) + range(21, 27))]),
                  ('232_venous-GT.pklz', '232_arterial-GT.pklz', [(x, y) for x, y in zip(range(1, 12) + range(17, 25), range(1, 12) + range(16, 24))]),
                  # ('234_venous-GT.pklz', '234_arterial-GT.pklz', range(0, 0, s)),
                  ('235_venous-GT.pklz', '235_arterial-GT.pklz', [(x, y) for x, y in zip(range(4, 42), range(4, 42))])]  # fujky
    data_struc = [(dirpath + x[0], dirpath + x[1], x[2]) for x in data_struc]

    for i, (data_ven_fn, data_art_fn, slics) in enumerate(data_struc):
        print '#%i/%i' % (i + 1, len(data_struc))
        data_ven, mask_ven, vs = tools.load_pickle_data(data_ven_fn)
        data_art, mask_art, vs = tools.load_pickle_data(data_art_fn)

        for j, (s_ven, s_art) in enumerate(slics):

            mask_im_ven = mask_ven[s_ven, ...]# > 0
            mask_im_art = mask_art[s_art, ...]# > 0

            dirs = data_ven_fn.split('/')
            fname = dirs[-1]
            salmap_fname = fname.replace('.pklz', '-sm-%i.pklz' % s_ven)
            dirp = '/'.join(dirs[:-1])
            out_ven = os.path.join(dirp, 'salmaps', salmap_fname)

            dirs = data_art_fn.split('/')
            fname = dirs[-1]
            salmap_fname = fname.replace('.pklz', '-sm-%i.pklz' % s_art)
            dirp = '/'.join(dirs[:-1])
            out_art = os.path.join(dirp, 'salmaps', salmap_fname)

            with gzip.open(out_ven, 'rb') as f:
                fcontent = f.read()
            data = pickle.loads(fcontent)
            # data_old = data[:]
            data.insert(1, (mask_im_ven, 'mask'))
            with gzip.open(out_ven, 'wb') as f:
                pickle.dump(data, f)

            with gzip.open(out_art, 'rb') as f:
                fcontent = f.read()
            data = pickle.loads(fcontent)
            data.insert(1, (mask_im_art, 'mask'))
            with gzip.open(out_art, 'wb') as f:
                pickle.dump(data, f)

            # with gzip.open(out_ven, 'rb') as f:
            #     fcontent = f.read()
            #     data = pickle.loads(fcontent)
            #
            # plt.figure()
            # n_imgs = len(data_old)
            # for i, (im, tit) in enumerate(data_old):
            #     # r = 1 + int(i > (n_imgs / 2))
            #     plt.subplot(201 + 10 * (np.ceil(n_imgs / 2)) + i)
            #     if i == 0:
            #         cmap = 'gray'
            #     else:
            #         cmap = 'jet'
            #     plt.imshow(im, cmap, interpolation='nearest')
            #     plt.title(tit)
            #
            # plt.figure()
            # n_imgs = len(data)
            # for i, (im, tit) in enumerate(data):
            #     # r = 1 + int(i > (n_imgs / 2))
            #     plt.subplot(201 + 10 * (np.ceil(n_imgs / 2)) + i)
            #     if i == 0:
            #         cmap = 'gray'
            #     else:
            #         cmap = 'jet'
            #     plt.imshow(im, cmap, interpolation='nearest')
            #     plt.title(tit)
            # plt.show()
        # break


def cell_smoothing():
    img = cv2.imread('/home/tomas/Dropbox/Work/Dizertace/figures/grow_cut/vyrez.png', 0)

    img1 = img < 10
    img2 = (img > 50) * (img < 150)
    img3 = img >= 150

    img = img1 + 2 * img2 + 3 * img3

    # plt.figure()
    # plt.subplot(141), plt.imshow(img1, 'gray')
    # plt.subplot(142), plt.imshow(img2, 'gray')
    # plt.subplot(143), plt.imshow(img3, 'gray')
    # plt.subplot(144), plt.imshow(img, 'gray')
    # plt.show()

    # median
    a = 5
    med = skifil.median(img, selem=np.ones((a, a)))

    # cell smoothing
    pts = np.argwhere(img).squeeze()
    r = 7
    nent = 0.5
    img_cs = img.copy()
    for p in pts:
        lbl = img[tuple(p)]
        m = np.zeros(img.shape)
        cv2.circle(m, (p[1], p[0]), r, 1, thickness=-1)
        surr_lbls = img[np.nonzero(m)]
        # surr_lbls = [data[int(self.neighborsM[q, p])] for q in range(self.nghood)
        #              if not np.isnan(self.neighborsM[q, p])]
        enemies = [x != lbl for x in surr_lbls]
        n_enemies = np.array(enemies).sum()
        if n_enemies >= nent * m.sum():
            hist, bins = skiexp.histogram(np.array(surr_lbls))
            enem_lbl = bins[np.argmax(hist)]
            img_cs[tuple(p)] = enem_lbl

    plt.figure()
    plt.subplot(131), plt.imshow(img, 'jet', interpolation='nearest'), plt.axis('off')
    plt.subplot(132), plt.imshow(med, 'jet', interpolation='nearest'), plt.axis('off')
    plt.subplot(133), plt.imshow(img_cs, 'jet', interpolation='nearest'), plt.axis('off')
    plt.show()


def circloids():
    im = cv2.imread('/home/tomas/Dropbox/Work/Dizertace/figures/saliency/input_img.png', 0)
    lm = cv2.imread('/home/tomas/Dropbox/Work/Dizertace/figures/saliency/local_mask.png', 0)
    gm = cv2.imread('/home/tomas/Dropbox/Work/Dizertace/figures/saliency/global_mask.png', 0)

    q1 = im[np.nonzero(lm == 200)].mean()
    q2 = im[np.nonzero(lm == 29)].mean()
    q3 = im[np.nonzero(lm == 104)].mean()
    q4 = im[np.nonzero(lm == 181)].mean()
    b = im[np.nonzero(gm == 200)].mean()
    c = im[np.nonzero(lm == 14)].mean()

    print c, ' - ', b, ' - ', q1, ', ', q2, ', ', q3, ', ', q4

    plt.figure()
    plt.subplot(231), plt.imshow(im, 'gray', interpolation='nearest', vmax=255)
    plt.subplot(232), plt.imshow(lm, 'jet', interpolation='nearest')
    plt.subplot(233), plt.imshow(gm, 'jet', interpolation='nearest')
    plt.show()


def hetero_split():
    orig = cv2.imread('/home/tomas/Dropbox/Data/medical/dataset/types/HCC/hcc_1_art.jpg', 0)
    im = cv2.imread('/home/tomas/Dropbox/Data/medical/features/hetero_split.png', 0)
    im = (im < 5 ) + 2 * (im > 250)

    # plt.figure()
    # plt.subplot(121), plt.imshow(orig, 'gray')
    # plt.subplot(122), plt.imshow(im, 'gray')
    # plt.show()

    over = skicol.label2rgb(im, image=orig, colors=['blue', 'red'], bg_label=0)

    mask = im > 0
    mask = skimor.binary_closing(mask, selem=skimor.disk(3))
    all = skiseg.mark_boundaries(orig, mask, color=(1, 0, 0), mode='thick')

    plt.figure()
    plt.subplot(131), plt.imshow(orig, 'gray'), plt.axis('off')
    plt.subplot(132), plt.imshow(over, 'gray'), plt.axis('off')
    plt.subplot(133), plt.imshow(all, 'gray'), plt.axis('off')
    plt.show()


def shopabas_example():
    fname = '/home/tomas/Dropbox/Data/medical/dataset/types/FNH/fnh_4_port.jpg'
    im = cv2.imread(fname, 0)
    shopa = spb.ShoPaBas(data=im, data_type=spb.DATA_IMG)
    spb.run()


def tumors_visualisation():
    # ICC - failed
    # im_a_fn = '/home/tomas/Dropbox/Data/medical/dataset/types/ICC/icc_3_art.jpg'
    # im_v_fn = '/home/tomas/Dropbox/Data/medical/dataset/types/ICC/icc_3_port.jpg'
    # mask_a_fn = '/home/tomas/Dropbox/Data/medical/dataset/types/ICC/icc_3_art_mask.jpg'
    # mask_v_fn = '/home/tomas/Dropbox/Data/medical/dataset/types/ICC/icc_3_port_mask.jpg'

    # HCC - failed
    # im_a_fn = '/home/tomas/Dropbox/Data/medical/dataset/types/HCC/hcc_1_art.jpg'
    # im_v_fn = '/home/tomas/Dropbox/Data/medical/dataset/types/HCC/hcc_1_port.jpg'
    # mask_a_fn = '/home/tomas/Dropbox/Data/medical/dataset/types/HCC/hcc_1_art_mask.jpg'
    # mask_v_fn = mask_a_fn

    # HCC - failed
    im_a_fn = '/home/tomas/Dropbox/Work/Dizertace/figures/lesion_class/hcc_ex_art.jpg'
    im_v_fn = '/home/tomas/Dropbox/Work/Dizertace/figures/lesion_class/hcc_ex_port.jpg'
    mask_a_fn = '/home/tomas/Dropbox/Work/Dizertace/figures/lesion_class/hcc_ex_art_mask.jpg'
    mask_v_fn = '/home/tomas/Dropbox/Work/Dizertace/figures/lesion_class/hcc_ex_port_mask.jpg'

    im_a = cv2.imread(im_a_fn, 0)
    im_v = cv2.imread(im_v_fn, 0)

    mask_a = cv2.imread(mask_a_fn, 0) > 50
    mask_v = cv2.imread(mask_v_fn, 0) > 50

    plt.figure()
    plt.subplot(121), plt.imshow(skiseg.mark_boundaries(im_a, mask_a, color=(1, 0, 0), mode='thick'), 'gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(122), plt.imshow(skiseg.mark_boundaries(im_v, mask_v, color=(1, 0, 0), mode='thick'), 'gray', interpolation='nearest')
    plt.axis('off')
    plt.show()


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # cells smoothing ----
    # cell_smoothing()
    # ----

    # circloids ----
    # circloids()
    #----

    # hetero split ----
    # hetero_split()
    #----

    # shopabas example
    # shopabas_example()
    #----

    # tumors visualisation ----
    tumors_visualisation()
    #----

    # add_mask()
    # fname = '/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/input.png'
    # img = cv2.imread(fname, 0)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # deriving_seeds_for_growcut(img)

    # glcm_dpgmm(img)

    # glcm_meanshift(img)