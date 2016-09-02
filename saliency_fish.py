# from __future__ import division

import math
import logging
import cv2
import numpy
from scipy.ndimage.filters import maximum_filter
import scipy.ndimage.morphology as scindimor

import scipy.stats as scista
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import skimage.exposure as skiexp
import skimage.morphology as skimor
import skimage.filters as skifil
import skimage.feature as skifea
from skimage import img_as_float
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import sklearn.mixture as sklmix

import os.path
import sys
import datetime
import itertools

# for conspicuity calculations
import blobs
import circloids
import lbp
import sliding_windows
import he_pipeline

import sys
if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in %s. You need to import package imtools: https://github.com/mjirik/imtools'\
          % os.path.basename(__file__)
    sys.exit(0)

# if sys.version_info[0] != 2:
#     raise Exception("This script was written for Python version 2.  You're running Python %s." % sys.version)

logger = logging.getLogger(__name__)

verbose = False  # whether to write debug comments or not


def set_verbose(val):
    global verbose
    verbose = val


def _debug(msg, msg_type="[INFO]"):
    if verbose:
        print '{} {} | {}'.format(msg_type, msg, datetime.datetime.now())


def cross_scale_diffs(image, channel, levels=9, start_size=(640,480), ):
    """
    Extracts features by down-scaling the image levels times,
    transforms the image by applying the function channel to
    each scaled version and computing the difference between
    the scaled, transformed	versions.
        image : 		the image
        channel : 		a function which transforms the image into
                        another image of the same size
        levels : 		number of scaling levels
        start_size : 	tuple. The size of the biggest image in
                        the scaling	pyramid. The image is first
                        scaled to that size and then scaled by half
                        levels times. Therefore, both entries in
                        start_size must be divisible by 2^levels.
    """
    # image = channel(image)
    if image.shape != start_size:
        image = cv2.resize(image, dsize=start_size)

    scales = [image]
    for l in xrange(levels - 1):
        logger.debug("scaling at level %d", l)
        new_s = tools.pyramid_down(scales[-1], scale=4./3, smooth=False)
        # new_s = cv2.pyrDown(src=scales[-1])
        if new_s is not None:
            new_s = tools.sigmoid(new_s)
            scales.append(new_s)
        else:
            break
    levels = len(scales)

    d1 = 3
    d2 = 4
    features = []
    for i in xrange(0, levels - d2):
    # for i in xrange(0, levels - 5):
        big = scales[i]
        for j in (d1, d2):
        # for j in (3, 4):
            logger.debug("computing features for levels %d and %d", i, i + j)
            small = scales[i + j]
            srcsize = small.shape[1], small.shape[0]
            dstsize = big.shape[1], big.shape[0]
            logger.debug("Shape source: %s, Shape target :%s", srcsize, dstsize)
            scaled = cv2.resize(src=small, dsize=dstsize)
            features.append(((i+1,j+1), cv2.absdiff(big, scaled)))
            # print 'added (%i, %i)' % (i, i + j)
            # pass

    # for i, f in enumerate(features):
    #     plt.figure()
    #     plt.imshow(f[1], 'gray', interpolation='nearest')
    #     plt.title(f[0])
    # plt.show()

    return features, levels


def sumNormalizedFeatures(features, levels=9, startSize=(640,480)):
    """
        Normalizes the feature maps in argument features and combines them into one.
        Arguments:
            features	: list of feature maps (images)
            levels		: the levels of the Gaussian pyramid used to
                            calculate the feature maps.
            startSize	: the base size of the Gaussian pyramid used to
                            calculate the feature maps.
        returns:
            a combined feature map.
    """
    commonWidth = startSize[0] / 2**(levels/2 - 1)
    commonHeight = startSize[1] / 2**(levels/2 - 1)
    commonSize = commonWidth, commonHeight
    logger.info("Size of conspicuity map: %s", commonSize)
    # consp = N(cv2.resize(features[0][1], commonSize))
    consp = N_DOG(cv2.resize(features[0][1], commonSize))
    for f in features[1:]:
        # resized = N(cv2.resize(f[1], commonSize))
        resized = N_DOG(cv2.resize(f[1], commonSize))
        consp = cv2.add(consp, resized)
    return consp


def N(image):
    """
        Normalization parameter as per Itti et al. (1998).
        returns a normalized feature map image.
    """
    M = 8.	# an arbitrary global maximum to which the image is scaled.
            # (When saving saliency maps as images, pixel values may become
            # too large or too small for the chosen image format depending
            # on this constant)
    image = cv2.convertScaleAbs(image, alpha=M/image.max(), beta=0.)
    w,h = image.shape
    maxima = maximum_filter(image, size=(w/10,h/1))
    maxima = (image == maxima)
    mnum = maxima.sum()
    logger.debug("Found %d local maxima.", mnum)
    maxima = numpy.multiply(maxima, image)
    mbar = float(maxima.sum()) / mnum
    logger.debug("Average of local maxima: %f.  Global maximum: %f", mbar, M)
    return image * (M-mbar)**2


def N_DOG(img, n_iters=10, c_ex=5, c_in=1.5, sig_ex_perc=0.001, sig_in_perc=0.4):
    """
        Normalization function as per Itti et al. (2000).
        returns a normalized feature map image.
    """
    sig_ex = sig_ex_perc * img.shape[1]
    sig_in = sig_in_perc * img.shape[1]

    imgs = [img.copy(),]
    for i in range(n_iters):
        img = c_ex * skifil.gaussian(img, sigma=sig_ex) - c_in * skifil.gaussian(img, sigma=sig_in)
        imgs.append(img.copy())

    # plt.figure()
    # plt.suptitle('DoG normalization')
    # for i, im in enumerate(imgs):
    #     plt.subplot(200 + 10 * (np.ceil(n_iters / 2) + 1) + i + 1)
    #     plt.imshow(im, 'gray', interpolation='nearest')
    # plt.show()

    return img


def markMaxima(saliency):
    """
        Mark the maxima in a saliency map (a gray-scale image).
    """
    maxima = maximum_filter(saliency, size=(20,20))
    maxima = numpy.array(saliency == maxima, dtype=numpy.float64) * 255
    r = cv2.max(saliency, maxima)
    g = saliency
    b = saliency
    marked = cv2.merge((b,g,r))
    return marked


def save_figs(intensty, gabor, rg, by, cout, saliency, saliency_mark_max, base_name='res'):
    cv2.imwrite(base_name + '_intensity.png', intensty)
    cv2.imwrite(base_name + '_gabor.png', gabor)
    cv2.imwrite(base_name + '_rg.png', rg)
    cv2.imwrite(base_name + '_by.png', by)
    cv2.imwrite(base_name + '_cout.png', cout)
    cv2.imwrite(base_name + '_saliency.png', saliency)
    cv2.imwrite(base_name + '_saliency_mark_max.png', saliency_mark_max)


# def dominant_colors(img, mask=None, k=3):
#     if mask is None:
#         mask = np.ones_like(img)
#     clt = KMeans(n_clusters=k)
#     data = img[np.nonzero(mask)]
#     clt.fit(data.reshape(len(data), 1))
#     cents = clt.cluster_centers_
#
#     return cents
#
#
# def color_clustering(img, mask=None, colors=None, k=3):
#     if mask is None:
#         mask = np.ones(img.shape[:2])
#     if colors is None:
#         colors = dominant_colors(img, mask=mask, k=k)
#     diffs = np.array([np.abs(img - x) for x in colors])
#
#     clust = np.argmin(diffs, 0)
#
#     # plt.figure()
#     # plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest')
#     # plt.subplot(122), plt.imshow(clust, 'jet', interpolation='nearest')
#     # plt.show()
#
#     return clust, colors


def conspicuity_int_diff(im, mask=None, use_sigmoid=False, morph_proc=True, type='hypo', a=3):
    if mask is None:
        mask = np.ones_like(im)

    mean_v = im[np.nonzero(mask)].mean()
    # hist, bins = skiexp.histogram(im[np.nonzero(mask)])
    # hist = tools.hist_smoothing(bins, hist, sigma=10)
    # max_peak_idx = hist.argmax()
    # mean_v = bins[max_peak_idx]
    # print 'mean = {:.2f}, peak = {:.2f}'.format(im[np.nonzero(mask)].mean(), mean_v)

    if type == 'both':
        im_int = np.abs(im - mean_v) * mask
    elif type == 'hypo':
        im_int = np.abs(im - mean_v) * ((im - mean_v) < 0) * mask
    else:  # type == 'hyper'
        im_int = np.abs(im - mean_v) * ((im - mean_v) > 0) * mask

    im_res = conspicuity_processing(im_int, mask, use_sigmoid=use_sigmoid, a=a, c=mean_v, sigm_t=0.2,
                                   use_morph=morph_proc, radius=3)

    return im_res


def conspicuity_int_hist(im, mask=None, use_sigmoid=False, morph_proc=True, type='hypo', a=3):
    if mask is None:
        mask = np.ones_like(im)

    class1, rv = tools.dominant_class(im, roi=mask)
    field = skimor.binary_closing(class1, selem=skimor.disk(3))
    field = skimor.binary_opening(field, selem=skimor.disk(3))
    field = (1 - field) * mask
    dist = scindimor.distance_transform_edt(field, return_distances=True)
    diff = abs(im - rv.mean())
    im_int = dist * diff

    # plt.figure()
    # i = cv2.resize(255*(class1 > 0).astype(np.uint8), (115, 215))
    # plt.imshow(i, 'gray'), plt.axis('off')
    # plt.show()

    # plt.figure()
    # plt.subplot(141), plt.imshow(cv2.resize(im, (115, 215)), 'gray'), plt.axis('off')
    # plt.subplot(142), plt.imshow(cv2.resize(dist, (115, 215)), 'jet'), plt.axis('off')
    # plt.subplot(143), plt.imshow(cv2.resize(diff, (115, 215)), 'jet'), plt.axis('off')
    # plt.subplot(144), plt.imshow(cv2.resize(im_int, (115, 215)), 'jet'), plt.axis('off')
    # plt.show()

    # im_int = skiexp.rescale_intensity(1 - rv.pdf(im), out_range=(0, 1))
    # plt.figure()
    # plt.subplot(121), plt.imshow(im_int, 'gray')
    # plt.subplot(122), plt.imshow(im_int2, 'gray'), plt.colorbar()
    # plt.show()
    mean_v = rv.mean()

    # plt.figure()
    # plt.subplot(141), plt.imshow(im, 'gray'), plt.title('input')
    # plt.subplot(142), plt.imshow(class1, 'gray'), plt.title('class1')
    # plt.subplot(143), plt.imshow(field, 'gray'), plt.title('field')
    # plt.subplot(144), plt.imshow(im_int, 'gray'), plt.title('dist')
    # plt.show()

    im_res = conspicuity_processing(im_int, mask, use_sigmoid=use_sigmoid, a=a, c=mean_v, sigm_t=0.2,
                                   use_morph=morph_proc, radius=3)

    return im_res


def conspicuity_int_glcm(im, mask=None, use_sigmoid=False, morph_proc=True, type='hypo', a=3):
    # im = tools.resize_ND(im, scale=0.5)
    # mask = tools.resize_ND(mask, scale=0.5)
    if mask is None:
        mask = np.ones_like(im)

    if im.max() <= 1:
        im = skiexp.rescale_intensity(im, (0, 1), (0, 255)).astype(np.int)

    glcm = tools.graycomatrix_3D(im, mask=mask)

    min_num = 2 * glcm.mean()
    glcm = np.where(glcm < min_num, 0, glcm)
    diag = np.ones(glcm.shape)
    k = 20
    tu = np.triu(diag, -k)
    tl = np.tril(diag, k)
    diag = tu * tl
    glcm *= diag.astype(glcm.dtype)

    # print 'data from glcm ...',
    data = tools.data_from_glcm(glcm)
    quantiles = [0.2, 0.1, 0.4]
    for q in quantiles:
        # print 'estimating bandwidth ...',
        bandwidth = estimate_bandwidth(data, quantile=q, n_samples=2000)
        # bandwidth = estimate_bandwidth(data, quantile=0.1, n_samples=2000)
        # print 'meanshift ...',
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)#, min_bin_freq=1000)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        n_clusters_ = len(np.unique(labels))
        # print q, n_clusters_

        if n_clusters_ > 1:
            break

    # n_clusters_ = 0
    if n_clusters_ > 1:
        # print 'number of estimated clusters : %d' % n_clusters_
        # print 'cluster centers: {}'.format(cluster_centers)
        lab_im = (1 + ms.predict(np.array(np.vstack((im.flatten(), im.flatten()))).T).reshape(im.shape)) * mask
    else:  # pokud meanshift najde pouze jeden mode, pouziji jiny pristup
        rvs = tools.analyze_glcm(glcm)
        rvs = sorted(rvs, key=lambda rv: rv.mean())
        lab_im = rvs[0].pdf(im)
        n_clusters_ = len(rvs)

    mean_v = im[np.nonzero(mask)].mean()
    labs = np.unique(lab_im)[1:]
    res = np.zeros_like(lab_im)
    for l in labs:
        tmp = lab_im == l
        mv = im[np.nonzero(tmp)].mean()
        if mv < mean_v:
            res = np.where(tmp, 1, res)

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
    # plt.show()

    # plt.figure()
    # plt.subplot(131), plt.imshow(im, 'gray', interpolation='nearest')
    # plt.subplot(132), plt.imshow(lab_im, 'jet', interpolation='nearest')
    # plt.subplot(133), plt.imshow(res, 'gray', interpolation='nearest')
    # plt.show()

    # thresholding graycomatrix (GCM)
    # c_t = 5
    # thresh = c_t * np.mean(glcm)
    # glcm_t = glcm > thresh
    # glcm_to = skimor.binary_opening(glcm_t, selem=skimor.disk(3))
    #
    # rvs = tools.analyze_glcm(glcm_to)

    # filtering
    # rvs = sorted(rvs, key=lambda rv: rv.mean())
    # im_int = rvs[0].pdf(im)
    # mean_v = rvs[0].mean()

    im_int = res

    a = 20
    c = mean_v / 255
    im_res = conspicuity_processing(im_int, mask, use_sigmoid=use_sigmoid, a=a, c=c, sigm_t=0.2,
                                   use_morph=morph_proc, radius=3)

    return im_res


def conspicuity_int_sliwin(im, mask=None, use_sigmoid=False, morph_proc=True, type='hypo', a=3):
    if mask is None:
        mask = np.ones_like(im)

    # plt.figure()
    # plt.subplot(121), plt.imshow(im, 'gray')
    # plt.subplot(122), plt.imshow(im[270:360, 250:360], 'gray')
    # plt.show()
    # im = im[270:380, 250:360]
    # mask = mask[270:380, 250:360]

    # hist, bins = skiexp.histogram(im[np.nonzero(mask)])
    # hist = tools.hist_smoothing(bins, hist)
    # liver_peak = bins[np.argmax(hist)]
    liver_peak = im[np.nonzero(mask)].mean()

    im_int = np.zeros(im.shape)
    win_w = 10 #20
    win_h = 10
    win_size = (win_w, win_h)
    step_size = int(0.4 * win_w)# / 2

    # its = [22, 68, 86]
    # wins = []
    # win_masks = []
    # ms = []
    # out_vals = []
    for i, (x, y, win_mask, win) in enumerate(tools.sliding_window(im, window_size=win_size, step_size=step_size)):
        # win_mean = win.mean()

        # abs of diff
        # out_val = abs(liver_peak - win_mean)

        # entropy
        # out_val = skifil.rank.entropy(win, skimor.disk(3)).mean()

        # procent. zastoupeni hypo
        # out_val = (win < liver_peak).sum() / (win_w * win_h)

        # procent. zastoupeni hypo + bin. open
        m = win < (0.9 * liver_peak)
        # m = skimor.binary_opening(m, skimor.disk(3))
        out_val = np.float(m.sum()) / (win_w * win_h)
        im_int[np.nonzero(win_mask)] += out_val

        # if i in its:
        #     wins.append(win)
        #     win_masks.append(win_mask)
        #     ms.append(m)
        #     out_vals.append(out_val)

    # # visualization --
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 200)]
    # im_in = cv2.cvtColor((255 * im.copy()).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # plt.figure()
    # for i, (win, win_mask, m) in enumerate(zip(wins, win_masks, ms)):
    #     rs, cs = np.nonzero(win_mask)
    #     cv2.rectangle(im_in, (cs.min(), rs.min()), (cs.max(), rs.max()), colors[i], thickness=2)
    #
    #     plt.subplot(150 + i + 3), plt.imshow(m, 'gray', interpolation='nearest', vmin=0, vmax=1), plt.axis('off')
    # # plt.subplot(151), plt.imshow(im_in, vmin=0, vmax=1), plt.axis('off')
    # plt.subplot(151), plt.imshow(im_in, vmin=100, vmax=100), plt.axis('off')
    # plt.subplot(152), plt.imshow(im_int * mask, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.show()

    im_int *= mask

    mean_v = im[np.nonzero(mask)].mean()
    c = mean_v / 255
    im_res = conspicuity_processing(im_int, mask, use_sigmoid=use_sigmoid, a=a, c=c, sigm_t=0.2,
                                   use_morph=morph_proc, radius=3)

    return im_res


# def conspicuity_prob_models(im, mask, type='both', show=False, show_now=True):
#     _debug('Running prob model conspicuity calculation ...')
#     clust, cents = color_clustering(im, mask, k=3)
#     if type == 'both':
#         pts_hypo = im[np.nonzero((clust == np.argmin(cents)) * mask)]
#         pts_hyper = im[np.nonzero((clust == np.argmax(cents)) * mask)]
#         # plt.figure()
#         # cs_i = np.argsort(cents.flatten())
#         # plt.subplot(131), plt.imshow((clust == cs_i[0]) * mask), plt.title('(a) hypo')
#         # plt.subplot(132), plt.imshow((clust == cs_i[1]) * mask), plt.title('(b) healthy')
#         # plt.subplot(133), plt.imshow((clust == cs_i[2]) * mask), plt.title('(c) hyper')
#         # plt.show()
#         pts = np.hstack((pts_hypo, pts_hyper))
#         rv = sklmix.GMM(n_components=2)
#         rv.fit(pts.reshape(len(pts), 1))
#     else:
#         if type == 'hypo':
#             pts = ((clust == np.argmin(cents)) * mask).flatten()
#         elif type == 'hyper':
#             pts = ((clust == np.argmax(cents)) * mask).flatten()
#         rv = scista.norm.fit(pts)
#
#     if show:
#         x = np.arange(256)
#         y = np.array(rv.predict_proba(x.reshape((len(x), 1)))).max(1)
#         plt.figure()
#         plt.plot(x, y, 'b-')
#         plt.axis('tight')
#     if show_now:
#         plt.show()


def conspicuity_texture(im, mask, p=24, r=3, lbp_type='flat', use_sigmoid=False, morph_proc=True, type='hypo'):
    lbp_res = skifea.local_binary_pattern(im, p, r, 'uniform')
    w = r - 1

    im_int = find_prototypes(im, mask, lbp_res, lbp_type, type, w, p, k=1)

    im_res = conspicuity_processing(im_int, mask, use_sigmoid=use_sigmoid, sigm_t=0.2, use_morph=morph_proc, radius=3)

    # flat_labels = list(range(0, w + 1)) + list(range(p - w, p + 2))
    # plt.figure()
    # counts, _, bars = plt.hist(lbp_res.ravel(), normed=True, bins=lbp_res.max() + 1, range=(0, lbp_res.max() + 1),
    #                            facecolor='0.5')
    # for i in flat_labels:
    #     bars[i].set_facecolor('r')
    # plt.gca().set_ylim(ymax=np.max(counts[:-1]))
    # plt.gca().set_xlim(xmax=p + 2)
    # plt.title('lbp hist with highlighted flat prototypes')
    # plt.figure()
    # plt.imshow(cv2.resize(im_int.astype(np.uint8), (115, 215)), 'gray'), plt.title('flat hypo')
    # plt.show()

    return im_res


def find_prototypes(im, mask, lbp, lbp_type, blob_type, w, n_points, k=1):
    if lbp_type == 'edge':
        labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    elif lbp_type == 'flat':
        labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    elif lbp_type == 'corner':
        i_14 = n_points // 4  # 1/4th of the histogram
        i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
        labels = (list(range(i_14 - w, i_14 + w + 1)) + list(range(i_34 - w, i_34 + w + 1)))
    else:
        raise IOError('Wrong prototype value.')

    t = k * im[np.nonzero(mask)].mean()
    if blob_type == 'hypo':
        blob_m = (im < t) * mask
    elif blob_type == 'hyper':
        blob_m = (im > t) * mask
    elif blob_type == 'both':
        blob_m = im * mask
    else:
        raise IOError('Wrong blobtype value.')
    lbp_m = np.logical_or.reduce([lbp == each for each in labels])
    lbp_prototype = np.logical_and(blob_m, lbp_m)

    return lbp_prototype


def conspicuity_blobs(im, mask, use_sigmoid=False, a=3, morph_proc=True, type='hypo', show=False, show_now=True):
    global verbose
    _debug('Running blob response conspicuity calculation ...')

    blob_types = blobs.BLOB_ALL
    # blob_types = [blobs.BLOB_CV,]
    # blobs_survs = []
    surv_overall = np.zeros_like(im)
    for blob_type in blob_types:
        blobs_res, survs_res, titles, params = blobs.detect_blobs(im, mask, blob_type, layer_id=0,
                                                                  show=show, show_now=show_now, verbose=verbose)
        # blobs_survs.append(survs_res[0])
        surv_overall += survs_res[0].astype(surv_overall.dtype)

    # survival overall
    surv = surv_overall
    # surv = blobs_survs

    return surv


def conspicuity_circloids(im, mask, use_sigmoid=False, a=3, morph_proc=True, type='hypo', show=False, show_now=True):
    _debug('Creating masks...')
    mask_shapes = [(15, 15, 0), (29, 15, 0), (29, 15, 45), (29, 15, 90), (29, 15, 135)]
    # mask_shapes = [(30, 30, 0), (58, 30, 0), (58, 30, 45), (58, 30, 90), (58, 30, 135)]
    # n_masks = len(mask_shapes)
    border_width = 1
    masks, elipses = circloids.create_masks(mask_shapes, border_width=border_width, show=False, show_masks=False)

    hist, bins = skiexp.histogram(im[np.nonzero(mask)])
    liver_peak = bins[np.argmax(hist)]

    # setting up values outside mask to better suppress responses on the border of the mask
    if type == 'hypo':
        im = np.where(mask, im, 0)
    else:
        im = np.where(mask, im, 1)

    # layer_positives = []  # list of candidates for current layer and all masks
    # layer_outs = []

    _debug('Calculating responses...')
    overall_positives = []
    masks_resp = []
    for mask_id, (m, e) in enumerate(zip(masks, elipses)):
        mask_positives = []
        win_size = m[0].shape[::-1]
        # step_size = 4
        step_size = 8
        for (x, y, win_mask, win) in tools.sliding_window(im, window_size=win_size, step_size=step_size, mask=mask):
            resp, m_resps = circloids.masks_response(win, m, offset=10./255, type=type, mean_val=liver_peak, show=False)

            if resp:
                # positives.append(((x, y), (x + win_size[0], y + win_size[1])))
                elip = ((e[0][0] + x, e[0][1] + y), e[1], e[2])
                mask_positives.append(elip)
                overall_positives.append(elip)
        masks_resp.append(mask_positives)

    # creating output image
    im_int = np.zeros(im.shape)
    for i in overall_positives:
        tmp = np.zeros(im.shape)
        cv2.ellipse(tmp, i[0], i[1], i[2], 0, 360, 1, thickness=-1)
        im_int += tmp

    # # creating mask response images
    # masks_survs = []
    # for m in masks_resp:
    #     surv = np.zeros(im.shape)
    #     for i in m:
    #         tmp = np.zeros_like(surv)
    #         cv2.ellipse(tmp, i[0], i[1], i[2], 0, 360, 1, thickness=-1)
    #         surv += tmp
    #     masks_survs.append(surv.copy())

    im_int *= mask

    mean_v = im[np.nonzero(mask)].mean()
    if mean_v > 1:
        c = mean_v / 255
    else:
        c = mean_v
    if im_int.max() > 0:
        im_res = conspicuity_processing(im_int, mask, use_sigmoid=use_sigmoid, a=a, c=c, sigm_t=0.2,
                                        use_morph=morph_proc, radius=3)
    else:
        im_res = im_int

    # plt.figure()
    # for i, im in enumerate(masks_survs):
    #     plt.subplot(151 + i)
    #     plt.imshow(im, 'jet')
    # plt.show()

    # return im_res, masks_survs
    return im_res




def conspicuity_he_pipeline(im, mask, proc, pyr_scale=1.5, min_pyr_size=(20, 20), show=False, show_now=True):
    heep_surv = he_pipeline.run(im, mask, proc=proc, pyr_scale=pyr_scale, min_pyr_size=min_pyr_size,
                                show=show, show_now=show_now, verbose=verbose)

    return heep_surv


def conspicuity_processing(map, mask, use_sigmoid=False, a=3, c=0.5, sigm_t=0.2, use_morph=False, radius=3):
    if map.max() == 0:
        return map

    # rescaling to <0, 1>
    map = skiexp.rescale_intensity(map.astype(np.float), out_range=(0, 1))

    # using sigmoid
    if use_sigmoid:
        map = tools.sigmoid(map, mask, a=a, c=c, sigm_t=sigm_t)
    # plt.figure()
    # plt.subplot(121), plt.imshow(map)
    # plt.subplot(122), plt.imshow(map1)
    # plt.show()

    # morphological postprocessing
    if use_morph:
        map = skimor.closing(skimor.opening(map, selem=skimor.disk(radius)))

    if map.max() > 0:
        map = skiexp.rescale_intensity(map.astype(np.float), out_range=np.float32)

    return map


def conspicuity_calculation(img, mask=None, sal_type='int_diff', n_levels=9, use_sigmoid=False, morph_proc=True,
                            type='hypo', calc_features=True, return_features=False, show=False, show_now=True, verbose=False):
    set_verbose(verbose)
    _debug('Running intensity conspicuity calculation, type: %s ...' % sal_type)

    if sal_type == 'int_diff':
        consp_fcn = conspicuity_int_diff
    elif sal_type == 'int_hist':
        consp_fcn = conspicuity_int_hist
    elif sal_type == 'int_glcm':
        consp_fcn = conspicuity_int_glcm
    elif sal_type == 'int_sliwin':
        consp_fcn = conspicuity_int_sliwin
    elif sal_type == 'texture':
        consp_fcn = conspicuity_texture
    elif sal_type == 'circloids':
        consp_fcn = conspicuity_circloids
    elif sal_type == 'blobs':
        consp_fcn = conspicuity_blobs
    else:
        raise ValueError('Wrong saliency types.')

    if mask is None:
        mask = np.ones_like(img)
    mask = mask.astype(np.float)

    n_rows, n_cols = img.shape
    if n_rows < n_cols:
        scale = 480 / n_rows
    else:
        scale = 480 / n_cols
    # img_0 = cv2.resize(img, (640, 480))
    # mask_0 = cv2.resize(mask, (640, 480))
    img_0 = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    mask_0 = cv2.resize(mask, (0, 0), fx=scale, fy=scale)

    # plt.figure()
    # plt.subplot(121), plt.imshow(img_0, 'gray')
    # plt.subplot(122), plt.imshow(img_1, 'gray')
    # plt.show()

    # im_pyramid = [img_0,]
    # mask_pyramid = [mask_0, ]
    im_pyramid = []
    mask_pyramid = []
    consp_pyramid = []
    # survs = []

    im_p = img_0
    mask_p = mask_0
    for l in xrange(n_levels - 1):
        _debug('Pyramid level %i' % (l + 1))
        params = {'mask': mask_p, 'use_sigmoid': use_sigmoid, 'morph_proc': morph_proc, 'type': type}
        # calculating conspicuity map if a consp_fcn was provided
        if consp_fcn is not None:
            consp_map = consp_fcn(im_p, **params)
        else:
            consp_map = im_p.copy()

        # survs.append(consp_map)
        # consp_map = consp_map[0]

        # append data to pyramids
        im_pyramid.append(im_p)
        mask_pyramid.append(mask_p)
        consp_pyramid.append(consp_map)

        # pyr-down data
        # logger.debug("scaling at level %d", l)
        im_p = tools.pyramid_down(im_pyramid[-1], scale=4. / 3, smooth=False)
        mask_p = tools.pyramid_down(mask_pyramid[-1], scale=4. / 3, smooth=False)

        # break the loop if the image can't be scaled down any further
        if im_p is None:
            break

    # blobs_survs = [cv2.resize(x, img.shape[::-1]) for x in survs[0]]
    # for pyr_lvl in range(1, len(survs)):
    #     for j in range(len(survs[pyr_lvl])):
    #         blobs_survs[j] += cv2.resize(survs[pyr_lvl][j], img.shape[::-1])
    # plt.figure()
    # n_rows = len(blobs_survs)
    # for i, im in enumerate(blobs_survs):
    #     plt.subplot(101 + 10 * n_rows + i)
    #     plt.imshow(im)
    #     plt.axis('off')
    #     divider = make_axes_locatable(plt.gca())
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     plt.colorbar(cax=cax)
    # plt.show()

    if calc_features:
        n_levels = len(consp_pyramid)
        d1 = 3
        d2 = 4
        features = []
        for i in xrange(0, n_levels - d2):
            # for i in xrange(0, levels - 5):
            big = consp_pyramid[i]
            for j in (d1, d2):
                # for j in (3, 4):
                # logger.debug("computing features for levels %d and %d", i, i + j)
                small = consp_pyramid[i + j]
                srcsize = small.shape[1], small.shape[0]
                dstsize = big.shape[1], big.shape[0]
                # logger.debug("Shape source: %s, Shape target :%s", srcsize, dstsize)
                scaled = cv2.resize(src=small, dsize=dstsize)
                f = cv2.absdiff(big, scaled)
                # f = cv2.resize(f, (img.shape[1], img.shape[0]))
                features.append(((i + 1, j + 1), f))
    else:
        features = [((0, 0), im) for im in consp_pyramid]

    # plt.figure()
    # n_cols = np.ceil(len(features) / 2)
    # for i, f in enumerate(features):
    #     plt.subplot(200 + 10 * n_cols + i + 1)
    #     plt.imshow(cv2.resize(f[1], img.shape[::-1]), 'jet')
    #     plt.axis('off')

    # plt.figure()
    # n_cols = np.ceil(len(im_pyramid) / 2.)
    # for i, f in enumerate(im_pyramid):
    #     plt.subplot(200 + 10 * n_cols + i + 1)
    #     plt.imshow(f, 'gray')
    # plt.show()
    #
    # plt.figure()
    # n_cols = np.ceil(len(consp_pyramid) / 2.)
    # for i, c in enumerate(consp_pyramid):
    #     plt.subplot(200 + 10 * n_cols + i + 1)
    #     plt.imshow(cv2.resize(c, img.shape[::-1]), 'gray')
    # plt.show()

    # calculating conspicuity map
    consp_map = sumNormalizedFeatures(features, levels=n_levels)
    consp_map = cv2.resize(consp_map, img.shape[::-1]) * mask
    consp_map = np.where(consp_map < 0, 0, consp_map)
    consp_map = skiexp.rescale_intensity(consp_map, out_range=(0, 1))

    if show:
        plt.figure()
        plt.subplot(121), plt.imshow(img * mask, 'gray'), plt.title('input')
        plt.axis('off')
        plt.subplot(122), plt.imshow(consp_map, 'jet'), plt.title('conspicuity map')
        plt.axis('off')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)
        if show_now:
            plt.show()

    if return_features:
        out = (consp_map, features)
    else:
        out = consp_map
    return out


def run(im, mask=None, save_fig=False, smoothing=False, return_all=False, show=False, show_now=True, verbose=True):
    """
    im ... grayscale image
    """
    show = True  # debug show

    set_verbose(verbose)

    if mask is None:
        mask = np.ones_like(im)
    else:
        im, mask = tools.crop_to_bbox(im, mask)
        mean_v = int(im[np.nonzero(mask)].mean())
        im = np.where(mask, im, mean_v)
        # im = np.where(mask, im, 255)

    im_orig = im.copy()
    orig_shape = im_orig.shape[:-1]

    if smoothing:
        im = tools.smoothing(im)

    # intensity
    use_sigmoid = True
    morph_proc = True
    # consp_int = conspicuity_intensity(im, mask=mask, int_type='diff', type='hypo', use_sigmoid=use_sigmoid, show=show, show_now=False)
    # consp_int = conspicuity_intensity(im, mask=mask, int_type='hist', type='hypo', use_sigmoid=use_sigmoid, show=show, show_now=False)
    # consp_int = conspicuity_intensity(im, mask=mask, int_type='glcm', type='hypo', use_sigmoid=use_sigmoid, show=show, show_now=False)
    # consp_sliwin = conspicuity_sliding_window(im, mask=mask, pyr_scale=1.5, show=show, show_now=False)

    im = img_as_float(im)
    # id = conspicuity_intensity(im.copy(), mask=mask, int_type='diff', type='hypo', use_sigmoid=use_sigmoid, show=show, show_now=False)
    # plt.show()
    # id = conspicuity_calculation(im, sal_type='int_diff', mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc, show=show, show_now=False)
    # id = conspicuity_calculation(im, sal_type='int_hist', mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc, show=show, show_now=False)
    # id = conspicuity_calculation(im, sal_type='int_glcm', mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc, show=show, show_now=False)
    # isw = conspicuity_calculation(im, sal_type='int_sliwin', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc, show=show, show_now=False)
    # it = conspicuity_calculation(im, sal_type='texture', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=False, show=show, show_now=False)
    # circ = conspicuity_calculation(im, sal_type='circloids', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True, show=show, show_now=False)
    blobs = conspicuity_calculation(im, sal_type='blobs', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True, show=show, show_now=False)

    # TODO: Conspicuity ... ND verze

    if show:
        plt.show()

    # saliency = skiexp.rescale_intensity(saliency, out_range=(0, 1))

    # if save_fig:
    #     save_figs(intensty, gabor, rg, by, cout, saliency, saliency_mark_max)

    # if show:
    #     plt.figure()
    #     plt.subplot(241), plt.imshow(im_orig, 'gray', interpolation='nearest'), plt.title('input')
    #     plt.subplot(242), plt.imshow(intensty, 'gray', interpolation='nearest'), plt.title('intensity')
    #     plt.subplot(243), plt.imshow(gabor, 'gray', interpolation='nearest'), plt.title('gabor')
    #     plt.subplot(244), plt.imshow(rg, 'gray', interpolation='nearest'), plt.title('rg')
    #     plt.subplot(245), plt.imshow(by, 'gray', interpolation='nearest'), plt.title('by')
    #     plt.subplot(246), plt.imshow(cout, 'gray', interpolation='nearest'), plt.title('cout')
    #     plt.subplot(247), plt.imshow(saliency, 'gray', interpolation='nearest'), plt.title('saliency')
    #     plt.subplot(248), plt.imshow(saliency_mark_max, 'gray', interpolation='nearest'), plt.title('saliency_mark_max')
    #     if show_now:
    #         plt.show()
    #
    # if return_all:
    #     return intensty, gabor, rg, by, cout, saliency, saliency_mark_max
    # else:
    #     return im_orig, im, saliency


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
    # im = cv2.cvtColor(data_s, cv2.COLOR_BAYER_GR2RGB)

    # # data_s, mask_s = tools.crop_to_bbox(data_s, mask_s)
    # im = cv2.cvtColor(data_s, cv2.COLOR_BAYER_GR2RGB)
    # # im = cv2.cvtColor(data_s, cv2.COLOR_GRAY2RGB)
    # # im = makeNormalizedColorChannels(im)
    # thresholdRatio = 10
    # threshold = im.max() / thresholdRatio
    # r, g, b = cv2.split(im)
    #
    # # plt.figure()
    # # plt.subplot(131), plt.imshow(r[120:160, 15:52], 'gray', interpolation='nearest')
    # # plt.subplot(132), plt.imshow(np.where(r[120:160, 15:52] <= threshold, 0, r[120:160, 15:52]), 'gray', interpolation='nearest')
    # # plt.subplot(133), plt.imshow(cv2.threshold(src=r[120:160, 15:52], thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)[1], 'gray', interpolation='nearest')
    # # plt.show()
    #
    # cv2.threshold(src=r, dst=r, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    # cv2.threshold(src=g, dst=g, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    # cv2.threshold(src=b, dst=b, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    # R = cv2.absdiff(r, cv2.add(g, b) / 2)  # r - (g + b) / 2
    # G = cv2.absdiff(g, cv2.add(r, b) / 2)  # g - (r + b) / 2
    # B = cv2.absdiff(b, cv2.add(g, r) / 2)  # b - (g + r) / 2
    # Y = cv2.absdiff(cv2.absdiff((cv2.add(r, g)) / 2, cv2.absdiff(r, g) / 2), b)  # (r + g) / 2 - cv2.absdiff(r, g) / 2 - b
    # im = cv2.merge((R, G, B, Y))
    # # im = cv2.merge((r, g, b))
    #
    # plt.figure()
    # plt.subplot(341), plt.imshow(g, 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.subplot(342), plt.imshow(b, 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.subplot(343), plt.imshow(cv2.add(g, b), 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.subplot(344), plt.imshow(cv2.absdiff(r, cv2.add(g, b) / 2), 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.subplot(347), plt.imshow(g + b, 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.subplot(348), plt.imshow(r - (g + b) / 2, 'gray', interpolation='nearest', vmin=0, vmax=255)
    # mean = g[np.nonzero(mask_s)].mean()
    # median = np.median(g[np.nonzero(mask_s)])
    # tum = 2 * cv2.absdiff(g, int(mean) * np.ones_like(g))
    # _, tum2 = cv2.threshold(src=r, thresh=mean * 0.9, maxval=0.0, type=cv2.THRESH_TOZERO_INV)
    # _, tum3 = cv2.threshold(src=r, thresh=median * 0.9, maxval=0.0, type=cv2.THRESH_TOZERO_INV)
    # plt.subplot(3,4,9), plt.imshow(tum, 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.subplot(3,4,10), plt.imshow(tum2, 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.subplot(3,4,11), plt.imshow(tum3, 'gray', interpolation='nearest', vmin=0, vmax=255)
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(151), plt.imshow(data_s, 'gray', interpolation='nearest'), plt.title('orig')
    # plt.subplot(152), plt.imshow(im[:, :, 0], 'gray', interpolation='nearest'), plt.title('red')
    # plt.subplot(153), plt.imshow(im[:, :, 1], 'gray', interpolation='nearest'), plt.title('green')
    # plt.subplot(154), plt.imshow(im[:, :, 2], 'gray', interpolation='nearest'), plt.title('blue')
    # plt.subplot(155), plt.imshow(im[:, :, 3], 'gray', interpolation='nearest'), plt.title('yellow')
    # # plt.subplot(155), plt.imshow(np.where(r < threshold, 0, r), 'gray', interpolation='nearest'), plt.title('yellow')
    # plt.show()

    # run(im, save_fig=False, show=True, show_now=False)
    run(data_s, mask=mask_s, smoothing=True, save_fig=False, show=True, verbose=verbose)