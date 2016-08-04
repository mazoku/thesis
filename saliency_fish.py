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
from skimage import img_as_float
from sklearn.cluster import KMeans
import sklearn.mixture as sklmix

import os.path
import sys
import datetime

# for conspicuity calculations
import blobs
import circloids
import lbp
import sliding_windows
import he_pipeline

import sys
if os.path.exists('../imtools/'):
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
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

    # rescaling to <0, 1>
    im_int = skiexp.rescale_intensity(im_int.astype(np.float), out_range=(0, 1))

    # using sigmoid
    sigm_t = 0.2
    if use_sigmoid:
        c = mean_v
        im_sigm = tools.sigmoid(im_int, mask, a=a, c=c, sigm_t=sigm_t)
        # im_sigm = (1. / (1 + (np.exp(-a * (im_diff - c))))) * mask
        # im_sigm = im_sigm * (im_sigm > sigm_t)
        im_res = im_sigm.copy()
    else:
        im_res = im_int.copy()

    # morphological postprocessing
    if morph_proc:
        im_morph = skimor.closing(skimor.opening(im_res, selem=skimor.disk(3)))
        im_res = im_morph.copy()

    im_res = skiexp.rescale_intensity(im_res.astype(np.float), out_range=np.float32)

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

    # rescaling to <0, 1>
    im_int = skiexp.rescale_intensity(im_int.astype(np.float), out_range=(0, 1))

    # using sigmoid
    sigm_t = 0.2
    if use_sigmoid:
        c = mean_v
        im_sigm = tools.sigmoid(im_int, mask, a=a, c=c, sigm_t=sigm_t)
        # im_sigm = (1. / (1 + (np.exp(-a * (im_diff - c))))) * mask
        # im_sigm = im_sigm * (im_sigm > sigm_t)
        im_res = im_sigm.copy()
    else:
        im_res = im_int.copy()

    # morphological postprocessing
    if morph_proc:
        im_morph = skimor.closing(skimor.opening(im_res, selem=skimor.disk(3)))
        im_res = im_morph.copy()

    im_res = skiexp.rescale_intensity(im_res.astype(np.float), out_range=np.float32)

    return im_res


def conspicuity_int_glcm(im, mask=None, use_sigmoid=False, morph_proc=True, type='hypo', a=3):
    if mask is None:
        mask = np.ones_like(im)

    if im.max() <= 1:
        im = skiexp.rescale_intensity(im, (0, 1), (0, 255)).astype(np.int)

    gcm = tools.graycomatrix_3D(im, mask=mask)

    # thresholding graycomatrix (GCM)
    c_t = 5
    thresh = c_t * np.mean(gcm)
    gcm_t = gcm > thresh
    gcm_to = skimor.binary_opening(gcm_t, selem=skimor.disk(3))

    rvs = tools.analyze_glcm(gcm_to)

    # filtering
    rvs = sorted(rvs, key=lambda rv: rv.mean())
    im_int = rvs[0].pdf(im)
    mean_v = rvs[0].mean()
    a = 20

    # rescaling to <0, 1>
    im_int = skiexp.rescale_intensity(im_int.astype(np.float), out_range=(0, 1))

    plt.figure()
    plt.subplot(131), plt.imshow(cv2.resize((im * mask).astype(np.float32), (115, 215)), 'gray'), plt.axis('off')
    plt.subplot(132), plt.imshow(gcm_to, 'gray'), plt.axis('off')
    plt.subplot(133), plt.imshow(cv2.resize(im_int.astype(np.float32), (115, 215)), 'jet'), plt.axis('off')
    plt.show()

    # using sigmoid
    sigm_t = 0.2
    if use_sigmoid:
        c = mean_v / 255
        im_sigm = tools.sigmoid(im_int, mask, a=a, c=c, sigm_t=sigm_t)
        im_res = im_sigm.copy()
    else:
        im_res = im_int.copy()

    # morphological postprocessing
    if morph_proc:
        im_morph = skimor.closing(skimor.opening(im_res, selem=skimor.disk(3)))
        im_res = im_morph.copy()

    im_res = skiexp.rescale_intensity(im_res.astype(np.float), out_range=np.float32)

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
        m = win < liver_peak
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

    # plt.figure()
    # plt.imshow(im_int, 'gray')
    # plt.show()

    # rescaling to <0, 1>
    im_int = skiexp.rescale_intensity(im_int.astype(np.float), out_range=(0, 1))

    # using sigmoid
    sigm_t = 0.2
    if use_sigmoid:
        c = mean_v / 255
        im_sigm = tools.sigmoid(im_int, mask, a=a, c=c, sigm_t=sigm_t)
        im_res = im_sigm.copy()
    else:
        im_res = im_int.copy()

    # morphological postprocessing
    if morph_proc:
        im_morph = skimor.closing(skimor.opening(im_res, selem=skimor.disk(3)))
        im_res = im_morph.copy()

    im_res = skiexp.rescale_intensity(im_res.astype(np.float), out_range=np.float32)

    return im_res


def conspicuity_intensity(im, mask=None, type='both', int_type='diff', use_sigmoid=True, morph_proc=True, a=0.1, c=20, show=False, show_now=True):
    _debug('Running intensity conspicuity calculation, type %s ...' % int_type)

    sigm_t = 0.2

    # intensity difference
    if int_type == 'diff':
        if mask is None:
            mask = np.ones_like(im)
        mean_v = im[np.nonzero(mask)].mean()
        if type == 'both':
            im_int = np.abs(im - mean_v) * mask
        elif type == 'hypo':
            im_int = np.abs(im - mean_v) * ((im - mean_v) < 0) * mask
        else:  # type == 'hyper'
            im_int = np.abs(im - mean_v) * ((im - mean_v) > 0) * mask
        a = 3

    # hist analysis
    elif int_type == 'hist':
        class1, rv = tools.dominant_class(im, roi=mask)
        field = skimor.binary_closing(class1, selem=skimor.disk(3))
        field = skimor.binary_opening(field, selem=skimor.disk(3))
        field = (1 - field) * mask
        im_int = scindimor.distance_transform_edt(field, return_distances=True)
        mean_v = rv.mean()
        a = 3

        # if show:
        #     plt.figure()
        #     plt.subplot(141), plt.imshow(im, 'gray'), plt.title('input')
        #     plt.subplot(142), plt.imshow(class1, 'gray'), plt.title('class1')
        #     plt.subplot(143), plt.imshow(field, 'gray'), plt.title('field')
        #     plt.subplot(144), plt.imshow(im_int, 'gray'), plt.title('dist')
        #     if show_now:
        #         plt.show()

    # GLCM analysis
    elif int_type == 'glcm':
        gcm = tools.graycomatrix_3D(im, mask=mask)

        # thresholding graycomatrix (GCM)
        c_t = 5
        thresh = c_t * np.mean(gcm)
        gcm_t = gcm > thresh
        gcm_to = skimor.binary_opening(gcm_t, selem=skimor.disk(3))

        # plt.figure()
        # plt.subplot(131), plt.imshow(gcm, 'gray')
        # plt.subplot(132), plt.imshow(gcm_t, 'gray')
        # plt.subplot(133), plt.imshow(gcm_to, 'gray')
        #
        # plt.figure()
        # plt.subplot(121), plt.imshow(im, 'gray')
        # plt.subplot(122)
        # plt.imshow(gcm, 'gray', vmax=100)
        plt.show()
        rvs = tools.analyze_gcm(gcm_to)

        # filtering
        rvs = sorted(rvs, key=lambda rv: rv.mean())
        im_int = rvs[0].pdf(im)
        mean_v = rvs[0].mean()
        a = 20

        # visualization of RV
        # probs = []
        # for rv in rvs:
        #     tmp = rv.pdf(im)
        #     probs.append(tmp)
        # plt.figure()
        # plt.subplot(100 + 10 * (len(probs) + 1) + 1)
        # plt.imshow(im, 'gray'), plt.title('input')
        # for i, im in enumerate(probs):
        #     plt.subplot(100 + 10 * (len(probs) + 1) + i + 2)
        #     plt.imshow(im, 'gray')
        #     plt.title('rv #%i' % (i + 1))
        # plt.show()
    elif int_type == 'sliwin':
        pass
    else:
        raise ValueError('Wrong type of intensity conspicuity.')

    # POSTPROCESSING
    im_int = skiexp.rescale_intensity(im_int.astype(np.float), out_range=(0, 1))

    if use_sigmoid:
        c = mean_v / 255
        im_sigm = tools.sigmoid(im_int, mask, a=a, c=c, sigm_t=sigm_t)
        # im_sigm = (1. / (1 + (np.exp(-a * (im_diff - c))))) * mask
        # im_sigm = im_sigm * (im_sigm > sigm_t)
        im_res = im_sigm.copy()
    else:
        im_res = im_int.copy()

    if morph_proc:
        im_morph = skimor.closing(skimor.opening(im_res, selem=skimor.disk(1)))
        im_res = im_morph.copy()

    im_res = skiexp.rescale_intensity(im_res.astype(np.float), out_range=np.float32)
    features, levels = cross_scale_diffs(im_res, channel=None)
    consp_int = cv2.resize(sumNormalizedFeatures(features, levels=levels), dsize=im.shape[::-1])

    plt.figure()
    n_cols = np.ceil(len(features) / 2)
    for i, f in enumerate(features):
        plt.subplot(200 + 10 * n_cols + i + 1)
        plt.imshow(cv2.resize(f[1], im.shape[::-1]), 'gray')
        plt.title('{}'.format(f[0]))
    plt.show()

    if show:
        imgs = [im, im_int, im_res, consp_int]
        titles = ['input', 'int %s' % int_type, 'result im', 'conspicuity']
        if morph_proc:
            imgs.insert(2, im_morph)
            titles.insert(2, 'morphology')
        if use_sigmoid:
            imgs.insert(2, im_sigm)
            titles.insert(2, 'sigmoid')
        n_imgs = len(imgs)
        plt.figure()
        for i, (im, tit) in enumerate(zip(imgs, titles)):
            plt.subplot(1, n_imgs, i + 1)
            plt.imshow(im, 'gray', interpolation='nearest')
            plt.title(tit)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(cax=cax)
        if show_now:
            plt.show()

    return consp_int#, im_int


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


def conspicuity_blobs(im, mask, pyr_scale=2, show=False, show_now=True):
    # _debug('Running blob response conspicuity calculation ...')
    blob_surv, survs = blobs.run_all(im, mask, pyr_scale=pyr_scale, show=False, show_now=False, verbose=verbose)

    if show:
        imgs = [im, blob_surv] + [s[0] for s in survs]
        titles = ['input', 'blob surv'] + [s[2] for s in survs]
        n_imgs = len(imgs)
        cmaps = ['gray'] + (n_imgs - 1) * ['jet']
        for i, (im, tit, cmap) in enumerate(zip(imgs, titles, cmaps)):
            plt.subplot(1, n_imgs, i + 1)
            plt.imshow(im, cmap=cmap, interpolation='nearest')
            plt.title(tit)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(cax=cax)
        if show_now:
            plt.show()

    return blob_surv


def conspicuity_circloids(im, mask, pyr_scale=2, show=False, show_now=True):
    circ_surv, circ_surv_layers, circ_surv_masks = circloids.run(im, mask, pyr_scale=pyr_scale, show=False, show_now=False, verbose=verbose)

    if show:
        imgs = [im, circ_surv] + circ_surv_masks
        titles = ['input', 'circ surv'] + ['mask #%i' % (i+1) for i in range(len(circ_surv_masks))]
        n_imgs = len(imgs)
        cmaps = ['gray'] + (n_imgs - 1) * ['jet']
        for i, (im, tit, cmap) in enumerate(zip(imgs, titles, cmaps)):
            plt.subplot(1, n_imgs, i + 1)
            plt.imshow(im, cmap=cmap, interpolation='nearest')
            plt.title(tit)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(cax=cax)
        if show_now:
            plt.show()

    return circ_surv


def conspicuity_texture(im, mask, pyr_scale=2, show=False, show_now=True):
    lbp_surv, _ = lbp.run(im, mask, n_points=24, radius=3, lbp_type='flat', blob_type='hypo', pyr_scale=1.5,
                          show=show, show_now=show_now, save_fig=False, verbose=verbose)

    return lbp_surv


def conspicuity_sliding_window(im, mask, pyr_scale=2., min_pyr_size=(20, 20), show=False, show_now=True):
    sliwin_surv, _ = sliding_windows.run(im, mask, pyr_scale=pyr_scale, min_pyr_size=min_pyr_size,
                                         show=show, show_now=show_now, verbose=verbose)

    return sliwin_surv


def conspicuity_he_pipeline(im, mask, proc, pyr_scale=1.5, min_pyr_size=(20, 20), show=False, show_now=True):
    heep_surv = he_pipeline.run(im, mask, proc=proc, pyr_scale=pyr_scale, min_pyr_size=min_pyr_size,
                                show=show, show_now=show_now, verbose=verbose)

    return heep_surv


def conspicuity_calculation(img, mask=None, consp_fcn=None, n_levels=9, use_sigmoid=False, morph_proc=True,
                            type='hypo', calc_features=True, return_features=False, show=False, show_now=True):
    _debug('Running intensity conspicuity calculation, fcn: %s ...' % str(consp_fcn.__name__))
    if mask is None:
        mask = np.ones_like(img)
    mask = mask.astype(np.float)

    img_0 = cv2.resize(img, (640, 480))
    mask_0 = cv2.resize(mask, (640, 480))

    # im_pyramid = [img_0,]
    # mask_pyramid = [mask_0, ]
    im_pyramid = []
    mask_pyramid = []
    consp_pyramid = []

    im_p = img_0
    mask_p = mask_0
    for l in xrange(n_levels - 1):
        params = {'mask': mask_p, 'use_sigmoid': use_sigmoid, 'morph_proc': morph_proc, 'type': type}
        # calculating conspicuity map if a consp_fcn was provided
        if consp_fcn is not None:
            consp_map = consp_fcn(im_p, **params)
        else:
            consp_map = im_p.copy()

        # append data to pyramids
        im_pyramid.append(im_p)
        mask_pyramid.append(mask_p)
        consp_pyramid.append(consp_map)

        # pyr-down data
        logger.debug("scaling at level %d", l)
        im_p = tools.pyramid_down(im_pyramid[-1], scale=4. / 3, smooth=False)
        mask_p = tools.pyramid_down(mask_pyramid[-1], scale=4. / 3, smooth=False)

        # break the loop if the image can't be scaled down any further
        if im_p is None:
            break

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
                logger.debug("computing features for levels %d and %d", i, i + j)
                small = consp_pyramid[i + j]
                srcsize = small.shape[1], small.shape[0]
                dstsize = big.shape[1], big.shape[0]
                logger.debug("Shape source: %s, Shape target :%s", srcsize, dstsize)
                scaled = cv2.resize(src=small, dsize=dstsize)
                f = cv2.absdiff(big, scaled)
                # f = cv2.resize(f, (img.shape[1], img.shape[0]))
                features.append(((i + 1, j + 1), f))
    else:
        features = [((0, 0), im) for im in consp_pyramid]

    plt.figure()
    n_cols = np.ceil(len(features) / 2)
    for i, f in enumerate(features):
        plt.subplot(200 + 10 * n_cols + i + 1)
        plt.imshow(cv2.resize(f[1], img.shape[::-1]), 'jet')
        plt.axis('off')
        # plt.title('{}'.format(f[0]))
    # plt.show()

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
    # id = conspicuity_calculation(im, consp_fcn=conspicuity_int_diff, mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc, show=show, show_now=False)
    # id = conspicuity_calculation(im, consp_fcn=conspicuity_int_hist, mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc, show=show, show_now=False)
    # id = conspicuity_calculation(im, consp_fcn=conspicuity_int_glcm, mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc, show=show, show_now=False)
    # isw = conspicuity_calculation(im, consp_fcn=conspicuity_int_sliwin, mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc, show=show, show_now=False)

    # blobs
    # consp_blobs = conspicuity_blobs(im, mask=mask, show=show, show_now=False)

    # shape
    # consp_circ = conspicuity_circloids(im, mask=mask, show=show, show_now=False)

    # texture
    consp_texture = conspicuity_texture(im, mask=mask, show=show, show_now=False)

    # consp_heep = conspicuity_he_pipeline(im, mask=mask, proc=['smo', 'equ', 'clo', 'con'], pyr_scale=1.5, show=show, show_now=False)

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