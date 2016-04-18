# from __future__ import division

import math
import logging
import cv2
import numpy
from scipy.ndimage.filters import maximum_filter
import scipy.stats as scista
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import skimage.exposure as skiexp
import skimage.morphology as skimor
from sklearn.cluster import KMeans
import sklearn.mixture as sklmix

import os.path
import sys
import datetime

import sys
sys.path.append('../imtools/')
from imtools import tools

if sys.version_info[0] != 2:
    raise Exception("This script was written for Python version 2.  You're running Python %s." % sys.version)

logger = logging.getLogger(__name__)

verbose = False


def _debug(msg, msgType="[INFO]"):
    if verbose:
        print '{} {} | {}'.format(msgType, msg, datetime.datetime.now())


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
        # new_s = tools.pyramid_down(scales[-1])
        new_s = cv2.pyrDown(src=scales[-1])
        if new_s is not None:
            scales.append(new_s)
        else:
            break
    levels = len(scales)
        # scales.append(cv2.pyrDown(scales[-1]))  # TODO: nahradit cv2.pyrDown vlastnim (cv2 pouziva gaussian bluring)

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
            startSize	: the base size of the Gaussian pyramit used to
                            calculate the feature maps.
        returns:
            a combined feature map.
    """
    commonWidth = startSize[0] / 2**(levels/2 - 1)
    commonHeight = startSize[1] / 2**(levels/2 - 1)
    commonSize = commonWidth, commonHeight
    logger.info("Size of conspicuity map: %s", commonSize)
    consp = N(cv2.resize(features[0][1], commonSize))
    for f in features[1:]:
        resized = N(cv2.resize(f[1], commonSize))
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


def conspicuity_intensity(im, mask=None, type='both', use_sigmoid=True, morph_proc=True, a=0.1, c=20, show=False, show_now=True):
    _debug('Running intensity conspicuity calculation ...')
    sigm_t = 0.2
    if mask is None:
        mask = np.ones_like(im)
    mean_v = im[np.nonzero(mask)].mean()
    if type == 'both':
        im_diff = np.abs(im - mean_v) * mask
    elif type == 'hypo':
        im_diff = np.abs(im - mean_v) * ((im - mean_v) < 0) * mask
    else:  # type == 'hyper'
        im_diff = np.abs(im - mean_v) * ((im - mean_v) > 0) * mask

    if use_sigmoid:
        im_sigm = (1. / (1 + (np.exp(-a * (im_diff - c))))) * mask
        im_sigm = im_sigm * (im_sigm > sigm_t)
        im_res = im_sigm.copy()
    else:
        im_res = im_diff.copy()

    if morph_proc:
        im_morph = skimor.closing(skimor.opening(im_res, selem=skimor.disk(1)))
        im_res = im_morph.copy()

    features, levels = cross_scale_diffs(im_diff, None)
    consp_int = cv2.resize(sumNormalizedFeatures(features, levels=levels), dsize=im.shape[::-1])

    if show:
        imgs = [im, im_diff, im_res, consp_int]
        titles = ['input', 'difference', 'result im', 'conspicuity']
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

    return consp_int, im_diff


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
#         plt.show()


def run(im, mask=None, save_fig=False, smoothing=False, return_all=False, show=False, show_now=True):
    """
    im ... grayscale image
    """
    show = True  # debug show

    if mask is None:
        mask = np.ones_like(im)
    else:
        im, mask = tools.crop_to_bbox(im, mask)
        mean_v = int(im[np.nonzero(mask)].mean())
        im = np.where(mask, im, mean_v)

    im_orig = im.copy()
    orig_shape = im_orig.shape[:-1]

    if smoothing:
        im = tools.smoothing(im)

    conspicuity_intensity(im, mask=mask, type='hypo', use_sigmoid=True, show=show, show_now=False)
    # conspicuity_prob_models(im, mask, show=show, show_now=False)

    # TODO: tady vypocitat ruzne conspicuity
    # TODO: Conspicuity ... blob odezvy
    # TODO: Conspicuity ... circloidy
    # TODO: Conspicuity ... textura (LBP, LIP)
    # TODO: Conspicuity ... HEQ pipeline
    # TODO: Conspicuity ... fuzzy

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
    run(data_s, mask=mask_s, smoothing=True, save_fig=False, show=True)