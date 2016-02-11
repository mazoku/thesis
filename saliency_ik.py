# from __future__ import division

# coding=utf-8
#
#!/usr/bin/python
#
# Copyright 2013 Johannes Bauer, Universitaet Hamburg
#
# This file is free software.  Do with it whatever you like.
# It comes with no warranty, explicit or implicit, whatsoever.
#
# This python script implements an early version of Itti and Koch's
# saliency model.  Specifically, it was written according to the
# information contained in the following paper:
#
#   Laurent Itti, Christof Koch, and Ernst Niebur. A model of
#   Saliency-Based visual attention for rapid scene analysis. IEEE
#   Transactions on Pattern Analysis and Machine Intelligence
#
# If you find it useful or if you have any questions, do not
# hesitate to contact me at
#   bauer at informatik dot uni dash hamburg dot de.
#
# For information on how to use this script, type
#   > python saliency.py -h
# on the command line.
#

import math
import logging
import cv2
import numpy
from scipy.ndimage.filters import maximum_filter
import matplotlib.pyplot as plt
import numpy as np

import os.path
import sys

import sys
sys.path.append('../imtools/')
from imtools import tools

if sys.version_info[0] != 2:
    raise Exception("This script was written for Python version 2.  You're running Python %s." % sys.version)

logger = logging.getLogger(__name__)



def features(image, channel, levels=9, start_size=(640,480), ):
    """
    Extracts features by down-scaling the image levels times,
    transforms the image by applying the function channel to
    each scaled version and computing the difference between
    the scaled, transformed	versions.
        image : 		the image
        channel : 		a function which transforms the image into
                        another image of the same size
        levels : 		number of scaling levels
        start_size : 	tuple.  The size of the biggest image in
                        the scaling	pyramid.  The image is first
                        scaled to that size and then scaled by half
                        levels times.  Therefore, both entries in
                        start_size must be divisible by 2^levels.
	"""
    image = channel(image)
    if image.shape != start_size:
        image = cv2.resize(image, dsize=start_size)

    scales = [image]
    for l in xrange(levels - 1):
        logger.debug("scaling at level %d", l)
        scales.append(cv2.pyrDown(scales[-1]))

    features = []
    for i in xrange(1, levels - 5):
        big = scales[i]
        for j in (3,4):
            logger.debug("computing features for levels %d and %d", i, i + j)
            small = scales[i + j]
            srcsize = small.shape[1],small.shape[0]
            dstsize = big.shape[1],big.shape[0]
            logger.debug("Shape source: %s, Shape target :%s", srcsize, dstsize)
            scaled = cv2.resize(src=small, dsize=dstsize)
            features.append(((i+1,j+1),cv2.absdiff(big, scaled)))

    return features


def intensity(image):
    """
        Converts a color image into grayscale.
        Used as `channel' argument to function `features'
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def makeGaborFilter(dims, lambd, theta, psi, sigma, gamma):
    """
        Creates a Gabor filter (an array) with parameters labmbd, theta,
        psi, sigma, and gamma of size dims.  Returns a function which
        can be passed to `features' as `channel' argument.

        In some versions of OpenCV, sizes greater than (11,11) will lead
        to segfaults (see http://code.opencv.org/issues/2644).
    """
    def xpf(i,j):
        return i*math.cos(theta) + j*math.sin(theta)

    def ypf(i,j):
        return -i*math.sin(theta) + j*math.cos(theta)

    def gabor(i,j):
        xp = xpf(i,j)
        yp = ypf(i,j)
        return math.exp(-(xp**2 + gamma**2*yp**2)/2*sigma**2) * math.cos(2*math.pi*xp/lambd + psi)

    halfwidth = dims[0]/2
    halfheight = dims[1]/2

    kernel = numpy.array([[gabor(halfwidth - i,halfheight - j) for j in range(dims[1])] for i in range(dims[1])])


    def theFilter(image):
        return cv2.filter2D(src = image, ddepth = -1, kernel = kernel, )

    return theFilter


def intensityConspicuity(image):
    """
        Creates the conspicuity map for the channel `intensity'.
    """
    # fs = features(image = im, channel = intensity)
    fs = features(image = image, channel = intensity)
    return sumNormalizedFeatures(fs)


def gaborConspicuity(image, steps):
    """
        Creates the conspicuity map for the channel `orientations'.
    """
    gaborConspicuity = numpy.zeros((60,80), numpy.uint8)
    for step in range(steps):
        theta = step * (math.pi/steps)
        gaborFilter = makeGaborFilter(dims=(10,10), lambd=2.5, theta=theta, psi=math.pi/2, sigma=2.5, gamma=.5)
        # gaborFeatures = features(image = intensity(im), channel = gaborFilter)
        gaborFeatures = features(image = intensity(image), channel = gaborFilter)
        summedFeatures = sumNormalizedFeatures(gaborFeatures)
        gaborConspicuity += N(summedFeatures)
    return gaborConspicuity


def rgConspicuity(image):
    """
        Creates the conspicuity map for the sub channel `red-green conspicuity'.
        of the color channel.
    """
    def rg(image):
        r,g,_,__ = cv2.split(image)
        return cv2.absdiff(r,g)
    fs = features(image = image, channel = rg)
    return sumNormalizedFeatures(fs)


def byConspicuity(image):
    """
        Creates the conspicuity map for the sub channel `blue-yellow conspicuity'.
        of the color channel.
    """
    def by(image):
        _,__,b,y = cv2.split(image)
        return cv2.absdiff(b,y)
    fs = features(image = image, channel = by)
    return sumNormalizedFeatures(fs)


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


def makeNormalizedColorChannels(image, thresholdRatio=10.):
    """
        Creates a version of the (3-channel color) input image in which each of
        the (4) channels is normalized.  Implements color opponencies as per
        Itti et al. (1998).
        Arguments:
            image			: input image (3 color channels)
            thresholdRatio	: the threshold below which to set all color values
                                to zero.
        Returns:
            an output image with four normalized color channels for red, green,
            blue and yellow.
    """
    intens = intensity(image)
    threshold = intens.max() / thresholdRatio
    logger.debug("Threshold: %d", threshold)
    r,g,b = cv2.split(image)
    cv2.threshold(src=r, dst=r, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=g, dst=g, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=b, dst=b, thresh=threshold, maxval=0.0, type=cv2.THRESH_TOZERO)
    R = r - (g + b) / 2
    G = g - (r + b) / 2
    B = b - (g + r) / 2
    Y = (r + g) / 2 - cv2.absdiff(r,g) / 2 - b

    # Negative values are set to zero.
    cv2.threshold(src=R, dst=R, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=G, dst=G, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=B, dst=B, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)
    cv2.threshold(src=Y, dst=Y, thresh=0., maxval=0.0, type=cv2.THRESH_TOZERO)

    image = cv2.merge((R,G,B,Y))
    return image


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


def get_var_name(var):
    var_name = [k for k, v in locals().iteritems() if v is var][0]


def save_figs(intensty, gabor, rg, by, cout, saliency, saliency_mark_max, base_name='res'):
    cv2.imwrite(base_name + '_intensity.png', intensty)
    cv2.imwrite(base_name + '_gabor.png', gabor)
    cv2.imwrite(base_name + '_rg.png', rg)
    cv2.imwrite(base_name + '_by.png', by)
    cv2.imwrite(base_name + '_cout.png', cout)
    cv2.imwrite(base_name + '_saliency.png', saliency)
    cv2.imwrite(base_name + '_saliency_mark_max.png', saliency_mark_max)


def run(im, mask=None, save_fig=False, smoothing=False, return_all=False, show=False, show_now=True):

    if mask is None:
        mask = np.ones_like(im)
        im_orig = im.copy()
    else:
        im, mask = tools.crop_to_bbox(im, mask)
        im_orig = im.copy()
        mean_v = int(im[np.nonzero(mask)].mean())
        im = np.where(mask, im, mean_v)

    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_BAYER_GR2RGB)

    im_orig = im.copy()
    orig_shape = im_orig.shape[:-1]

    if smoothing:
        im = tools.smoothing(im)

    intensty = intensityConspicuity(im)
    gabor = gaborConspicuity(im, 4)

    im = makeNormalizedColorChannels(im)
    rg = rgConspicuity(im)
    by = byConspicuity(im)
    c = rg + by
    saliency = 1./3 * (N(intensty) + N(c) + N(gabor))
    saliency_mark_max = markMaxima(saliency)
    cout = .25 * c

    intensty = cv2.resize(intensty, dsize=orig_shape[::-1]) * mask
    gabor = cv2.resize(gabor, dsize=orig_shape[::-1]) * mask
    rg = cv2.resize(rg, dsize=orig_shape[::-1]) * mask
    by = cv2.resize(by, dsize=orig_shape[::-1]) * mask
    cout = cv2.resize(cout, dsize=orig_shape[::-1]) * mask
    saliency = cv2.resize(saliency, dsize=orig_shape[::-1]) * mask
    saliency_mark_max = cv2.resize(saliency_mark_max, dsize=orig_shape[::-1])

    if save_fig:
        save_figs(intensty, gabor, rg, by, cout, saliency, saliency_mark_max)

    if show:
        plt.figure()
        plt.subplot(241), plt.imshow(im_orig, 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(242), plt.imshow(intensty, 'gray', interpolation='nearest'), plt.title('intensity')
        plt.subplot(243), plt.imshow(gabor, 'gray', interpolation='nearest'), plt.title('gabor')
        plt.subplot(244), plt.imshow(rg, 'gray', interpolation='nearest'), plt.title('rg')
        plt.subplot(245), plt.imshow(by, 'gray', interpolation='nearest'), plt.title('by')
        plt.subplot(246), plt.imshow(cout, 'gray', interpolation='nearest'), plt.title('cout')
        plt.subplot(247), plt.imshow(saliency, 'gray', interpolation='nearest'), plt.title('saliency')
        plt.subplot(248), plt.imshow(saliency_mark_max, 'gray', interpolation='nearest'), plt.title('saliency_mark_max')
        if show_now:
            plt.show()

    if return_all:
        return intensty, gabor, rg, by, cout, saliency, saliency_mark_max
    else:
        return im_orig, im, saliency


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    data_s, mask_s = tools.crop_to_bbox(data_s, mask_s)
    mean_v = int(data_s[np.nonzero(mask_s)].mean())
    data_s = np.where(mask_s, data_s, mean_v)
    # data_s *= mask_s
    im = cv2.cvtColor(data_s, cv2.COLOR_BAYER_GR2RGB)

    # run(im, save_fig=False, show=True, show_now=False)
    run(im, smoothing=True, save_fig=False, show=True)