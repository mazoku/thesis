import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import skimage.exposure as skiexp
import skimage.filters as skifil

import cv2

if os.path.exists('../imtools/'):
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

def hist_analysis(img):
    edge = skifil.scharr(img)
    edge = skiexp.rescale_intensity(edge, out_range=(0, 1))
    edge_t = 0.1
    pts = img[np.nonzero(edge < edge_t)]
    pts = pts[pts > 0]
    pts = pts[pts < 255]
    # plt.figure()
    # plt.subplot(141), plt.imshow(img, 'gray')
    # plt.subplot(142), plt.imshow(edge, 'gray'), plt.colorbar()
    # plt.subplot(143), plt.imshow(edge < 0.1, 'gray')
    # plt.subplot(144), plt.imshow(edge < 0.2, 'gray')
    # plt.show()
    hist, bins = skiexp.histogram(pts)

    # windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'gaussian']
    # hists = []
    # for w in windows:
    #     hist_tmp = tools.hist_smoothing(bins, hist, window=w)
    #     hists.append(hist_tmp)
    #
    # plt.figure()
    # for i, (h, w) in enumerate(zip(hists, windows)):
    #     plt.subplot(321 + i)
    #     plt.fill(bins, h, 'b', alpha=1)
    #     # plt.plot(bins, h, 'b', alpha=1)
    #     plt.title(w)
    # plt.show()

    win = 'hanning'
    hist_s = tools.hist_smoothing(bins, hist, window=win)
    inds = tools.peak_in_hist(bins, hist_s)

    plt.figure()
    plt.plot(bins, hist, 'r', linewidth=2)
    plt.fill(bins, hist_s, 'b', alpha=1)
    for i in inds:
        plt.plot(bins[i], hist_s[i], 'go', markersize=15)
    plt.show()

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fname = '/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/inputImg.png'
    img = cv2.imread(fname, 0)

    hist_analysis(img)