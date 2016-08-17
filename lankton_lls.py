from __future__ import division


# import matlab.engine


import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import os

import scipy.ndimage.filters as scindifil
import skimage.transform as skitra
import skimage.exposure as skiexp
import skimage.io as skiio
from skimage import img_as_ubyte
import skimage.segmentation as skiseg
import skimage.color as skicol

if os.path.exists('../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


import matlab.engine
import matlab


def run(im, init_mask, method, slice=0, max_iter=1000, rad=20, alpha=0.1, energy_type=2, display=False, show=False, show_now=True):
    # print 'starting matlab engine ...',
    eng = matlab.engine.start_matlab()

    im_matlab = matlab.uint8(im.tolist())
    mask_matlab = matlab.logical(init_mask.tolist())

    # plt.figure()
    # plt.subplot(221), plt.imshow(im, 'gray')
    # plt.subplot(222), plt.imshow(im_matlab, 'gray')
    # plt.subplot(223), plt.imshow(init_mask, 'gray')
    # plt.subplot(224), plt.imshow(mask_matlab, 'gray')
    # plt.show()

    # print 'segmenting ...',
    if method == 'lls':
        eng.addpath('localized_seg')
        seg = eng.localized_seg(im_matlab, mask_matlab, max_iter, float(rad), alpha, energy_type, display, nargout=1)
    elif method == 'sfm':
        eng.addpath('sfm_chanvese_demo')
        # eng.sfm_local_demo(nargout=0)
        seg = eng.sfm_local_chanvese(im_matlab, mask_matlab, max_iter, alpha, float(rad), display, nargout=1)
    else:
        seg = init_mask.copy()
    seg = np.array(seg._data).reshape(seg._size[::-1]).T
    eng.quit()

    if show:
        im_vis = im if im.ndim ==2 else im[slice,...]
        init_mask_vis = init_mask if init_mask.ndim ==2 else init_mask[slice,...]
        seg_vis = seg if seg.ndim == 2 else seg[slice,...]
        mask_bounds = skiseg.mark_boundaries(im_vis, init_mask_vis, color=(1, 0, 0), mode='thick')
        seg_over = skicol.label2rgb(seg_vis, im_vis, colors=['red', 'green', 'blue'], bg_label=0)
        seg_bounds = skiseg.mark_boundaries(im_vis, seg_vis, color=(1, 0, 0), mode='thick')

        plt.figure()
        plt.subplot(231), plt.imshow(im_vis, 'gray'), plt.title('input')
        plt.subplot(232), plt.imshow(init_mask_vis, 'gray'), plt.title('init mask')
        plt.subplot(233), plt.imshow(mask_bounds, 'gray'), plt.title('init mask')
        plt.subplot(234), plt.imshow(seg_vis, 'gray'), plt.title('segmentation')
        plt.subplot(235), plt.imshow(seg_over, 'gray'), plt.title('segmentation')
        plt.subplot(236), plt.imshow(seg_bounds, 'gray'), plt.title('segmentation')

        if show_now:
            plt.show()

    return seg


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    im = skiio.imread('localized_seg/monkey.png', as_grey=True)

    mask = np.zeros(im.shape[:2], dtype=np.bool)
    mask[37:213, 89:227] = 1

    im = skitra.rescale(im, scale=0.5)
    # im = skiexp.rescale_intensity(im, in_range=(0, 1), out_range=np.int8)
    im = img_as_ubyte(im)
    mask = skitra.rescale(mask, scale=0.5, preserve_range=True).astype(np.bool)

    run(im, mask, method='sfm')