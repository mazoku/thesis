from __future__ import division

import sys
import os

import matplotlib.pyplot as plt

import cv2
import skimage.segmentation as skiseg
import skimage.morphology as skimor

import identify_liver_blob as ilb
import active_contours as ac

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools
else:
    print 'Import error in liver_segmentation.py. You need to import package imtools: https://github.com/mjirik/imtools'

if os.path.exists('/home/tomas/projects/growcut/'):
    sys.path.insert(0, '/home/tomas/projects/growcut/')
    from growcut.growcut import GrowCut
else:
    print 'Import error in liver_segmentation.py. You need to import package growcut: https://github.com/mzaoku/growcut'


def coarse_segmentation(img, show=False, show_now=True):
    seeds, peaks = tools.seeds_from_glcm_meanshift(img, show=True, show_now=False)

    gc = GrowCut(img, seeds, maxits=100, enemies_T=0.7, smooth_cell=False)
    gc.run()
    labs = gc.get_labeled_im()
    seg = skiseg.clear_border(labs[0, ...])

    print 'identifying liver blob ...',
    blob = ilb.score_data(img, seg)
    print 'done'

    if show:
        plt.figure()
        plt.suptitle('input | seeds | segmentation | blob')
        plt.subplot(141), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(142), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(143), plt.imshow(seg, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(144), plt.imshow(blob, 'gray', interpolation='nearest'), plt.axis('off')

        if show_now:
            plt.show()

    return blob


def segmentation_refinement(data, mask, smoothing=True, save_fig=False, show=False, show_now=True):
    method = 'morphsnakes'
    params = {'alpha': 100, 'sigma': 1, 'smoothing_ls': 1, 'threshold': 0.6, 'balloon': 1, 'max_iters': 1000,
              'scale': 0.5, 'init_scale': 0.25,
              'smoothing': smoothing, 'save_fig': save_fig, 'show': False, 'show_now': show_now}
    im, mask, seg = ac.run(data, mask=mask, method=method, **params)

    if show:
        plt.figure()
        plt.suptitle('input | initialization | segmentation')
        plt.subplot(131), plt.imshow(data, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(132), plt.imshow(mask, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(133), plt.imshow(seg, 'gray', interpolation='nearest'), plt.axis('off')
        if show_now:
            plt.show()

    return seg

################################################################################
################################################################################
if __name__ == '__main__':
    data_fname = '/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/input.png'
    img = cv2.imread(data_fname, 0)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    blob = coarse_segmentation(img, show=True, show_now=False)
    blob_init = blob.copy()
    blob = skimor.remove_small_holes(blob, min_size=2000, connectivity=1)
    blob = skimor.binary_opening(blob, selem=skimor.disk(1))
    blob = skimor.binary_erosion(blob, selem=skimor.disk(1))

    # plt.figure()
    # plt.subplot(141), plt.imshow(blob, 'gray', interpolation='nearest')
    # plt.subplot(142), plt.imshow(blob_hf, 'gray', interpolation='nearest')
    # plt.subplot(143), plt.imshow(blob_o, 'gray', interpolation='nearest')
    # plt.subplot(144), plt.imshow(blob_e, 'gray', interpolation='nearest')
    # plt.show()

    seg = segmentation_refinement(img, blob, show=True, show_now=False)

    tools.visualize_seg(img, blob_init, title='growcut init', show_now=False)
    tools.visualize_seg(img, seg, title='morph snakes', show_now=False)

    plt.show()