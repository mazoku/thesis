from __future__ import division

import sys
sys.path.append('../imtools/')
from imtools import tools

from skimage import feature
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

import matplotlib.pyplot as plt

import numpy as np
import cv2


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')


def run(image, mask=None, n_points=24, radius=3):

    # # extract the histogram of Local Binary Patterns
    # lbp = feature.local_binary_pattern(im, numPoints, radius, method="uniform")
    # (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))
    #
    # # optionally normalize the histogram
    # eps = 1e-7
    # hist = hist.astype("float")
    # hist /= (hist.sum() + eps)

    radius = 3
    n_points = 8 * radius

    if mask is None:
        mask = np.ones_like(image)

    image = tools.windowing(image).astype(np.uint8)
    im_bb, mask_bb = tools.crop_to_bbox(image, mask)
    im_bb = tools.smoothing(im_bb, sliceId=0)

    lbp = local_binary_pattern(im_bb, n_points, radius, 'uniform')

    # plot histograms of LBP of textures
    # fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    # plt.gray()

    titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    # edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    # flat_labels = [0, n_points + 1]
    # flat_labels = [n_points + 1,]
    # i_14 = n_points // 4            # 1/4th of the histogram
    # i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    # corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
    #                  list(range(i_34 - w, i_34 + w + 1)))

    # label_sets = (edge_labels, flat_labels, corner_labels)
    #
    # for ax, labels in zip(ax_img, label_sets):
    #     ax.imshow(overlay_labels(image, lbp, labels))
    #
    # for ax, labels, name in zip(ax_hist, label_sets, titles):
    #     counts, _, bars = hist(ax, lbp)
    #     highlight_bars(bars, labels)
    #     ax.set_ylim(ymax=np.max(counts[:-1]))
    #     ax.set_xlim(xmax=n_points + 2)
    #     ax.set_title(name)
    #
    # ax_hist[0].set_ylabel('Percentage')
    # for ax in ax_img:
    #     ax.axis('off')
    #
    # plt.show()

    k = 1
    t = k * im_bb[np.nonzero(mask_bb)].mean()
    hypo_m = im_bb < t * mask_bb
    lbp_m = np.logical_or.reduce([lbp == each for each in flat_labels])
    hypo_lbp = np.logical_and(hypo_m, lbp_m)

    # plt.figure()
    # plt.subplot(221), plt.imshow(im_bb, 'gray'), plt.title('input')
    # plt.subplot(222), plt.imshow(label2rgb(hypo_m, image=im_bb, bg_label=0, alpha=0.5)), plt.title('input < input.mean()')
    # plt.subplot(223)
    # counts, _, bars = plt.hist(lbp.ravel(), normed=True, bins=lbp.max() + 1, range=(0, lbp.max() + 1), facecolor='0.5')
    # for i in flat_labels:
    #     bars[i].set_facecolor('r')
    # plt.gca().set_ylim(ymax=np.max(counts[:-1]))
    # plt.gca().set_xlim(xmax=n_points + 2)
    # plt.subplot(224), plt.imshow(label2rgb(hypo_lbp, image=im_bb, bg_label=0, alpha=0.5)), plt.title('hypo lbp')
    # plt.show()

    plt.figure()
    plt.subplot(131), plt.imshow(im_bb, 'gray'), plt.title('input')
    plt.subplot(132), plt.imshow(label2rgb(hypo_m, image=im_bb, bg_label=0, alpha=0.5)), plt.title('input < input.mean()')
    plt.subplot(133), plt.imshow(label2rgb(hypo_lbp, image=im_bb, bg_label=0, alpha=0.5)), plt.title('hypo lbp')

    plt.figure()
    counts, _, bars = plt.hist(lbp.ravel(), normed=True, bins=lbp.max() + 1, range=(0, lbp.max() + 1), facecolor='0.5')
    for i in flat_labels:
        bars[i].set_facecolor('r')
    plt.gca().set_ylim(ymax=np.max(counts[:-1]))
    plt.gca().set_xlim(xmax=n_points + 2)
    plt.title('lbp hist with highlighted flat prototypes')

    plt.show()



#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'  # slice_id = 17
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_185_48441644_venous_5.0_B30f-.pklz'  # slice_id = 14
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_186_49290986_venous_5.0_B30f-.pklz'  # slice_id = 5
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    # tools.show_3d(tools.windowing(data))

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