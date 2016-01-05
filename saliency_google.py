from __future__ import division

import sys

import numpy as np
import tools
import cv2

sys.path.append('../sight-spot/')
import SightSpotUtil

import matplotlib.pyplot as plt

import time


def run(image):
    print 'Calculate saliency map for test image:'
    # image = Image.open('flower.jpg')
    rgb_image = np.asarray(image, dtype='float32')
    orgb_image = SightSpotUtil.eval_orgb_image(rgb_image)

    start = time.clock()
    saliency_map = SightSpotUtil.eval_saliency_map(orgb_image, 3.0, 60.0, 'auto')
    print 'Saliency map extracted in', time.clock() - start, 'sec.'

    start = time.clock()
    heatmap_image = SightSpotUtil.eval_heatmap(saliency_map)
    print 'Heatmap extracted in', time.clock() - start, 'sec.'


    plt.figure()
    plt.subplot(131), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
    plt.subplot(132), plt.imshow(saliency_map, 'gray', interpolation='nearest'), plt.title('saliency map')
    plt.subplot(133), plt.imshow(np.asarray(heatmap_image), 'jet', interpolation='nearest'), plt.title('heat map')
    plt.show()
    # saliency_image = Image.fromarray(saliency_map * 255)
    # image.show('Source image')
    # saliency_image.show('Saliency image')
    # heatmap_image.show('Heatmap')


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
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

    run(im)