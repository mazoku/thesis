from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('../imtools/')
from imtools import tools

import sys
import os

import saliency_akisato as salaki
import saliency_google as salgoo
import saliency_ik as salik
import saliency_mayo as salmay

AKISATO = 'akisato'
GOOGLE = 'google'
IK = 'ik'
MAYO = 'mayo'


def saliency_map(image, mask, method, layer_id, smooth=True, show=False, show_now=True, save_fig=False):
    if method == AKISATO:
        im_orig, img_calc, saliency = salaki.run(image, mask, smoothing=smooth, show=False)
    elif method == GOOGLE:
        im_orig, img_calc, saliency = salgoo.run(image, mask, smoothing=smooth, show=False)
    elif method == IK:
        # intensty, gabor, rg, by, cout, saliency, saliency_mark_max = salik.run(im, return_all=True, smoothing=smooth, save_fig=False, show=True)
        im_orig, img_calc, saliency = salik.run(image, return_all=False, smoothing=smooth, save_fig=False, show=True)
    elif method == MAYO:
        im_orig, img_calc, saliency = salmay.run(image, mask=mask, smoothing=smooth, show=False)

    return saliency


def run(image, mask, pyr_scale, method, show=False, show_now=True, save_fig=False):
    fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/saliency/%s/' % method
    if save_fig:
        dirs = fig_dir.split('/')
        for i in range(2, len(dirs)):
            subdir = '/'.join(dirs[:i])
            if not os.path.exists(subdir):
                os.mkdir(subdir)
        # if not os.path.exists(fig_dir):
        #     os.mkdir(fig_dir)

    image, mask = tools.crop_to_bbox(image, mask)
    image = tools.smoothing(image, sliceId=0)

    pyr_saliencies = []
    # pyr_survs = []
    # pyr_titles = []
    pyr_imgs = []
    pyr_masks = []
    for layer_id, (im_pyr, mask_pyr) in enumerate(zip(tools.pyramid(image, scale=pyr_scale, inter=cv2.INTER_NEAREST),
                                                      tools.pyramid(mask, scale=pyr_scale, inter=cv2.INTER_NEAREST))):
        saliency = saliency_map(im_pyr, mask_pyr, method, layer_id=layer_id, show=show, show_now=show_now, save_fig=save_fig)

        pyr_saliencies.append(saliency)
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

    n_layers = len(pyr_imgs)

    # survival overall
    survival = np.zeros(image.shape)
    for sal in pyr_saliencies:
        survival += cv2.resize(sal, image.shape[::-1])

    if show or save_fig:
        fig_sal_layers = plt.figure(figsize=(24, 14))
        for layer_id, sal in enumerate(pyr_saliencies):
            plt.subplot(1, n_layers, layer_id + 1)
            plt.imshow(pyr_saliencies[layer_id], 'jet', interpolation='nearest')
            plt.title('%s, layer #%i'% (method, layer_id + 1))
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(cax=cax)

        fig_urv_overall = plt.figure(figsize=(24, 14))
        plt.subplot(121)
        plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(122)
        plt.imshow(survival, 'jet', interpolation='nearest'), plt.title('%s, survival' % method)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)

        if save_fig:
            fig_sal_layers.savefig(os.path.join(fig_dir, '%s_layers.png' % method), dpi=100, bbox_inches='tight', pad_inches=0)
            fig_urv_overall.savefig(os.path.join(fig_dir, '%s_surv_overall.png' % method), dpi=100, bbox_inches='tight', pad_inches=0)

        if show_now:
            plt.show()

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    show = True
    show_now = False
    save_fig = True
    pyr_scale = 1.5
    method = AKISATO

    run(data_s, mask_s, pyr_scale=pyr_scale, method=method, show=show, show_now=show_now, save_fig=save_fig)

    if show:
        plt.show()
    # # AKISATO ------------------------------------
    # aki_im_o, aki_img, aki_saliency = salaki.run(data_s, mask_s, show=False)
    # aki_im_o_s, aki_img_s, aki_saliency_s = salaki.run(data_s, mask_s, smoothing=True, show=False)
    #
    # plt.figure()
    # plt.subplot(141), plt.imshow(aki_im_o, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(142), plt.imshow(aki_saliency, 'gray', interpolation='nearest'), plt.title('akisato saliency')
    # plt.subplot(143), plt.imshow(aki_img_s, 'gray', interpolation='nearest'), plt.title('smoothed')
    # plt.subplot(144), plt.imshow(aki_saliency_s, 'gray', interpolation='nearest'), plt.title('akisato saliency from smoothed')
    #
    # # GOOGLE -------------------------------
    # google_im_o, google_img, google_saliency = salgoo.run(data_s, mask_s, show=False)
    # google_im_o_s, google_img_s, google_saliency_s = salgoo.run(data_s, mask_s, smoothing=True, show=False)
    #
    # plt.figure()
    # plt.subplot(141), plt.imshow(google_im_o, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(142), plt.imshow(google_saliency, 'gray', interpolation='nearest'), plt.title('google saliency')
    # plt.subplot(143), plt.imshow(google_im_o_s, 'gray', interpolation='nearest'), plt.title('smoothed')
    # plt.subplot(144), plt.imshow(google_saliency_s, 'gray', interpolation='nearest'), plt.title('google saliency from smoothed')
    #
    # # ITTY-KOCH -----------------------------
    # data_bb, mask_bb = tools.crop_to_bbox(data_s, mask_s)
    # mean_v = int(data_bb[np.nonzero(mask_bb)].mean())
    # data_bb = np.where(mask_bb, data_bb, mean_v)
    # im = cv2.cvtColor(data_bb, cv2.COLOR_BAYER_GR2RGB)
    #
    # ik_intensty, ik_gabor, ik_rg, ik_by, ik_cout, ik_saliency, ik_saliency_mark_max = salik.run(im, return_all=True, smoothing=True, save_fig=False, show=True)
    #
    # plt.figure()
    # plt.subplot(241), plt.imshow(data_bb, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(242), plt.imshow(ik_intensty, 'gray', interpolation='nearest'), plt.title('ik intensity')
    # plt.subplot(243), plt.imshow(ik_gabor, 'gray', interpolation='nearest'), plt.title('ik gabor')
    # plt.subplot(244), plt.imshow(ik_rg, 'gray', interpolation='nearest'), plt.title('ik rg')
    # plt.subplot(245), plt.imshow(ik_by, 'gray', interpolation='nearest'), plt.title('ik by')
    # plt.subplot(246), plt.imshow(ik_cout, 'gray', interpolation='nearest'), plt.title('ik cout')
    # plt.subplot(247), plt.imshow(ik_saliency, 'gray', interpolation='nearest'), plt.title('ik saliency')
    # plt.subplot(248), plt.imshow(ik_saliency_mark_max, 'gray', interpolation='nearest'), plt.title('ik saliency_mark_max')
    #
    # # MAYO -----------------------------------
    # mayo_im_o, mayo_img, mayo_saliency = salmay.run(data_s, mask=mask_s, smoothing=False, show=False)
    # mayo_im_o_s, mayo_img_s, mayo_saliency_s = salmay.run(data_s, mask=mask_s, smoothing=True, show=False)
    #
    # plt.figure()
    # plt.subplot(141), plt.imshow(mayo_im_o, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(142), plt.imshow(mayo_saliency, 'gray', interpolation='nearest'), plt.title('google saliency')
    # plt.subplot(143), plt.imshow(mayo_im_o_s, 'gray', interpolation='nearest'), plt.title('smoothed')
    # plt.subplot(144), plt.imshow(mayo_saliency_s, 'gray', interpolation='nearest'), plt.title('mayo saliency from smoothed')
    #
    # plt.show()