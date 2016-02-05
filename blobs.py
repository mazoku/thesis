from __future__ import division
# Tady je to zamotany - u circloidu jsem mel nekolik masek - da se rict, ze maska byl jeden parametr s celkem 5 hodnotami.
#   U blobu mam nekolik parametru s nekolika hodnotami.

import sys
sys.path.append('../imtools/')
from imtools import tools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import skimage.exposure as skiexp
import skimage.feature as skifea
import scipy.stats as scista

import cv2

import copy
import math
import os


# constants
BLOB_DOG = 'dog'
BLOB_LOG = 'log'
BLOB_DOH = 'doh'
BLOB_CV = 'cv'

def check_blob_intensity(image, intensity, mask=None, show=False, show_now=True):
    if mask is None:
        mask = np.ones_like(image)
    if intensity in ['bright', 'light', 'white']:
        im = image.copy() * mask
    elif intensity in ['dark', 'black']:
        im = (image.copy().max() - image) * mask
    else:
        raise ValueError('Wrong blob intensity.')
    if show:
        plt.figure()
        plt.subplot(121), plt.imshow(image, 'gray'), plt.title('input')
        plt.subplot(122), plt.imshow(im, 'gray'), plt.title('output')
        if show_now:
            plt.show()
    return im


def dog(image, mask=None, intensity='bright', min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.1, overlap=0.5):
    if mask is None:
        mask = np.ones_like(image)
    im = check_blob_intensity(image, intensity, show=False)
    try:
        blobs = skifea.blob_dog(im, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio,
                                threshold=threshold, overlap=overlap)
    except:
        return []
    blobs = [x for x in blobs if mask[x[0], x[1]]]
    blobs = np.array(blobs)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    return blobs


def log(image, mask=None, intensity='bright', min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.1, overlap=0.5, log_scale=False):
    if mask is None:
        mask = np.ones_like(image)
    im = check_blob_intensity(image, intensity)
    try:
        blobs = skifea.blob_log(im, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                                threshold=threshold, overlap=overlap, log_scale=log_scale)
    except:
        return []
    blobs = [x for x in blobs if mask[x[0], x[1]]]
    blobs = np.array(blobs)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    return blobs


def doh(image, mask=None, intensity='bright', min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False):
    if mask is None:
        mask = np.ones_like(image)
    im = check_blob_intensity(image, intensity)
    try:
        blobs = skifea.blob_doh(im, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                                threshold=threshold, overlap=overlap, log_scale=log_scale)
    except:
        return []
    blobs = [x for x in blobs if mask[x[0], x[1]]]
    blobs = np.array(blobs)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    return blobs


def cv_blobs(image, mask=None, min_threshold=10, max_threshold=255, min_area=1, min_circularity=0, min_convexity=0, min_inertia=0):
    if mask is None:
        mask = np.ones_like(image)
    try:
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.filterByArea = True
        params.minArea = min_area
        if min_circularity > 0:
            params.filterByCircularity = True
            params.minCircularity = min_circularity
        else:
            params.filterByCircularity = False
        if min_convexity > 0:
            params.filterByConvexity = True
            params.minConvexity = min_convexity
        else:
            params.filterByConvexity = False
        if min_inertia > 0:
            params.filterByInertia = True
            params.minInertiaRatio = min_inertia
        else:
            params.filterByInertia = False

        detector = cv2.SimpleBlobDetector(params)
        kp = detector.detect(image)
        # im_kp = cv2.drawKeypoints(image, kp, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.figure()
        # plt.imshow(im_kp, 'gray')
        # plt.show()
        blobs = [(int(p.pt[1]), int(p.pt[0]), max(1, int(np.ceil(p.size / 2.)))) for p in kp]
    except:
        return []
    blobs = [x for x in blobs if mask[x[0], x[1]]]
    blobs = np.array(blobs)
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    return blobs


def save_figure(image, mask, blobs, fname, blob_name, param_name, param_vals, show=False, show_now=True, max_rows=3, max_cols=5):
    n_images = len(blobs)
    if max_cols < n_images:
        n_cols = max_cols
        n_rows = math.ceil(n_images / max_cols)
    else:
        n_cols = n_images
        n_rows = 1

    dirs = fname.split('/')
    dir_name = os.path.curdir
    for i in range(len(dirs)):
        dir_name = os.path.join(dir_name, dirs[i])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    fig = plt.figure(figsize=(20, n_rows * 5))
    # fig = plt.figure()
    for i, p_blobs in enumerate(blobs):
        # plt.subplot(1, len(blobs), i + 1)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image, 'gray', vmin=0, vmax=255, interpolation='nearest')
        plt.title('%s, %s=%.2f' % (blob_name, param_name, param_vals[i]))
        for blob in p_blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='m', linewidth=2, fill=False)
            plt.gca().add_patch(c)
    fig.savefig(os.path.join(fname, '%s_%s.png' % (blob_name, param_name)), bbox_inches='tight', pad_inches=0)
    if show:
        if show_now:
            plt.show()
    else:
        plt.close(fig)


def calc_survival_fcn(blobs, mask, show=False, show_now=True):
    surv_im = np.zeros(mask.shape)
    n_imgs = len(blobs)
    surv_k = 1. / n_imgs
    for b_im in blobs:
        for y, x, r in b_im:
            tmp = np.zeros_like(surv_im)
            # cv2.circle(tmp, (x, y), r, color=(1, 1, 1), thickness=-1)
            cv2.circle(tmp, (x, y), r, color=1, thickness=-1)
            tmp *= surv_k
            surv_im += tmp

            if show:
                plt.figure()
                plt.subplot(121), plt.imshow(surv_im, 'gray', interpolation='nearest'), plt.title('surv_im')
                plt.subplot(122), plt.imshow(tmp, 'gray', interpolation='nearest'), plt.title('tmp')
                if show_now:
                    plt.show()

    return surv_im * mask


def compose_resp_im(image, blobs):
    resp_im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for b_im in blobs:
        for y, x, r in b_im:
            cv2.circle(resp_im, (x, y), r, color=(255, 0, 255), thickness=1)

    return resp_im


def detect_dog(image, mask, sigma_ratios, thresholds, overlaps):
    dogs = []

    # SIGMA RATIO  ----------------------------------------------------
    dogs_sr = []
    for i in sigma_ratios:
        blobs = dog(image, mask=mask, intensity='dark', sigma_ratio=i)
        dogs_sr.append(blobs)
        dogs.append(blobs)
    # dog_sr_surv = calc_survival_fcn(dogs_sr, mask, show=False)

    # THRESHOLD  ------------------------------------------------------
    dogs_t = []
    for i in thresholds:
        blobs = dog(image, mask=mask, intensity='dark', threshold=i)
        dogs_t.append(blobs)
        dogs.append(blobs)
    # dog_t_surv = calc_survival_fcn(dogs_t, mask, show=False)

    # # OVERLAP  -------------------------------------------------------
    # dogs_o = []
    # for i in overlaps:
    #     blobs = dog(image, mask=mask, intensity='dark', overlap=i)
    #     dogs_o.append(blobs)
    #     dogs.append(blobs)
    # dog_o_surv = calc_survival_fcn(dogs_o, mask, show=False)

    # surv_overall = calc_survival_fcn(dogs, mask)
    #
    # if show or save_fig:
    #     cmap = 'jet'
    #     fig = plt.figure(figsize=(24, 14))
    #     plt.subplot(141), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
    #     plt.subplot(142), plt.imshow(surv_overall, cmap=cmap, interpolation='nearest'), plt.title('DOG survival')
    #     divider = make_axes_locatable(plt.gca())
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     plt.colorbar(cax=cax)
    #     plt.subplot(143), plt.imshow(dog_sr_surv, cmap=cmap, interpolation='nearest'), plt.title('sigma ratio surv')
    #     divider = make_axes_locatable(plt.gca())
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     plt.colorbar(cax=cax)
    #     plt.subplot(144), plt.imshow(dog_t_surv, cmap=cmap, interpolation='nearest'), plt.title('threshold surv')
    #     divider = make_axes_locatable(plt.gca())
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     plt.colorbar(cax=cax)
    #     # plt.subplot(236), plt.imshow(dog_o_surv, cmap=cmap, interpolation='nearest'), plt.title('overlap surv')
    #
    #     if save_fig:
    #         save_figure(image, mask, dogs_sr, 'blobs/dogs', 'DOG', 'sigma_ratio', sigma_ratios, show=True, show_now=False)
    #         save_figure(image, mask, dogs_t, 'blobs/dogs', 'DOG', 'threshold', thresholds, show=True, show_now=False)
    #         # save_figure(image, mask, dogs_o, 'blobs/dogs', 'DOG', 'overlap', overlaps, show=True)
    #         fig.savefig('blobs/dogs/dog_survival_function.png', dpi=100, bbox_inches='tight', pad_inches=0)
    #
    #     if show and show_now:
    #         plt.show()

    return dogs, dogs_sr, dogs_t


def detect_log(image, mask, num_sigmas, thresholds, overlaps, log_scales):
    logs = []

    # NUMBER OF SIGMAS  ----------------------------------------------------
    logs_ns = []
    for i in num_sigmas:
        blobs = log(image, mask=mask, intensity='dark', num_sigma=i)
        logs_ns.append(blobs)
        logs.append(blobs)
    # log_ns_surv = calc_survival_fcn(logs_ns, mask, show=False)

    # THRESHOLD  ------------------------------------------------------
    logs_t = []
    for i in thresholds:
        blobs = log(image, mask=mask, intensity='dark', threshold=i)
        logs_t.append(blobs)
        logs.append(blobs)
    # log_t_surv = calc_survival_fcn(logs_t, mask, show=False)

    # OVERLAP  -------------------------------------------------------
    # logs_o = []
    # for i in overlaps:
    #     blobs = log(image, mask=mask, intensity='dark', overlap=i)
    #     logs_o.append(blobs)
    #     logs.append(blobs)
    # log_o_surv = calc_survival_fcn(logs_o, mask, show=False)

    # LOG SCALE  -------------------------------------------------------
    logs_ls = []
    for i in log_scales:
        blobs = log(image, mask=mask, intensity='dark', log_scale=i)
        logs_ls.append(blobs)
        logs.append(blobs)
    # log_ls_surv = calc_survival_fcn(logs_o, mask, show=False)

    # surv_overall = calc_survival_fcn(logs, mask)

    # cmap = 'jet'
    # fig = plt.figure(figsize=(20, 10))
    # plt.subplot(231), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(232), plt.imshow(surv_overall, cmap=cmap, interpolation='nearest'), plt.title('LOG survival')
    # plt.subplot(233), plt.imshow(log_ns_surv, cmap=cmap, interpolation='nearest'), plt.title('num. of sigmas surv')
    # plt.subplot(234), plt.imshow(log_t_surv, cmap=cmap, interpolation='nearest'), plt.title('threshold surv')
    # plt.subplot(235), plt.imshow(log_o_surv, cmap=cmap, interpolation='nearest'), plt.title('overlap surv')
    # plt.subplot(236), plt.imshow(log_ls_surv, cmap=cmap, interpolation='nearest'), plt.title('log scale surv')
    # fig.savefig('blobs/logs/log_survival_function.png')
    #
    # if save_fig:
    #     save_figure(image, mask, logs_ns, 'blobs/logs', 'LOG', 'num_sigma', num_sigmas, show=True, show_now=False)
    #     save_figure(image, mask, logs_t, 'blobs/logs', 'LOG', 'threshold', thresholds, show=True, show_now=False)
    #     save_figure(image, mask, logs_o, 'blobs/logs', 'LOG', 'overlap', overlaps, show=True, show_now=False)
    #     save_figure(image, mask, logs_ls, 'blobs/logs', 'LOG', 'log_scale', log_scales, show=True, show_now=False)
    #     fig.savefig('blobs/logs/log_survival_function.png')

    # plt.show()

    return logs, logs_ns, logs_t, logs_ls


def detect_doh(image, mask, num_sigmas, thresholds, overlaps, log_scales):
    dohs = []

    # NUMBER OF SIGMAS  ----------------------------------------------------
    dohs_ns = []
    for i in num_sigmas:
        blobs = doh(image, mask=mask, intensity='dark', num_sigma=i)
        dohs_ns.append(blobs)
        dohs.append(blobs)
    # doh_ns_surv = calc_survival_fcn(dohs_ns, mask, show=False)

    # THRESHOLD  ------------------------------------------------------
    dohs_t = []
    for i in thresholds:
        blobs = doh(image, mask=mask, intensity='dark', threshold=i)
        dohs_t.append(blobs)
        dohs.append(blobs)
    # doh_t_surv = calc_survival_fcn(dohs_t, mask, show=False)

    # # OVERLAP  -------------------------------------------------------
    # dohs_o = []
    # for i in overlaps:
    #     blobs = doh(image, mask=mask, intensity='dark', overlap=i)
    #     dohs_o.append(blobs)
    #     dohs.append(blobs)
    # doh_o_surv = calc_survival_fcn(dohs_o, mask, show=False)

    # LOG SCALE  -------------------------------------------------------
    dohs_ls = []
    for i in log_scales:
        blobs = doh(image, mask=mask, intensity='dark', log_scale=i)
        dohs_ls.append(blobs)
        dohs.append(blobs)
    # doh_ls_surv = calc_survival_fcn(dohs_ls, mask, show=False)

    # calculating overall survival function
    surv_overall = calc_survival_fcn(dohs, mask)

    # cmap = 'jet'
    # fig = plt.figure(figsize=(20, 10))
    # plt.subplot(231), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(232), plt.imshow(surv_overall, cmap=cmap, interpolation='nearest'), plt.title('DOG survival')
    # plt.subplot(233), plt.imshow(doh_ns_surv, cmap=cmap, interpolation='nearest'), plt.title('num. of sigmas surv')
    # plt.subplot(234), plt.imshow(doh_t_surv, cmap=cmap, interpolation='nearest'), plt.title('threshold surv')
    # plt.subplot(235), plt.imshow(doh_o_surv, cmap=cmap, interpolation='nearest'), plt.title('overlap surv')
    # plt.subplot(236), plt.imshow(doh_ls_surv, cmap=cmap, interpolation='nearest'), plt.title('log scale surv')
    #
    # if save_fig:
    #     save_figure(image, mask, dohs_ns, 'blobs/dohs', 'DOH', 'num_sigma', num_sigmas, show=True, show_now=False)
    #     save_figure(image, mask, dohs_t, 'blobs/dohs', 'DOH', 'threshold', thresholds, show=True, show_now=False)
    #     save_figure(image, mask, dohs_o, 'blobs/dohs', 'DOH', 'overlap', overlaps, show=True, show_now=False)
    #     save_figure(image, mask, dohs_ls, 'blobs/dohs', 'DOH', 'log_scale', log_scales, show=True, show_now=False)
    #     fig.savefig('blobs/dohs/doh_survival_function.png')
    #
    # plt.show()

    return dohs, dohs_ns, dohs_t, dohs_ls


def detect_opencv_detector(image, mask, min_thresholds, min_areas, min_circularities, min_convexities, min_inertias, save_fig=False):
    blobs_all = []

    # MIN THRESHOLDS -------------------------------------------------
    blobs_mt = []
    for i in min_thresholds:
        blobs = cv_blobs(image, mask, min_threshold=i)
        blobs_mt.append(blobs)
        blobs_all.append(blobs)
    # cv_mt_surv = calc_survival_fcn(blobs_mt, mask, show=False)

    # MIN AREAS ------------------------------------------------------
    blobs_ma = []
    for i in min_areas:
        blobs = cv_blobs(image, mask, min_area=i)
        blobs_ma.append(blobs)
        blobs_all.append(blobs)
    # cv_ma_surv = calc_survival_fcn(blobs_ma, mask, show=False)

    # MIN CIRCULARITIES -----------------------------------------------
    blobs_mcir = []
    for i in min_circularities:
        blobs = cv_blobs(image, mask, min_circularity=i)
        blobs_mcir.append(blobs)
        blobs_all.append(blobs)
    # cv_mcir_surv = calc_survival_fcn(blobs_mcir, mask, show=False)

    # MIN CONVEXITIES -----------------------------------------------
    blobs_mcon = []
    for i in min_convexities:
        blobs = cv_blobs(image, mask, min_convexity=i)
        blobs_mcon.append(blobs)
        blobs_all.append(blobs)
    # cv_mcon_surv = calc_survival_fcn(blobs_mcon, mask, show=False)

    # MIN INERTIAS -----------------------------------------------
    blobs_mi = []
    for i in min_inertias:
        blobs = cv_blobs(image, mask, min_inertia=i)
        blobs_mi.append(blobs)
        blobs_all.append(blobs)
    # cv_mi_surv = calc_survival_fcn(blobs_mi, mask, show=False)

    # CALC SURVIVAL FUNCTION ------------------------------------
    # surv_overall = calc_survival_fcn(blobs_all, mask)

    # visualization
    # cmap = 'jet'
    # fig = plt.figure(figsize=(20, 10))
    # plt.subplot(241), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(242), plt.imshow(surv_overall, cmap=cmap, interpolation='nearest'), plt.title('OPENCV survival')
    # plt.subplot(243), plt.imshow(cv_mt_surv, cmap=cmap, interpolation='nearest'), plt.title('min threshold surv')
    # plt.subplot(244), plt.imshow(cv_ma_surv, cmap=cmap, interpolation='nearest'), plt.title('min area surv')
    # plt.subplot(245), plt.imshow(cv_mcir_surv, cmap=cmap, interpolation='nearest'), plt.title('min circularity surv')
    # plt.subplot(246), plt.imshow(cv_mcon_surv, cmap=cmap, interpolation='nearest'), plt.title('min convexity surv')
    # plt.subplot(247), plt.imshow(cv_mi_surv, cmap=cmap, interpolation='nearest'), plt.title('min inertia surv')
    #
    # if save_fig:
    #     save_figure(image, mask, blobs_mt, 'blobs/cv', 'CV', 'min_thresh', min_thresholds, show=True, show_now=False)
    #     save_figure(image, mask, blobs_ma, 'blobs/cv', 'CV', 'min_area', min_areas, show=True, show_now=False)
    #     save_figure(image, mask, blobs_mcir, 'blobs/cv', 'CV', 'min_circularity', min_circularities, show=True, show_now=False)
    #     save_figure(image, mask, blobs_mcon, 'blobs/cv', 'CV', 'min_convexity', min_convexities, show=True, show_now=False)
    #     save_figure(image, mask, blobs_mi, 'blobs/cv', 'CV', 'min_inertia', min_inertias, show=True, show_now=False)
    #     fig.savefig('blobs/cv/survival_function.png')
    #
    # plt.show()

    return blobs_all, blobs_mt, blobs_ma, blobs_mcir, blobs_mcon, blobs_mi


def detect_blobs(image, mask, blob_type, layer_id, show=False, show_now=True, save_fig=False, fig_dir='.'):
    if blob_type == BLOB_DOG:
        # DOG detection -----------------
        print 'DOG detection ...',
        params = ('sigma_ratio', 'threshold')
        sigma_ratios = np.arange(0.6, 2, 0.2)
        thresholds = np.arange(0, 1, 0.1)
        overlaps = np.arange(0, 1, 0.2)
        blobs, blobs_sr, blobs_t = detect_dog(image, mask, sigma_ratios, thresholds, overlaps)
        blobs_sr_surv = calc_survival_fcn(blobs_sr, mask)
        blobs_t_surv = calc_survival_fcn(blobs_t, mask)
        blobs_surv_overall = calc_survival_fcn(blobs, mask)
        # dog_resp_overall = compose_resp_im(image, dogs)
        print 'done'

    elif blob_type == BLOB_LOG:
        # LOG detection -----------------
        print 'LOG detection ...',
        params = ('num_sigma', 'threshold', 'log_scale')
        num_sigmas = np.arange(5, 15, 2)
        thresholds = np.arange(0, 1, 0.1)
        overlaps = np.arange(0, 1, 0.2)
        log_scales = [False, True]
        blobs, blobs_ns, blobs_t, blobs_ls = detect_log(image, mask, num_sigmas, thresholds, overlaps, log_scales)
        blobs_ns_surv = calc_survival_fcn(blobs_ns, mask)
        blobs_t_surv = calc_survival_fcn(blobs_t, mask)
        blobs_ls_surv = calc_survival_fcn(blobs_ls, mask)
        blobs_surv_overall = calc_survival_fcn(blobs, mask)
        print 'done'

    elif blob_type == BLOB_DOH:
        # DOH detection -----------------
        print 'DOH detection ...',
        num_sigmas = np.arange(5, 15, 2)
        thresholds = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.3, 0.5, 1]
        overlaps = np.arange(0, 1, 0.2)
        log_scales = [False, True]
        blobs, blobs_ns, blobs_t, blobs_ls = detect_doh(image, mask, num_sigmas, thresholds, overlaps, log_scales)
        blobs_ns_surv = calc_survival_fcn(blobs_ns, mask, show=False)
        blobs_t_surv = calc_survival_fcn(blobs_t, mask, show=False)
        blobs_ls_surv = calc_survival_fcn(blobs_ls, mask, show=False)
        blobs_surv_overall = calc_survival_fcn(blobs, mask)
        print 'done'

    elif blob_type == BLOB_CV:
        # OPENCV BLOB DETECTOR -------
        print 'OpenCV blob detection ...',
        params = ('min_threshold', 'min_area', 'min_circularity', 'min_convexity', 'min_inertia')
        min_thresholds = [1, 10, 30, 50, 80, 100, 150, 200]
        min_areas = [1, 5, 10, 30, 50, 80, 100, 150]
        min_circularities = [0, 0.1, 0.3, 0.5, 0.6, 0.8]
        min_convexities = [0, 0.1, 0.3, 0.5, 0.6, 0.8]
        min_inertias = [0, 0.1, 0.3, 0.5, 0.6, 0.8]
        blobs_all, blobs_mt, blobs_ma, blobs_mcir, blobs_mcon, blobs_mi = \
            detect_opencv_detector(image, mask, min_thresholds, min_areas, min_circularities, min_convexities, min_inertias)
        blobs_mt_surv = calc_survival_fcn(blobs_mt, mask, show=False)
        blobs_ma_surv = calc_survival_fcn(blobs_ma, mask, show=False)
        blobs_mcir_surv = calc_survival_fcn(blobs_mcir, mask, show=False)
        blobs_mcon_surv = calc_survival_fcn(blobs_mcon, mask, show=False)
        blobs_mi_surv = calc_survival_fcn(blobs_mi, mask, show=False)
        blobs_surv_overall = calc_survival_fcn(blobs_all, mask)
        print 'done'

    else:
        raise ValueError('Unknown blob type.')

    if blob_type == BLOB_DOG:
        survival_imgs = (blobs_surv_overall, blobs_sr_surv, blobs_t_surv)
        blobs = (blobs, blobs_sr, blobs_t)
        titles = ('layer #%i, overall' % (layer_id + 1), 'layer #%i, sigma ratio' % (layer_id + 1),
                  'layer #%i, threshold' % (layer_id + 1))
    elif blob_type == BLOB_LOG:
        survival_imgs = (blobs_surv_overall, blobs_ns_surv, blobs_t_surv, blobs_ls_surv)
        blobs = (blobs, blobs_ns, blobs_t, blobs_ls)
        titles = ('layer #%i, overall' % (layer_id + 1), 'layer #%i, sigma ratio' % (layer_id + 1),
                  'layer #%i, threshold' % (layer_id + 1), 'layer #%i, log scale' % (layer_id + 1))
    elif blob_type == BLOB_DOH:
        survival_imgs = (blobs_surv_overall, blobs_ns_surv, blobs_t_surv, blobs_ls_surv)
        blobs = (blobs, blobs_ns, blobs_t, blobs_ls)
        titles = ('layer #%i, overall' % (layer_id + 1), 'layer #%i, num sigma' % (layer_id + 1),
                  'layer #%i, threshold' % (layer_id + 1), 'layer #%i, log scale' % (layer_id + 1))
    elif blob_type == BLOB_CV:
        survival_imgs = (blobs_surv_overall, blobs_mt_surv, blobs_ma_surv, blobs_mcir_surv, blobs_mcon_surv, blobs_mi_surv)
        blobs = (blobs_all, blobs_mt, blobs_ma, blobs_mcir, blobs_mcon, blobs_mi)
        titles = ('layer #%i, overall' % (layer_id + 1), 'layer #%i, min thresholh' % (layer_id + 1),
                  'layer #%i, min area' % (layer_id + 1), 'layer #%i, min circularity' % (layer_id + 1),
                  'layer #%i, min convexity' % (layer_id + 1), 'layer #%i, min inertia' % (layer_id + 1))

    dog_res = blobs, survival_imgs, titles, params
    return dog_res


def run(image, mask, pyr_scale, blob_type, show=False, show_now=True, save_fig=False):
    fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/blobs/%s/' % blob_type
    if save_fig:
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

    image, mask = tools.crop_to_bbox(image, mask)
    image = tools.smoothing(image, sliceId=0)

    blobs_pyr = []
    survs_pyr = []
    titles_pyr = []
    pyr_imgs = []
    pyr_masks = []
    for layer_id, (im_pyr, mask_pyr) in enumerate(zip(tools.pyramid(image, scale=pyr_scale, inter=cv2.INTER_NEAREST),
                                                      tools.pyramid(mask, scale=pyr_scale, inter=cv2.INTER_NEAREST))):
        blobs_res, survs_res, titles, params = detect_blobs(im_pyr, mask_pyr, blob_type, layer_id=layer_id,
                                                            show=show, show_now=show_now, save_fig=save_fig)
        blobs_pyr.append(blobs_res)
        survs_pyr.append(survs_res)
        titles_pyr.append(titles)
        pyr_imgs.append(im_pyr)
        pyr_masks.append(mask_pyr)

    n_layers = len(pyr_imgs)
    n_params = len(survs_pyr[0]) - 1  # -1 because the first surv is overall surv

    # survival layers
    # for layer_id, layer in enumerate(survs_pyr):
    #     fig_surv_layer =
    #     for im in layer[1:]:  # the first surv is overall
    #         surv_layer += im
    #     surv_im_layers.append(surv_layer)

    # survival param -


    # survival overall
    surv_overall = np.zeros(image.shape)
    for survs_l in survs_pyr:
        surv_overall += cv2.resize(survs_l[0], image.shape[::-1])

    # responses
    all_blobs = []
    for layer_id, layer_p in enumerate(blobs_pyr):
        scale = pyr_scale ** layer_id
        for blobs in layer_p[0]:
            if len(blobs) > 0:
                for blob in blobs:
                    y, x, r = blob
                    all_blobs.append((scale * y, scale * x, scale * r))

    if show or save_fig:
        # SURVIVAL IMAGES  ----------
        # survival fcn - params per layer, number = n_layers
        for layer_id, survs_l in enumerate(survs_pyr):
            fig_surv_layer = plt.figure(figsize=(24, 14))
            for param_id, (im, tit) in enumerate(zip(survs_l[1:], titles_pyr[layer_id][1:])):
                plt.subplot(1, n_layers, param_id + 1)
                plt.imshow(im, 'jet', interpolation='nearest')
                plt.title(tit)
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(cax=cax)
            if save_fig:
                fig_surv_layer.savefig(os.path.join(fig_dir, '%s_surv_layer_%i.png' % (blob_type, layer_id)),
                                       dpi=100, bbox_inches='tight', pad_inches=0)

        # survival fcn - layers per param, number = n_params
        for param_id in range(n_params):
            fig_surv_layer = plt.figure(figsize=(24, 14))
            for layer_id, survs_l in enumerate(survs_pyr):
                plt.subplot(1, n_layers, layer_id + 1)
                plt.imshow(survs_l[param_id + 1], 'jet', interpolation='nearest')  # +1 because the first surv is the overall surv
                plt.title(titles_pyr[layer_id][param_id + 1])  # +1 because the first surv is the overall surv
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(cax=cax)
            if save_fig:
                fig_surv_layer.savefig(os.path.join(fig_dir, '%s_surv_%s.png' % (blob_type, params[param_id])),
                                       dpi=100, bbox_inches='tight', pad_inches=0)

        # survival fcn - layers, number = 1, [layer1, layer2, ...]
        fig_surv_layers = plt.figure(figsize=(24, 14))
        for layer_id, survs_l in enumerate(survs_pyr):
            plt.subplot(1, n_layers, layer_id + 1)
            plt.imshow(survs_l[0], 'jet', interpolation='nearest')
            plt.title('layer #%i, surv'% (layer_id + 1))
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(cax=cax)

        # survival fcn - params, number = 1, [param1, param2, ...]
        surv_params = []
        for i in range(n_params):
            surv_params.append(np.zeros(image.shape))
        for layer in survs_pyr:
            for i, surv in enumerate(layer[1:]):  # from 1 because the first surv is overall
                surv_params[i] += cv2.resize(surv, image.shape[::-1])
        fig_surv_params = plt.figure(figsize=(24, 14))
        for param_id, (surv_p, param) in enumerate(zip(surv_params, params)):
            plt.subplot(1, n_params, param_id + 1)
            plt.imshow(surv_p, 'jet', interpolation='nearest')
            plt.title('%s, surv' % param)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(cax=cax)

        # survival fcn - overall, number = 1
        fig_surv = plt.figure(figsize=(24, 14))
        plt.subplot(121), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(122), plt.imshow(surv_overall, 'jet', interpolation='nearest')
        plt.title('%s - overall survival fcn' % blob_type)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)

        # RESPONSE IMAGES  ----------
        # response image - layers per param, number = n_params, [p1l1, p1l2, ...]
        for param_id in range(n_params):
            fig_resps_layer = plt.figure(figsize=(24, 14))
            for layer_id, blobs_l in enumerate(blobs_pyr):
                plt.subplot(1, n_layers, layer_id + 1)
                plt.imshow(pyr_imgs[layer_id], 'gray', interpolation='nearest')
                for blobs in blobs_l[param_id + 1]:
                    if len(blobs) > 0:
                        for blob in blobs:
                            y, x, r = blob
                            c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
                            plt.gca().add_patch(c)
                plt.title(titles_pyr[layer_id][param_id + 1])  # +1 because the first surv is the overall surv
            if save_fig:
                fig_resps_layer.savefig(os.path.join(fig_dir, '%s_resps_%s.png' % (blob_type, params[param_id])),
                                        dpi=100, bbox_inches='tight', pad_inches=0)

        # response image - params per layer, number = n_layers, [l1p1, l1p2, ...]
        for layer_id, blobs_l in enumerate(blobs_pyr):
            fig_resps_layer = plt.figure(figsize=(24, 14))
            for param_id, (param, tit) in enumerate(zip(params, titles_pyr[layer_id][1:])):
                plt.subplot(1, n_layers, param_id + 1)
                plt.imshow(pyr_imgs[layer_id], 'gray', interpolation='nearest')
                for blobs in blobs_l[param_id + 1]:
                    if len(blobs) > 0:
                        for blob in blobs:
                            y, x, r = blob
                            c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
                            plt.gca().add_patch(c)
                plt.title(tit)
            if save_fig:
                fig_resps_layer.savefig(os.path.join(fig_dir, '%s_resps_layer_%i.png' % (blob_type, layer_id)),
                                        dpi=100, bbox_inches='tight', pad_inches=0)

        # response image - params, number = 1, [param1, param2, ...]
        fig_resps_params = plt.figure(figsize=(24, 14))
        for param_id, param in enumerate(params):
            for layer_id, blobs_l in enumerate(blobs_pyr):
                scale = pyr_scale ** layer_id
                plt.subplot(1, n_params, param_id + 1)
                plt.imshow(pyr_imgs[0], 'gray', interpolation='nearest')
                for blobs in blobs_l[param_id + 1]:
                    if len(blobs) > 0:
                        for blob in blobs:
                            y, x, r = blob
                            c = plt.Circle((scale * x, scale * y), scale * r, color='r', linewidth=2, fill=False)
                            plt.gca().add_patch(c)
                plt.title('%s' % param)

        # response image - layers, number = 1
        fig_resps_layers = plt.figure(figsize=(24, 14))
        for layer_id, blobs_l in enumerate(blobs_pyr):
            plt.subplot(1, n_layers, layer_id + 1)
            plt.imshow(pyr_imgs[layer_id], 'gray', interpolation='nearest')
            for blobs in blobs_l[0]:
                if len(blobs) > 0:
                    for blob in blobs:
                        y, x, r = blob
                        c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
                        plt.gca().add_patch(c)
            plt.title('layer #%i'% (layer_id + 1) )

        # response image - overall, number = 1
        fig_resp = plt.figure(figsize=(24, 14))
        plt.subplot(121), plt.imshow(image, 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(122), plt.imshow(image, 'gray', interpolation='nearest')
        for blob in all_blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
            plt.gca().add_patch(c)
        plt.title('%s - overall response image' % blob_type)

        if save_fig:
            fig_surv.savefig(os.path.join(fig_dir, '%s_surv_overall.png' % blob_type), dpi=100, bbox_inches='tight', pad_inches=0)
            fig_surv_layers.savefig(os.path.join(fig_dir, '%s_surv_layers.png' % blob_type), dpi=100, bbox_inches='tight', pad_inches=0)
            fig_surv_params.savefig(os.path.join(fig_dir, '%s_surv_params.png' % blob_type), dpi=100, bbox_inches='tight', pad_inches=0)
            fig_resp.savefig(os.path.join(fig_dir, '%s_resps_overall.png' % blob_type), dpi=100, bbox_inches='tight', pad_inches=0)
            fig_resps_layers.savefig(os.path.join(fig_dir, '%s_resps_layers.png' % blob_type), dpi=100, bbox_inches='tight', pad_inches=0)
            fig_resps_params.savefig(os.path.join(fig_dir, '%s_resps_params.png' % blob_type), dpi=100, bbox_inches='tight', pad_inches=0)

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
    blob_type = BLOB_DOG
    run(data_s, mask_s, pyr_scale=pyr_scale, blob_type=blob_type, show=show, show_now=show_now, save_fig=save_fig)
    if show and not show_now:
        plt.show()

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