__author__ = 'tomas'

import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

import os

from PyQt4 import QtGui
import numpy as np
import matplotlib.patches as matpat
import matplotlib.pyplot as plt
import cv2

import pickle
import ConfigParser

from skimage import img_as_float, data
import skimage.measure as skimea
import skimage.exposure as skiexp
import skimage.color as skicol
import skimage.draw as skidra
import skimage.morphology as skimor
import skimage.segmentation as skiseg

import Viewer_3D

import fast_marching as fm
import snakes
import rw_segmentation as rw
import he_pipeline as heep
import slics

import tools

from sys import stdout

################################################################################
def load_parameters(config_path='config.ini'):
    config = ConfigParser.ConfigParser()
    config.read('config.ini')

    params = dict()

    for section in config.sections():
        for option in config.options(section):
            try:
                params[option] = config.getint(section, option)
            except ValueError:
                params[option] = config.getfloat(section, option)
            except:
                params[option] = config.get(section, option)
    return params


def contour_test(img, mask):
    img, mask = tools.crop_to_bbox(img, mask)

    img_o = img.copy()

    # write mean value outside the mask ----------------------------------------
    mean_val = np.mean(img[np.nonzero(mask)])

    img = np.where(mask, img, mean_val)

    # rescale to UINT8 ---------------------------------------------------------
    img = skiexp.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)

    # smoothing ----------------------------------------------------------------
    img = tools.smoothing(img)

    # Contrast stretching -----------------------------------------------------
    # ps = [(2, 98), (5, 95), (10, 90), (20, 80)]
    ps = [(10, 40), (10, 90)]
    imgs_rescale = []
    for p in ps:
        pmin, pmax = np.percentile(img, p)
        imgs_rescale.append(skiexp.rescale_intensity(img, in_range=(pmin, pmax)))

    # Equalization --------------------------------------------------------------
    img_eq = skiexp.equalize_hist(img)

    # canny edge detector -----------------------------------------------------
    canny_s = tools.auto_canny(img, sigma=0.05)
    cannys_res = []
    for i in imgs_rescale:
        cannys_res.append(tools.auto_canny(i, sigma=0.1))
    canny_eq = tools.auto_canny(skiexp.rescale_intensity(img_eq, out_range=np.uint8).astype(np.uint8), sigma=0.1)

    # morphology -------------------------------------------------------------
    img_eq_morph = skimor.closing(img_eq, selem=skimor.disk(7))
    # img_eq_morph = skimor.closing(imgs_rescale[0], selem=skimor.disk(7))
    # plt.figure()
    # plt.subplot(121), plt.imshow(img_eq, 'gray', interpolation='nearest'), plt.title('img equalized')
    # plt.subplot(122), plt.imshow(img_eq_morph, 'gray', interpolation='nearest'), plt.title('closed')
    # plt.show()

    # contours ----------------------------------------------------------------
    contours = skimea.find_contours(img_eq_morph, 0.2)
    # plt.figure()
    # plt.imshow(img_eq_morph, 'gray', interpolation='nearest')
    # for n, contour in enumerate(contours):
    #     plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # plt.axis('image')
    # plt.show()

    # TODO: zkusit contury na canny a primo na im_eq

    vmin = 50 - 175
    vmax = 50 + 175
    plt.figure()
    plt.imshow(img_o, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax)

    for (im, p) in zip(imgs_rescale, ps):
        plt.figure()
        plt.subplot(121), plt.imshow(im, 'gray', interpolation='nearest'), plt.title('contrast stretching - {}'.format(p))
        plt.subplot(122), plt.imshow(skimor.closing(im, selem=skimor.disk(3)), 'gray', interpolation='nearest'), plt.title('closing')

    plt.figure()
    plt.imshow(img_eq, 'gray', interpolation='nearest')
    plt.title('equalization')

    # PLOT CANNY -------------------------------------------
    # plt.figure()
    # plt.imshow(canny_s, 'gray', interpolation='nearest')
    # plt.title('canny - smoothed')

    # for (im, p) in zip(cannys_res, ps):
    #     plt.figure()
    #     plt.imshow(im, 'gray', interpolation='nearest')
    #     plt.title('canny - contrast stretching - {}'.format(p))

    # plt.figure()
    # plt.imshow(canny_eq, 'gray', interpolation='nearest')
    # plt.title('canny - equalization')

    plt.show()


def morph_hat_test(im):
    im = tools.windowing(im)
    im = tools.smoothing(im)

    min_size = 11
    max_size = 31
    strels = []
    for i in range(min_size, max_size + 1, 6):
        strels.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i)))
    tophats = tools.morph_hat(im, strels, show=True, show_now=False)

    min_size = 41
    max_size = 61
    strels = []
    for i in range(min_size, max_size + 1, 6):
        strels.append(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i)))

    blackhats = tools.morph_hat(im, strels, type='blackhat', show=True, show_now=True)


def comparing(data, methods, params, mask=None, slice_idx=None):
    """
    A function for comparing different methods.
    :param data: Input data to be analysed, 2D or 3D array.
    :param methods: List of method names/identifiers.
    :param params: Dictionary with parameters.
    :param slice_idx: In case of 3D data input, provide index of a slice for 2D visualisation.
    :return:
    """

    if 'fm' in methods:
        fm_hypo_init, fm_hyper_init, fm_speed_hypo, fm_speed_hyper = fm.run(data, params, mask)

    if 'snakes' in methods:
        snakes.run(data, params, mask=mask)


# ---------------------------------------------------------------------------------------------------------------------
def run(data_fname, params_fname):
    data, mask, voxel_size = tools.load_pickle_data(data_fname)
    params = load_parameters(params_fname)

    seeds_fname = tools.get_subdir_fname(data_fname, 'seeds')

    # -- FAST MARCHING --
    # fm.run(data, params, mask)

    # -- SNAKES --
    # snakes.run(data, params, mask=mask)

    # -- RANDOM WALKER --
    # segs = rw.run(data, seeds_fname)
    # tools.save_pickle(data_fname, 'seg_rw', data, mask, segs, voxel_size)

    # -- HISTOGRAM EQUALIZATION PIPELINE --
    # segs, vis = heep.run(data, mask, proc=['smoo', 'equa', 'clos', 'cont'], disp=False, show=False)
    # tools.save_pickle(data_fname, 'seg_he_pipeline', data, mask, segs, voxel_size)
    # tools.save_figs(data_fname, 'figs_he_pipeline', data, mask, vis)

    # -- SLIC --
    return_vis = True
    out = slics.run(data, mask, voxel_size, return_vis=return_vis)
    if return_vis:
        segs, vis, ranges, cmaps = out
        tools.save_figs(data_fname, 'figs_seg_slic', data, mask, vis, ranges=ranges, cmaps=cmaps)
    else:
        segs = out
    # tools.save_pickle(data_fname, 'seg_slic', data, mask, segs, voxel_size)

    # -- VISUALIZATION --
    # tools.show_3d(segs)

    # # data visualization
    # slice_idx = 14
    # methods = ['fm', 'snakes']
    # comparing(data, methods, params, mask, slice_idx=slice_idx)

    # morph_hat_test(data[ slice_idx, :, :])

    # data visualization
    # app = QtGui.QApplication(sys.argv)
    # viewer = Viewer_3D.Viewer_3D(segs, range=False)
    # viewer.show()
    # sys.exit(app.exec_())


################################################################################
################################################################################
if __name__ == '__main__':

    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation//org-exp_183_46324212_arterial_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_180_49509315_venous_5.0_B30f-.pklz'
    config_fname = 'config.ini'
    data_fnames = [data_fname,]

    dir_name = '/home/tomas/Data/liver_segmentation/'
    files = os.listdir(dir_name)
    # data_fnames = [os.path.join(dir_name, x) for x in files if '_5.0_' in x]

    for i, data_fname in enumerate(data_fnames):
        # print 'Processing file #%i/%i -- %s' % (i + 1, len(data_fnames), data_fname.split('/')[-1])
        print 'Processing file #%i/%i -- %s' % (i + 1, len(data_fnames), data_fname[data_fname.rfind('/') + 1:])
        run(data_fname, config_fname)


        # pyed = QTSeedEditor(self.data3d, mode='seed', voxelSize=self.voxelsize_mm)

#TODO: SLICS - vyplnit diry po morp openingu