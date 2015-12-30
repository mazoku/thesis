__author__ = 'tomas'

import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

from PyQt4 import QtGui
import numpy as np
import matplotlib.pyplot as plt

import pickle
import ConfigParser

from skimage import img_as_float, data
import skimage.measure as skimea
import skimage.exposure as skiexp
import skimage.color as skicol
import skimage.draw as skidra
import skimage.morphology as skimor

import Viewer_3D

import fast_marching as fm
import snakes
import cv2

import tools


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


def load_pickle_data(fname, slice_idx=-1):
    ext_list = ('pklz', 'pickle')
    if fname.split('.')[-1] in ext_list:

        try:
            import gzip
            f = gzip.open(fname, 'rb')
            fcontent = f.read()
            f.close()
        except Exception as e:
            logger.warning("Input gzip exception: " + str(e))
            f = open(fname, 'rb')
            fcontent = f.read()
            f.close()
        data_dict = pickle.loads(fcontent)

        # data = tools.windowing(data_dict['data3d'], level=params['win_level'], width=params['win_width'])
        data = data_dict['data3d']

        mask = data_dict['segmentation']

        voxel_size = data_dict['voxelsize_mm']

        if slice_idx != -1:
            data = data[slice_idx, :, :]
            mask = mask[slice_idx, :, :]

        return data, mask, voxel_size

    else:
        msg = 'Wrong data type, supported extensions: ', ', '.join(ext_list)
        raise IOError(msg)


def my_pipeline(img, mask):
    # write mean value outside the mask
    mean_val = np.mean(img[np.nonzero(mask)])
    img = np.where(mask, img, mean_val)

    # rescale to UINT8
    img = skiexp.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)

    # smoothing
    img_s = tools.smoothing(img)

    # Equalization
    img_eq = skiexp.equalize_hist(img_s)

    # canny edge detector
    # canny = tools.auto_canny(skiexp.rescale_intensity(img_eq, out_range=np.uint8).astype(np.uint8), sigma=0.1)
    p = (10, 40)
    pmin, pmax = np.percentile(img_s, p)
    img_rescale = skiexp.rescale_intensity(img_s, in_range=(pmin, pmax))

    # morphology
    img_eq_morph = skimor.closing(img_eq, selem=skimor.disk(7))

    # contours
    contours = skimea.find_contours(img_eq_morph, 0.2)
    n_rows, n_cols = [1, 6]
    im_vis = (img, img_s, img_rescale, img_eq, img_eq_morph, img)
    titles = ['input', 'smoothing', 'stretching', 'equalization', 'equa-closing', 'contours']

    plt.figure()
    for i, (im, title) in enumerate(zip(im_vis, titles)):
        plt.subplot(n_rows, n_cols, i + 1), plt.imshow(im, 'gray', interpolation='nearest'), plt.title(title)

    try:
        ind = titles.index('contours')
        plt.subplot(n_rows, n_cols, ind + 1)
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.axis('image')
    except:
        pass


    # plt.figure()
    # plt.subplot(171), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(172), plt.imshow(img_s, 'gray', interpolation='nearest'), plt.title('smoothing')
    # plt.subplot(173), plt.imshow(img_eq, 'gray', interpolation='nearest'), plt.title('equalization')
    # plt.subplot(174), plt.imshow(img_eq_morph, 'gray', interpolation='nearest'), plt.title('equa-closing')
    # plt.subplot(175), plt.imshow(img_rescale, 'gray', interpolation='nearest'), plt.title('stretching')
    # plt.subplot(177), plt.imshow(img_eq_morph, 'gray', interpolation='nearest'), plt.title('contours')
    # for n, contour in enumerate(contours):
    #     plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # plt.axis('image')
    plt.show()


def contour_test(img, mask):
    img, mask = tools.crop_to_bbox(img, mask)

    my_pipeline(img, mask)

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


def run(data_fname, params_fname):
    data, mask, voxel_size = load_pickle_data(data_fname)
    params = load_parameters(params_fname)

    # fm.run(data, params, mask)
    # snakes.run(data, params, mask=mask)
    slice_id = 17
    contour_test(data[slice_id, :, :], mask[slice_id, :, :])

    # data visualization
    # app = QtGui.QApplication(sys.argv)
    # viewer = Viewer_3D.Viewer_3D(data, range=False)
    # viewer.show()
    # sys.exit(app.exec_())


################################################################################
################################################################################
if __name__ == '__main__':

    data_fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    config_fname = 'config.ini'
    run(data_fname, config_fname)