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
import tools

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


def my_pipeline(img, mask, proc=['smoo', 'equa', 'clos', 'cont'], disp=True, show=True):
    n_rows, n_cols = [1, len(proc) + 1]

    img, mask = tools.crop_to_bbox(img, mask)

    # write mean value outside the mask
    mean_val = np.mean(img[np.nonzero(mask)])
    img = np.where(mask, img, mean_val)

    # rescale to UINT8
    img = skiexp.rescale_intensity(img, in_range=(50 - 175, 50 + 175), out_range=np.uint8).astype(np.uint8)

    vis = []
    img_pipe = img.copy()

    # saving input
    vis.append(('input', img * mask))

    # smoothing
    if 'smo' in proc or 'smoothing' in proc:
        img_s = tools.smoothing(img)
        img_pipe = img_s.copy()
        vis.append(('smoothing', img_s * mask))

    # # write mean value outside the mask
    # mean_val = np.mean(img_pipe[np.nonzero(mask)])
    # img_pipe = np.where(mask, img_pipe, mean_val)

    # Equalization
    if 'equ' in proc or 'equalization' in proc:
        img_h = skiexp.equalize_hist(img_pipe)
        img_pipe = img_h.copy()
        vis.append(('equalization', img_h * mask))

    # canny edge detector
    # canny = tools.auto_canny(skiexp.rescale_intensity(img_eq, out_range=np.uint8).astype(np.uint8), sigma=0.1)

    # contrast stretch
    if 'str' in proc or 'stretch' in proc:
        p = (20, 40)
        pmin, pmax = np.percentile(img_pipe, p)
        img_rescale = skiexp.rescale_intensity(img_pipe, in_range=(pmin, pmax))
        img_pipe = img_rescale.copy()
        vis.append(('stretch {}'.format(p), img_rescale * mask))

    # morphology
    if 'clo' in proc or 'closing' in proc:
        r = 7
        img_morph = skimor.closing(img_pipe, selem=skimor.disk(r))
        img_pipe = img_morph.copy()
        vis.append(('closing r=%i'%r, img_morph * mask))

    # contours
    if 'con' in proc or 'contours' in proc:
        contours = skimea.find_contours(img_pipe, 0.2)
        vis.append(('contours', contours))

    if disp:
        plt.figure()
        for i, (title, im) in enumerate(vis):
            plt.subplot(n_rows, n_cols, i + 1)
            if title == 'contours':
                plt.imshow(vis[0][1], 'gray', interpolation='nearest'), plt.title(title)
                plt.hold(True)
                for n, contour in enumerate(im):
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.axis('image')
            else:
                plt.imshow(im, 'gray', interpolation='nearest'), plt.title(title)

        # if 'con' in proc or 'contours' in proc:
        #     for n, contour in enumerate(contours):
        #         plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #     plt.axis('image')

        if show:
            plt.show()

    return vis


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
    data, mask, voxel_size = load_pickle_data(data_fname)
    params = load_parameters(params_fname)

    dirs = data_fname.split('/')
    dir = dirs[-1]
    patient_id = dir[8:11]
    if 'venous' in dir or 'ven' in dir:
        phase = 'venous'
    elif 'arterial' in dir or 'art' in dir:
        phase = 'arterial'
    else:
        phase = 'phase_unknown'

    # fm.run(data, params, mask)
    # snakes.run(data, params, mask=mask)

    # batch pipeline -----------------------------------------------------------------------------------
    data, mask = tools.crop_to_bbox(data, mask)
    res = []
    for i in range(data.shape[0]):
        # print 'Processing slice #%i/%i' % (i + 1, data.shape[0]),
        print tools.get_status_text('\tProcessing slices', iter=i, max_iter=data.shape[0]),
        vis = my_pipeline(data[i, :, :], mask[i, :, :], proc=['smo', 'equ', 'clo', 'con'], disp=False)
        res.append(vis)
        # print 'done'
    print tools.get_status_text('\tProcessing slice', iter=-1, max_iter=data.shape[0])

    fig_dir = os.path.join(os.path.curdir, 'figs_%s_%s' % (patient_id, phase))
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # coords = np.nonzero(mask)
    # r_max = min(mask.shape[1], max(coords[1]) + 1)
    # c_max = min(mask.shape[2], max(coords[2]) + 1)
    fig = plt.figure(figsize=(20, 5))
    for s in range(len(res)):
        # print 'Saving figure #%i/%i ...' % (s + 1, len(res)),
        print tools.get_status_text('\tSaving figures', iter=s, max_iter=len(res)),
        for i, (title, im) in enumerate(res[s]):
            plt.subplot(1, len(res[0]), i + 1)
            if title == 'contours':
                plt.imshow(res[s][0][1], 'gray', interpolation='nearest'), plt.title(title)
                plt.hold(True)
                for n, contour in enumerate(im):
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.axis('image')
            else:
                if i == 0:
                    coords = np.nonzero(mask[s, :, :])
                    r_min = max(0, min(coords[0]))
                    r_max = min(mask.shape[1], max(coords[0]))
                    c_min = max(0, min(coords[1]))
                    c_max = min(mask.shape[2], max(coords[1]))
                    plt.imshow(data[s,:,:], 'gray', interpolation='nearest', vmin=50 - 175, vmax=50 + 175), plt.title(title)
                    plt.hold(True)
                    bbox = matpat.Rectangle((c_min, r_min), (c_max - c_min + 1), (r_max - r_min + 1), color='red', fill=False, lw=2)
                    plt.gca().add_patch(bbox)
                    plt.axis('image')
                else:
                    plt.imshow(im, 'gray', interpolation='nearest'), plt.title(title)

                # plt.imshow(im, 'gray', interpolation='nearest'), plt.title(title)
        # plt.show()
        fig.savefig(os.path.join(fig_dir, 'slice_%i.png'%s))
        fig.clf()
        # print 'done'
    print tools.get_status_text('\tSaving figures', iter=-1, max_iter=len(res))
    print '--  Work done  --'

    # slice_id = 17
    # my_pipeline(data[slice_id, :, :], mask[slice_id, :, :], proc=['smo', 'equ', 'clo', 'con'], show=False)
    # my_pipeline(data[slice_id, :, :], mask[slice_id, :, :], proc=['smo', 'str', 'clo', 'con'], show=True)
    # contour_test(data[slice_id, :, :], mask[slice_id, :, :])

    # data visualization
    # slice_idx = 14
    # methods = ['fm', 'snakes']
    # comparing(data, methods, params, mask, slice_idx=slice_idx)

    # morph_hat_test(data[ slice_idx, :, :])

    # # data visualization
    # app = QtGui.QApplication(sys.argv)
    # viewer = Viewer_3D.Viewer_3D(data, range=False)
    # viewer.show()
    # sys.exit(app.exec_())


################################################################################
################################################################################
if __name__ == '__main__':

    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation//org-exp_183_46324212_arterial_5.0_B30f-.pklz'
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_180_49509315_venous_5.0_B30f-.pklz'
    config_fname = 'config.ini'

    dir_name = '/home/tomas/Data/liver_segmentation/'
    files = os.listdir(dir_name)
    data_fnames = [os.path.join(dir_name, x) for x in files if '_5.0_' in x]

    for i, data_fname in enumerate(data_fnames):
        print 'Processing file #%i/%i -- %s' % (i + 1, len(data_fnames), data_fname.split('/')[-1])
        run(data_fname, config_fname)


        pyed = QTSeedEditor(self.data3d, mode='seed',
                            voxelSize=self.voxelsize_mm)