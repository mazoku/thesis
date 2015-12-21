__author__ = 'tomas'

import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

from PyQt4 import QtGui
import numpy as np
import cv2
import tools

import pickle
import ConfigParser

import Viewer_3D

import fast_marching as fm
import snakes


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

    slice_idx = 14
    methods = ['fm', 'snakes']
    # comparing(data, methods, params, mask, slice_idx=slice_idx)

    morph_hat_test(data[ slice_idx, :, :])

    # # data visualization
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