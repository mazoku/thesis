__author__ = 'tomas'

import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

from PyQt4 import QtGui
import numpy as np

import pickle
import ConfigParser

import Viewer_3D

import fast_marching as fm


def run(data_fname, params_fname):
    data, mask, voxel_size = load_pickle_data(data_fname)
    params = load_parameters(params_fname)

    fm.run(data, params, mask)

    # # data visualization
    # app = QtGui.QApplication(sys.argv)
    # viewer = Viewer_3D.Viewer_3D(data, range=False)
    # viewer.show()
    # sys.exit(app.exec_())


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


################################################################################
################################################################################
if __name__ == '__main__':

    data_fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    config_fname = 'config.ini'
    run(data_fname, config_fname)