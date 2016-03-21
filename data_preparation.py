from __future__ import division

import sys
import os

import numpy as np

if os.path.exists('../imtools/'):
    sys.path.append('../imtools/')
    from imtools import tools
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


def slice_data(fname):
    datap = tools.load_pickle_data(fname, return_datap=True)
    data3d = datap['data3d']
    labels = datap['segmentation']


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Dropbox/Data/medical/235_venous-.pklz'

