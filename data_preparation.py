from __future__ import division

import sys
import os

import numpy as np

if os.path.exists('../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

if os.path.exists('../data_viewers/'):
    # sys.path.append('../data_viewers/')
    sys.path.insert(0, '../data_viewers/')
    from dataviewers import seg_viewer, viewer_3D
else:
    print 'You need to import package dataviewers: https://github.com/mazoku/data_viewers'
    sys.exit(0)


def slice_data(fname, reshape=True, shape=(20, 20)):
    datap = tools.load_pickle_data(fname, return_datap=True)
    data3d = datap['data3d']
    labels = datap['segmentation']
    voxelsize_mm = datap['voxelsize_mm']
    data_cubes = ()

    # resizing data to have same spacing
    orig_shape = data3d.shape
    n_slices = orig_shape[0] * (voxelsize_mm[0] / voxelsize_mm[1])
    new_shape = (n_slices, orig_shape[1], orig_shape[2])
    data3d = misc.resize_to_shape(data3d, shape=new_shape, order=1)
    labels = misc.resize_to_shape(labels, shape=new_shape, order=0)

    data_cubes = [cube for cube in tools.sliding_window_3d(data3d, 10, new_shape, only_whole=False, include_last=True)]

    # seg_viewer.show(data3d, 100 * labels)

    return data_cubes


def process_dir(data_dir):
    data_names = ()
    for (rootDir, dirNames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename[-5:] == '.pklz':
                data_names.append(filename)


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_dir = '/home/tomas/Dropbox/Data/medical/dataset/gt'

    data_fname = '/home/tomas/Data/medical/dataset/gt/180_arterial-GT.pklz'
    shape = (20, 20, 20)
    data_cubes = slice_data(data_fname)