from __future__ import division

#TODO: test na syntetickych datech

import sys
import os
import pickle
import gzip

import numpy as np
from PyQt4 import QtCore, QtGui
import matplotlib.pyplot as plt

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


def slice_data(fname, reshape=True, cube_shape=(20, 20, 20), step=10):
    datap = tools.load_pickle_data(fname, return_datap=True)
    data3d = datap['data3d']
    labels = datap['segmentation']
    voxelsize_mm = datap['voxelsize_mm']
    data_cubes = []
    label_cubes = []

    # data3d = np.arange(1, 401).reshape((4, 10, 10))
    # labels = 2 * np.ones((4, 10, 10))

    # resizing data to have same spacing
    orig_shape = data3d.shape
    n_slices = orig_shape[0] * (voxelsize_mm[0] / voxelsize_mm[1])
    new_shape = (n_slices, orig_shape[1], orig_shape[2])
    data3d = misc.resize_to_shape(data3d, shape=new_shape, order=1)
    labels = misc.resize_to_shape(labels, shape=new_shape, order=0)

    # print [(x, (labels == x).sum()) for x in np.unique(labels)]
    lesion_lbl = 1
    liver_lbl = 2
    bgd_lbl = 0
    data_cubes_all = [cube[4] for cube in tools.sliding_window_3d(data3d, step, cube_shape, only_whole=True, include_last=True)]
    label_cubes_all = [cube[4] for cube in tools.sliding_window_3d(labels, step, cube_shape, only_whole=True, include_last=True)]
    areaT = 0.5 * np.prod(cube_shape)  # prah pro minimalni zastoupeni jater v kostce
    # for data_cube, label_cube in zip(tools.sliding_window_3d(data3d, 10, cube_shape, only_whole=False, include_last=True)[4],
    #                                  tools.sliding_window_3d(labels, 10, cube_shape, only_whole=False, include_last=True)[4]):
    for data_cube, label_cube in zip(data_cubes_all, label_cubes_all):
        if (label_cube == liver_lbl).sum() >= areaT:
            mean_v = data_cube[np.nonzero(label_cube == liver_lbl)].mean()
            # tmp = data_cube[0,...].copy()
            data_cube = np.where(label_cube == bgd_lbl, mean_v, data_cube)

            # plt.figure()
            # plt.subplot(131), plt.imshow(tmp, 'gray', vmin=-125, vmax=225), plt.title('orig')
            # plt.subplot(132), plt.imshow(data_cube[0,...], 'gray', vmin=-125, vmax=225), plt.title('processed, mean=%.1f' % mean_v)
            # plt.subplot(133), plt.imshow(label_cube[0,...], 'gray', vmin=0, vmax=2), plt.title('mask')
            # plt.show()

            data_cubes.append(data_cube)
            label_cubes.append(label_cube == lesion_lbl)  # ulozim pouze oznacena loziska

    # seg_viewer.show(data3d, 100 * labels)

    return data_cubes, label_cubes


def save_cubes(data_cubes, label_cubes, orig_fname, shape):
    dirs = orig_fname.split(os.sep)
    # dirname = os.path.join(dirs[:-1], 'cubes', dirs[-1])
    dirname = os.sep + os.path.join(os.path.join(*dirs[:-1]), 'cubes_%i' % shape, dirs[-1][:-8] + '/')
    dirs = dirname.split(os.sep)
    for i in range(2, len(dirs)):
        subdir = os.sep.join(dirs[:i])
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    for i, (data_cube, label_cube) in enumerate(zip(data_cubes, label_cubes)):
        # base_fname = dirname[:dirname.find('.pklz')]
        data_fname = os.path.join(dirname, dirs[-2] + '_%i_datacube.npz' % i)
        label_fname = os.path.join(dirname, dirs[-2] + '_%i_labelcube.npz' % i)

        data_file = gzip.open(data_fname, 'wb', compresslevel=1)
        np.save(data_file, data_cube)
        data_file.close()

        labels_file = gzip.open(label_fname, 'wb', compresslevel=1)
        np.save(labels_file, label_cube)
        labels_file.close()


def process_dir(data_dir, cube_shape, step):
    data_names = []
    for (rootDir, dirNames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename[-5:] == '.pklz':
                data_names.append(filename)

    for i, dirname in enumerate(data_names):
        print 'Processing dir #%i/%i, %s:' % (i + 1, len(data_names), dirname),
        data_path = os.path.join(data_dir, dirname)
        print 'slicing ...',
        data_cubes, label_cubes = slice_data(data_path, cube_shape=cube_shape, step=step)
        print 'saving ...',
        save_cubes(data_cubes, label_cubes, data_path, cube_shape[0])
        print 'done.'


def check_data(data_dir, ext='npz'):
    data_fnames, label_fnames = find_datapairs(data_dir)

    app = QtGui.QApplication(sys.argv)
    for i, (data_fname, label_fname) in enumerate(zip(data_fnames, label_fnames)):
        with gzip.open(os.path.join(data_dir, data_fname), 'rb') as file:
            data = np.load(file)
        with gzip.open(os.path.join(data_dir, label_fname), 'rb') as file:
            labels = np.load(file)
        print 'Showing %i/%i: %s, %s' % (i + 1, len(data_fnames), data_fname, label_fname)
        seg_viewer.show(data, 200 * labels, app=app)


def find_datapairs(data_dir, ext='npz'):
    data_fnames = []
    label_fnames = []

    # rozdelim soubory do dvou skupin: obsahujici leze a obsahujici jatra
    for (rootDir, dirNames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename.split('.')[-1] == ext:
                if 'datacube' in filename:
                    data_fnames.append(filename)
                elif 'labelcube' in filename:
                    label_fnames.append(filename)

    # dam dohromady prislusne liver a lesion soubory
    data_aligned = []
    labels_aligned = []
    for data_fname in data_fnames:
        label_fname = data_fname.replace('datacube', 'labelcube')
        if label_fname in label_fnames:
            data_aligned.append(data_fname)
            labels_aligned.append(label_fname)
        else:
            print 'Cannot find label data for \'%s\'' % data_fname

    return data_aligned, labels_aligned


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # cube_shape = (20, 20, 20)
    cube_shape = (50, 50, 50)
    step = 25
    data_dir = '/home/tomas/Data/medical/dataset/gt'
    # process_dir(data_dir, cube_shape, step)

    # test_dir = '/home/tomas/Data/medical/dataset/gt/189a_venous-GT.pklz'
    # data_cubes, label_cubes = slice_data(test_dir, cube_shape=(1, 5, 5), step=2)
    # app = QtGui.QApplication(sys.argv)
    # for i, (data_cube, label_cube) in enumerate(zip(data_cubes, label_cubes)):
    #     print 'cube #%i/%i' % (i + 1, len(data_cubes))
    #     print data_cube
        # seg_viewer.show(data_cube, 200 * label_cube, app=app)

    cube_dir = '/home/tomas/Data/medical/dataset/gt/cubes_50/180_venous'
    check_data(cube_dir)