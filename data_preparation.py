from __future__ import division

#TODO: test na syntetickych datech

import sys
import os
import pickle
import gzip

import numpy as np
from PyQt4 import QtCore, QtGui
import matplotlib.pyplot as plt

import tensorflow as tf

from skdata.mnist.views import OfficialVectorClassification
from tqdm import tqdm

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


def data2files(data_dir, cube_shape, step, shape=(50, 50), resize_labels_fac=1):
    data_names = []
    for (rootDir, dirNames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename[-5:] == '.pklz':
                data_names.append(filename)

    for i, dirname in enumerate(data_names):
        print 'Processing dir #%i/%i, %s:' % (i + 1, len(data_names), dirname),
        data_path = os.path.join(data_dir, dirname)
        print 'slicing ...',
        data_cubes, label_cubes = slice_data(data_path, reshape=False, cube_shape=cube_shape, step=step)

        print 'saving to list ...',
        # data po rezech do listu
        data_cubes_resh = []
        for d in data_cubes:
            # print d.shape
            for s in d:
                data_cubes_resh.append(misc.resize_to_shape(s, shape=shape))

        # labely po rezech do listu
        label_cubes_resh = []
        for d in label_cubes:
            # print d.shape
            for s in d:
                new_shape = tuple((x / resize_labels_fac for x in shape))
                lab = misc.resize_to_shape(s, shape=new_shape)
                label_cubes_resh.append(lab)

        posit_flag = np.array(label_cubes_resh).sum(axis=1).sum(axis=1) > 0
        perc_posit = 100 * (posit_flag).sum() / len(posit_flag)
        print 'rel. posit. = %.1f%% ... ' % perc_posit,
        print 'done'

    # data na disk
    res_dir = '/home/tomas/Data/medical/dataset/gt/slicewise'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    print 'Saving data file ...',
    data_fname = os.path.join(res_dir, 'data.npz')
    data_file = gzip.open(data_fname, 'wb', compresslevel=1)
    np.save(data_file, np.array(data_cubes_resh))
    data_file.close()
    print 'done'

    print 'Saving label file ...',
    label_fname = os.path.join(res_dir, 'labels.npz')
    label_file = gzip.open(label_fname, 'wb', compresslevel=1)
    np.save(label_file, np.array(label_cubes_resh))
    label_file.close()
    print 'done'


def data2TF_records(data, labels, tfrecords_path):
    # data = OfficialVectorClassification()
    # trIdx = data.sel_idxs[:]
    trIdx = np.arange(labels.shape[0])

    writer = tf.python_io.TFRecordWriter(tfrecords_path)
    np.random.shuffle(trIdx)
    print 'Saving tf records ...',
    # iterate over each example
    # wrap with tqdm for a progress bar
    for example_idx in tqdm(trIdx):
        features = data[example_idx, :, :][:].flatten()
        label = labels[example_idx, :, :][:].flatten()

        # features2 = data.all_vectors[example_idx]
        # label2 = data.all_labels[example_idx]

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=label.astype("int64"))),
                    'image': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=features.astype("int64"))),
                }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    print 'done'


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
    if reshape:
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

    # cube_dir = '/home/tomas/Data/medical/dataset/gt/cubes_50/180_venous'
    # check_data(cube_dir)

    # slicing data to cubes and writing to file
    cube_shape = (1, 50, 50)
    step = 25
    shape = (60, 60)
    resize_labels_fac = 3
    data2files(data_dir, cube_shape, step=(1, step, step), shape=shape, resize_labels_fac=resize_labels_fac)

    # serializing data to TFRecords format and writing to file
    data_path = '/home/tomas/Data/medical/dataset/gt/slicewise/data.npz'
    label_path = '/home/tomas/Data/medical/dataset/gt/slicewise/labels.npz'
    tfrecords_path = '/home/tomas/Data/medical/dataset/gt/slicewise/data.tfrecords'

    data_file = gzip.open(data_path, 'rb', compresslevel=1)
    data = np.load(data_file)
    data_file.close()

    label_file = gzip.open(label_path, 'rb', compresslevel=1)
    labels = np.load(label_file)
    label_file.close()

    data2TF_records(data, labels, tfrecords_path)