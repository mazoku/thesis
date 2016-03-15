from __future__ import division

import os
import sys

from imtools import tools
import numpy as np
import skimage.morphology as skimor
import skimage.measure as skimea
import scipy.ndimage as scindi

from PyQt4 import QtCore, QtGui

import pickle

if os.path.exists('../data_viewers/'):
    sys.path.append('../data_viewers/')
    from dataviewers import seg_viewer
else:
    print 'You need to import package dataviewers: https://github.com/mazoku/data_viewers'
    sys.exit(0)


# def process_lesion_mask(mask):
#     mask_o = np.zeros_like(mask)
#     for id, slice in enumerate(mask):
#         if slice.max() == 1:
#             mask_o[id, :, :] = scindi.binary_fill_holes(slice)
        # labels, num = skimea.label(slice,return_num=True)
        # if num > 0:
        #     slice_o = np.zeros_like(slice)
        #     for lbl in range(1, num + 1):
        #         obj = labels == lbl
        #         slice_o += scindi.binary_fill_holes(obj)


def merge_dataset(data_dir):
    lesion_data = []
    liver_data = []
    data_dict = dict()

    # rozdelim soubory do dvou skupin: obsahujici leze a obsahujici jatra
    for (rootDir, dirNames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename[-5:] == '.pklz':
                if 'leze' in filename:
                    lesion_data.append(filename)
                else:
                    liver_data.append(filename)

    # dam dohromady prislusne liver a lesion soubory
    dataset = []
    for lesion_fname in lesion_data:
        ext = lesion_fname[lesion_fname.rfind('.'):]
        liver_fname = lesion_fname[:lesion_fname.find('leze')] + ext
        if liver_fname in liver_data:
            dataset.append([liver_fname, lesion_fname])
        else:
            print 'Cannot find liver data for \'%s\'' % lesion_fname

    # nactu data a sloucim masky
    for liver_fname, lesion_fname in dataset:
        liver_path = os.path.join(data_dir, liver_fname)
        lesion_path = os.path.join(data_dir, lesion_fname)
        data_l, mask_l, voxel_size_l = tools.load_pickle_data(liver_path)
        data_t, mask_t, voxel_size_t = tools.load_pickle_data(lesion_path)

        mask_t = scindi.binary_fill_holes(mask_t)

        seg_viewer.show(liver_path, lesion_path)

        ext = liver_fname[liver_fname.rfind('.'):]
        output_fname = liver_fname[:liver_fname.rfind('.')] + 'GT' + ext
        output_path = os.path.join(data_dir, 'gt', output_fname)
        if not os._exists(os.path.join(data_dir, 'gt')):
            os.mkdir(os.path.join(data_dir, 'gt'))

        data_dict['data3d'] = data_l
        data_dict['segmentation'] = mask_l + mask_t  # jatra maji 1, leze 2
        data_dict['voxelsize_mm'] = voxel_size_l
        pickle.dump(data_dict, open(output_path, 'wb'))

        print 'liver:', liver_fname
        print 'lesion:', lesion_fname
        print 'ground truth: ', output_fname
        print '----------------------------'


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # fname_1 = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_arterial_5.0_B30f-.pklz'
    # fname_2 = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    #
    # datap_1 = tools.load_pickle_data(fname_1, return_datap=True)
    # datap_2 = tools.load_pickle_data(fname_2, return_datap=True)
    # print datap_1['segmentation'].shape
    # print datap_2['segmentation'].shape
    #
    # # starting application
    # app = QtGui.QApplication(sys.argv)
    # le = SegViewer(datap1=datap_1, datap2=datap_2)
    # le.show()
    # sys.exit(app.exec_())

    #---------------------------------------------------------------
    data_dir = '/home/tomas/Data/medical/dataset'

    res_fname = ''
    gt_fname = ''

    show = True
    show_now = False
    save_fig = False

    merge_dataset(data_dir)