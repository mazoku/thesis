from __future__ import division

import os
import sys

import numpy as np
import skimage.morphology as skimor
import skimage.measure as skimea
import scipy.ndimage as scindi

import matplotlib.pyplot as plt

from PyQt4 import QtCore, QtGui

import pickle
import gzip

from collections import namedtuple

# if os.path.exists('../imtools/'):
#     sys.path.append('../imtools/')
#     from imtools import tools
# else:
#     print 'You need to import package imtools: https://github.com/mjirik/imtools'
#     sys.exit(0)

if os.path.exists('../data_viewers/'):
    sys.path.append('../data_viewers/')
    from dataviewers import seg_viewer, viewer_3D
else:
    print 'You need to import package dataviewers: https://github.com/mazoku/data_viewers'
    sys.exit(0)

# if os.path.exists('../imtools/'):
#     sys.path.append('../imtools/')
#     from imtools import tools
# else:
#     print 'You need to import package imtools: https://github.com/mjirik/imtools'
#     sys.exit(0)

import imp
tools = imp.load_source('tools', '../imtools/imtools/tools.py')
# import tools


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

BGD_LABEL = 0
HEALTHY_LABEL = 2
HYPO_LABEL = 1
HYPER_LABEL = 3


def get_blobs(data, min_area=50):
    data = tools.windowing(data.copy(), sliceId=0)
    data = data == 0
    data = tools.opening3D(data, sliceId=0)

    labels, num = skimea.label(data, return_num=True)
    blobs = []
    blobNT = namedtuple('blob', ['area', 'cent', 'width', 'height'])

    for i in range(1, num + 1):
        lab = labels == i
        area = lab.sum()
        if area > min_area:
            pts = np.nonzero(lab)
            # cent = np.array([pts[0].mean(), pts[1].mean(), pts[2].mean()])
            cent = np.array([coor.mean() for coor in pts])
            width = pts[1].max() - pts[1].min()
            height = pts[0].max() - pts[0].min()
            blob = blobNT(area=area, cent=cent, width=width, height=height)
            blobs.append(blob)

    return blobs


def register_data(liver, lesion, lesion_m):#, max_area_diff=5, max_dist=20):
    liver_blobs = get_blobs(liver)
    lesion_blobs = get_blobs(lesion)

    dists = []
    for livblo in liver_blobs:
        for lesblo in lesion_blobs:
            area_diff = livblo.area - lesblo.area
            width_diff = livblo.width - lesblo.width
            height_diff = livblo.height - lesblo.height
            cent_dist = (livblo.cent - lesblo.cent).astype(np.int)
            if area_diff + width_diff + height_diff == 0:
                dists.append(cent_dist)

    unique = []
    counts = []
    for i in range(len(dists)):
        if not (np.any([(dists[i] == x).all() for x in unique])):
            unique.append(dists[i])
            c = 1
            for j in range(i + 1, len(dists)):
                if (dists[i] == dists[j]).all():
                    c += 1
            counts.append(c)

    best_dist = dists[np.argmax(counts)]

    pts = np.nonzero(lesion_m)
    pts_reg = [x + y for x, y in zip(pts, best_dist)]
    reg_les = np.zeros_like(liver)
    # coords_ok = [(0 <= pts_reg[0]) * (pts_reg[0] < reg_les.shape[0]),
    #              (0 <= pts_reg[1]) * (pts_reg[1] < reg_les.shape[1]),
    #              (0 <= pts_reg[2]) * (pts_reg[2] < reg_les.shape[2])]
    coords_ok = [(0 <= pts_reg[x]) * (pts_reg[x] < reg_les.shape[x]) for x in range(reg_les.ndim)]
    coords_ok = np.array(coords_ok).sum(0) == 3
    pts_reg_filt = [pts_reg[x][np.nonzero(coords_ok)] for x in range(reg_les.ndim)]
    reg_les[pts_reg_filt] = 1

    return reg_les


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

    # liver_fname =
    # lesion_fname =
    # nactu data a sloucim masky
    for i, (liver_fname, lesion_fname) in enumerate(dataset):
        print 'Merging %i/%i: %s, %s ...' % (i + 1, len(dataset), liver_fname, lesion_fname),
        liver_path = os.path.join(data_dir, liver_fname)
        lesion_path = os.path.join(data_dir, lesion_fname)
        liver_datap = tools.load_pickle_data(liver_path, return_datap=True)
        data_l = liver_datap['data3d']
        mask_l = liver_datap['segmentation'].astype(np.int)
        voxel_size_l = liver_datap['voxelsize_mm']
        data_t, mask_t, voxel_size_t = tools.load_pickle_data(lesion_path)

        # mask_t2 = scindi.binary_fill_holes(mask_t)
        mask_t = tools.fill_holes(mask_t, slicewise=True, slice_id=0)
        # for i in range(mask_t.shape[0]):
        #     plt.figure()
        #     plt.subplot(131), plt.imshow(mask_t[i, :, :], 'gray'), plt.title('mask_t, rez %i' % i)
        #     plt.subplot(132), plt.imshow(mask_t2[i, :, :], 'gray'), plt.title('mask_t2')
        #     plt.subplot(133), plt.imshow(scindi.binary_erosion(mask_t[i, :, :]), 'gray'), plt.title('filled')
        #     plt.show()
        # seg_viewer.show(255 * mask_t, 255 * mask_t2)

        # registruji data na sebe
        mask_t = register_data(data_l, data_t, mask_t)
        mask_t *= mask_l  # oznaceni lezi muze pretekat oznaceni jater -> proto toto maskovani

        mask_final = np.where(mask_l, HEALTHY_LABEL, BGD_LABEL)
        mask_final = np.where(mask_t, HYPO_LABEL, mask_final)

        # viewer_3D.show(mask_final, range=True)

        # print 'liver:', liver_path
        # print 'lesion:', lesion_path
        # seg_viewer.show(liver_path, lesion_path)

        ext = liver_fname[liver_fname.rfind('.'):]
        output_fname = liver_fname[:liver_fname.rfind('.')] + 'GT' + ext
        output_path = os.path.join(data_dir, 'gt', output_fname)
        if not os.path.exists(os.path.join(data_dir, 'gt')):
            os.mkdir(os.path.join(data_dir, 'gt'))

        data_dict['data3d'] = data_l
        # mask_final = np.where(mask_final == 2, liver_datap['slab']['lesions'], mask_final)
        data_dict['segmentation'] = mask_final

        # data_dict['segmentation'] = mask_t  # jatra maji 1, leze 2
        data_dict['voxelsize_mm'] = voxel_size_l
        data_dict['slab'] = liver_datap['slab']
        file = gzip.open(output_path, 'wb', compresslevel=1)
        pickle.dump(data_dict, file)
        file.close()
        # pickle.dump(data_dict, open(output_path, 'wb'))
        print 'done'
        # if i == 1:
        #     seg_viewer.show(liver_path, data_dict)

        # print 'liver:', liver_fname
        # print 'lesion:', lesion_fname
        # print 'ground truth: ', output_fname
        # print '----------------------------'


def check_data(data_dir):
    data = []
    # najdu vsechny pklz soubory
    for (rootDir, dirNames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename[-5:] == '.pklz':
                data.append(filename)

    app = QtGui.QApplication(sys.argv)
    for i, fname in enumerate(data):
        print 'Showing %i/%i: %s' % (i + 1, len(data), fname)
        seg_viewer.show(os.path.join(data_dir, fname), os.path.join(data_dir, fname), app=app)
        # datap = tools.load_pickle_data(os.path.join(data_dir, fname), return_datap=True)
        # print ', values:', np.unique(datap['segmentation'])

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
    # seg_viewer.show(datap_1, datap_2)

    #---------------------------------------------------------------
    # liver_path = '/home/tomas/Data/medical/dataset/183a_arterial-.pklz'
    # lesion_path = '/home/tomas/Data/medical/dataset/183a_arterial-leze.pklz'
    # seg_viewer.show(liver_path, lesion_path)

    #---------------------------------------------------------------
    data_dir = '/home/tomas/Data/medical/dataset'
    gt_dir = '/home/tomas/Data/medical/dataset/gt'

    res_fname = ''
    gt_fname = ''

    show = True
    show_now = False
    save_fig = False

    # merge_dataset(data_dir)

    # check merged data
    check_data(gt_dir)