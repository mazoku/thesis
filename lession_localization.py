from __future__ import division

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float

import saliency_fish as sf

from PyQt4 import QtGui

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in %s. You need to import package imtools: https://github.com/mjirik/imtools'\
          % os.path.basename(__file__)
    sys.exit(0)

if os.path.exists('/home/tomas/projects/data_viewers/'):
    sys.path.insert(0, '/home/tomas/projects/data_viewers/')
    from dataviewers.seg_viewer import SegViewer
else:
    print 'Import error in %s. You need to import package imtools: https://github.com/mazoku/dataviewers'\
          % os.path.basename(__file__)
    sys.exit(0)


def calc_saliencies(data, mask, smoothing=True, show=False, show_now=True):
    use_sigmoid = True
    morph_proc = True

    mean_v = int(data[np.nonzero(mask)].mean())
    data = np.where(mask, data, mean_v)

    if smoothing:
        data = tools.smoothing(data)

    data_struc = [(data, 'input'),]
    data = img_as_float(data)
    print 'calculating:',
    print 'int diff ...',
    id = sf.conspicuity_calculation(data, sal_type='int_diff', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
    data_struc.append((id, 'int diff'))
    print 'int hist ...',
    ih = sf.conspicuity_calculation(data, sal_type='int_hist', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
    data_struc.append((ih, 'int hist'))
    print 'int glcm ...',
    ig = sf.conspicuity_calculation(data, sal_type='int_glcm', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
    data_struc.append((ig, 'int glcm'))
    print 'int sliwin ...',
    isw = sf.conspicuity_calculation(data, sal_type='int_sliwin', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
    data_struc.append((isw, 'int sliwin'))
    print 'texture ...',
    t = sf.conspicuity_calculation(data, sal_type='texture', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=False)
    data_struc.append((t, 'texture'))
    print 'circloids ...',
    circ = sf.conspicuity_calculation(data, sal_type='circloids', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True)
    data_struc.append((circ, 'circloids'))
    print 'blobs ...',
    blobs = sf.conspicuity_calculation(data, sal_type='blobs', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True, verbose=False)
    data_struc.append((blobs, 'blobs'))
    print 'done'

    if show:
        plt.figure()
        n_imgs = len(data_struc)
        for i, (im, tit) in enumerate(data_struc):
            # r = 1 + int(i > (n_imgs / 2))
            plt.subplot(201 + 10 * (np.ceil(n_imgs / 2)) + i)
            if i == 0:
                cmap = 'gray'
            else:
                cmap = 'jet'
            plt.imshow(im, cmap, interpolation='nearest')
            plt.title(tit)
        if show_now:
            plt.show()


def localize(data, mask, show=False, show_now=True):
    data, mask = tools.crop_to_bbox(data, mask>0)

    sals = calc_saliencies(data, mask, show=show, show_now=show_now)


def get_data_struc():
    '''
    datastruc: [(venous data, arterial data, slices, offset), ...]
    '''
    dirpath = '/home/tomas/Dropbox/Data/medical/dataset/gt/'
    s = 4
    data_struc = [('180_venous-GT.pklz', '180_arterial-GT.pklz', range(6,14), 0),
                  ('183a_venous-GT.pklz', '183a_arterial-GT.pklz', range(11, 24, s), -1),
                  ('185a_venous-GT.pklz', '185a_arterial-GT.pklz', range(9, 24, s), 1),
                  ('186a_venous-GT.pklz', '186a_arterial-GT.pklz', range(1, 5, s), 0),
                  ('189a_venous-GT.pklz', '189a_arterial-GT.pklz', range(12, 26, s), -1),
                  ('221_venous-GT.pklz', '221_arterial-GT.pklz', range(17, 23, s), 0),
                  ('222a_venous-GT.pklz', '222a_arterial-GT.pklz', range(0, 0, s), -2),
                  ('232_venous-GT.pklz', '232_arterial-GT.pklz', range(0, 0, s), -2),
                  ('234_venous-GT.pklz', '234_arterial-GT.pklz', range(0, 0, s), 0),
                  ('235_venous-GT.pklz', '235_arterial-GT.pklz', range(0, 0, s), 0)]

    data_struc = [(dirpath + x[0], dirpath + x[1], x[2], x[3]) for x in data_struc]

    return data_struc


################################################################################
################################################################################
if __name__ == '__main__':
    # zoomovani tak aby mela data stejny pocet rezu
    d1, m1, vs1 = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/222a_venous-GT.pklz')
    d2, m2, vs2 = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/222a_arterial-GT.pklz')
    d3 = tools.match_size(d2, d1.shape)
    m3 = tools.match_size(m2, m1.shape)

    datap = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/222a_arterial-GT.pklz', return_datap=True)

    slab = {'none': 0, 'liver': 1, 'lesions': 6}
    datap_1 = {'data3d': d1, 'segmentation': m1, 'voxelsize_mm': vs1, 'slab': slab}
    datap_2 = {'data3d': d3, 'segmentation': m3, 'voxelsize_mm': vs2, 'slab': slab}
    app = QtGui.QApplication(sys.argv)
    le = SegViewer(datap1=datap_1, datap2=datap_2)
    le.show()
    app.exec_()

    # vizualizace dat
    # ds = get_data_struc()
    #
    # # for i, (vfn, afn, slcs, off) in enumerate(ds):
    # i = 6
    # vfn, afn, slcs, off = ds[i]
    #
    # print i, ' ... ', vfn.split('/')[-1], afn.split('/')[-1]
    #
    # datap_1 = tools.load_pickle_data(vfn, return_datap=True)
    # datap_2 = tools.load_pickle_data(afn, return_datap=True)
    #
    # app = QtGui.QApplication(sys.argv)
    # le = SegViewer(datap1=datap_1, datap2=datap_2)
    # le.show()
    # app.exec_()


    # files = []
    # for (dirpath, dirnames, filenames) in os.walk('/home/tomas/Dropbox/Data/medical/dataset/gt'):
    #     filenames = [x for x in filenames if x.split('.')[-1] == 'pklz']
    #     for fname in filenames:
    #         files.append(os.path.join(dirpath, fname))
    #     break
    #
    # for i in range(1, len(files)): #39
    #     f = files[i]
    #     print i, f
    #     data, gt_mask, voxel_size = tools.load_pickle_data(f)
    #     data = tools.windowing(data)
    #     tools.show_3d(data, range=False)

    # data, mask, _ = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/183a_venous-GT.pklz')
    # slice_ind = 17
    # data = data[slice_ind,...]
    # mask = mask[slice_ind,...]
    # data = tools.windowing(data)
    #
    # localize(data, mask, show=True)

    # tools.show_3d(data, range=False)
    # plt.figure()
    # plt.imshow(data, 'gray', interpolation='nearest')
    # plt.show()