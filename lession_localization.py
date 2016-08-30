from __future__ import division

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import saliency_fish as sf

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in %s. You need to import package imtools: https://github.com/mjirik/imtools'\
          % os.path.basename(__file__)
    sys.exit(0)


def calc_saliencies(data, mask, show=False, show_now=True):
    use_sigmoid = True
    morph_proc = True
    print 'calculating:',
    print 'int diff ...'
    id = sf.conspicuity_calculation(data, sal_type='int_diff', mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc)
    print 'int hist ...',
    ih = sf.conspicuity_calculation(data, sal_type='int_hist', mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc)
    print 'int glcm ...',
    ig = sf.conspicuity_calculation(data, sal_type='int_glcm', mask=mask, calc_features=False, use_sigmoid=use_sigmoid, morph_proc=morph_proc)
    print 'int sliwin ...',
    isw = sf.conspicuity_calculation(data, sal_type='int_sliwin', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
    print 'texture ...',
    t = sf.conspicuity_calculation(data, sal_type='texture', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=False)
    print 'circloids ...',
    circ = sf.conspicuity_calculation(data, sal_type='circloids', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True)
    print 'blobs ...',
    blobs = sf.conspicuity_calculation(data, sal_type='blobs', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True)
    print 'done'

    if show:
        imgs = [(data, 'input'), (id, 'int diff'), (ih, 'int hist'), (ig, 'int glcm'), (isw, 'int sliwin'), (t, 'texture'),
                (circ, 'circloiods'), (blobs, 'blobs')]
        plt.figure()
        n_imgs = len(imgs)
        for i, (im, tit) in enumerate(imgs):
            plt.subplot(201 + 10 * n_imgs + i)
            if i == 0:
                cmap = 'gray'
            else:
                cmap = 'jet'
            plt.imshow(im, cmap, interpolation='nearest')
            plt.title(tit)
        if show_now:
            plt.show()


def localize(data, mask, show=False, show_now=True):
    sals = calc_saliencies(data, mask, show=show, show_now=show_now)


################################################################################
################################################################################
if __name__ == '__main__':
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

    data, mask, _ = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/183a_venous-GT.pklz')
    slice_ind = 17
    data = data[slice_ind,...]
    mask = mask[slice_ind,...]
    data = tools.windowing(data)

    localize(data, mask, show=True)

    # tools.show_3d(data, range=False)
    # plt.figure()
    # plt.imshow(data, 'gray', interpolation='nearest')
    # plt.show()