from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import lession_localization as lesloc

import os
import sys

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in %s. You need to import package imtools: https://github.com/mjirik/imtools'\
          % os.path.basename(__file__)
    sys.exit(0)


def reshape_struc(data):
    data_out = []
    for im1, im2, slics in data:
        for s1, s2 in slics:
            data_out.append((im1, im2, s1, s2))
    return data_out


def find_file(q_fname, q_sl, files):
    base_fn = q_fname.split('/')[-1].split('.')[0]
    fnames = [x for x in files if base_fn in x and '-%i-' % q_sl in x]
    return fnames


################################################################################
################################################################################
if __name__ == '__main__':
    datas = lesloc.get_data_struc()
    datas = reshape_struc(datas)

    files = []
    datadir = '/home/tomas/Dropbox/Data/medical/dataset/gt/salmaps/vyber/hypo/res_pklz'
    for (dirpath, dirnames, filenames) in os.walk(datadir):
        filenames = [x for x in filenames if x.split('.')[-1] == 'pklz']
        for fname in filenames:
            files.append(os.path.join(dirpath, fname))
        break

    for im1, im2, s1, s2 in datas:
        fname1 = find_file(im1, s1, files)
        fname2 = find_file(im2, s2, files)
        if fname1 and fname2:
            data, gt_mask, voxel_size = tools.load_pickle_data(f)
            plt.figure()
            plt.subplot(121), plt.imshow()