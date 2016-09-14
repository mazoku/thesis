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


################################################################################
################################################################################
if __name__ == '__main__':
    input_fnames = lesloc.get_data_struc()
    input_fnames = reshape_struc(input_fnames)



    for im1, im2, s1, s2 in data:
        data, gt_mask, voxel_size = tools.load_pickle_data(f)
        plt.figure()
        plt.subplot(121), plt.imshow()