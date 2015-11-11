from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scindi

import io3d
import tools


def compact_shape(data, mask_size=5):
    comp = scindi.filters.uniform_filter(data, size=mask_size)

    return comp


def run(data, mask):
    comp = compact_shape(data)

    tools.show_slice(comp, windowing=True)


################################################################################
if __name__ == '__main__':
    slice = 17
    fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'

    dr = io3d.DataReader()
    datap = dr.Get3DData(fname, dataplus_format=True)
    data = datap['data3d']
    mask = datap['segmentation']

    run(data[slice, ...], mask[slice, ...])
