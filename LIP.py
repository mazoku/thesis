from __future__ import division

import skimage.morphology as skimor
import matplotlib.pyplot as plt
import numpy as np

import io3d
import tools


def get_circle(im, pt, rad=3):
    mask = skimor.disk(rad) - skimor.binary_erosion(skimor.disk(rad))
    pts = np.nonzero(mask)
    # pts[0] += pt[1]
    # pts[1] += pt[0]
    pts_shifted = [pts[0] + pt[1] - rad, pts[1] + pt[0] - rad]
    ints = im[pts_shifted]

    return ints, pts_shifted, mask


def get_pt_interactively(data, windowing=False, win_l=50, win_w=350):
    if windowing:
        vmin = win_l - win_w / 2
        vmax = win_l + win_w / 2
    else:
        vmin = data.min()
        vmax = data.max()

    f = plt.figure()
    plt.imshow(data, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.axis('image')
    pt = plt.ginput(1, timeout=0)  # returns list of tuples
    plt.close(f)

    if len(pt) != 1:
        print 'Need exactly one point.'
        return

    pt = (int(round(pt[0][0])), int(round(pt[0][1])))

    return pt


def run(data, rad=3, pt=None, windowing=False):
    if pt is None:
        pt = get_pt_interactively(data, windowing=windowing)
    ints, pts, mask = get_circle(data, pt, rad)

    print 'int0 = ', data[pt[1], pt[0]], ', ints: ', ints

    tools.show_slice(data, windowing=windowing, show=False)
    plt.hold(True)
    for i in range(len(pts[0])):
        plt.plot(pts[1][i], pts[0][i], 'ro')

    plt.show()


################################################################################
if __name__ == '__main__':
    rad = 1

    # slice = 17
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    #
    # dr = io3d.DataReader()
    # datap = dr.Get3DData(fname, dataplus_format=True)
    # data = datap['data3d'][slice, ...]
    # mask = datap['segmentation'][slice, ...]

    data = np.array([[  5,   8,  12,   4,   8,   6,   9,   0,   5,   8],
                     [  1,  10,  16,   5,   2,   0,   2,   8,   7,   5],
                     [  9,   2,   6,   8,   2,  10,   7,   1,   9,  11],
                     [  7,   6,  10,  15,  58,  90,  12,   9,  15,   6],
                     [ 12,   8,  11, 130, 200, 110,  96,  12,  18,   1],
                     [  8,   9,  26,  99, 120,  80,  78,  20,   5,  11],
                     [ 20,  12,  14, 150, 160,  90, 100,  16,  11,  15],
                     [ 10,  13,   9,  12, 136, 120, 110,  20,   8,   8],
                     [ 11,  13,  11,  10,  76,  60,   9,   0,  10,  36],
                     [  9,  11,  18,   4,   7,  20,  18,  14,   9,   1]])

    run(data, rad=rad, windowing=False)