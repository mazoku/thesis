from __future__ import division

import skimage.morphology as skimor
import skimage.filters as skifil
import skimage.restoration as skires
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scindi

import io3d
import tools


def get_circle(im, pt, rad=3):
    max_x = im.shape[1] - 1
    max_y = im.shape[0] - 1
    mask = skimor.disk(rad)# - skimor.binary_erosion(skimor.disk(rad))
    pts = np.nonzero(mask)
    # pts[0] += pt[1]
    # pts[1] += pt[0]
    pts_shifted = np.array([pts[0] + pt[1] - rad, pts[1] + pt[0] - rad])
    # idx_trimmed = [i.all() for i in pts_shifted.transpose() < np.array([max_x, max_y])]
    pts_trimmed = np.array([x for x in pts_shifted.transpose() if 0 <= x[0] <= max_x and 0 <= x[1] <= max_y]).transpose()
    pts_trimmed = [x for x in pts_trimmed]
    # pts_trimmed = [pt for pt in pts_shifted.transpose() if 0 <= pt[0] <= max_x and 0 <= pt[1] <= max_y]
    # x_trimed = [x for x in pts_shifted[0] if 0 <= x <= max_x]
    # y_trimed = [x for x in pts_shifted[1] if 0 <= x <= max_y]
    ints = im[pts_trimmed]

    return ints, pts_trimmed, mask


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


def lip_im(data, rad=3):
    x_coords, y_coords = np.indices(data.shape)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    n_pts = len(x_coords)

    ad_im = np.zeros(data.shape)
    ssd_im = np.zeros(data.shape)
    var_im = np.zeros(data.shape)
    for i in range(n_pts):
        ad, ssd, var = run(data, rad=rad, pt=(y_coords[i], x_coords[i]))
        ad_im[y_coords[i], x_coords[i]] = ad
        ssd_im[y_coords[i], x_coords[i]] = ssd
        var_im[y_coords[i], x_coords[i]] = var

    kernel = skimor.disk(rad)
    ad_im_m = scindi.filters.convolve(ad_im, kernel)
    ssd_im_m = scindi.filters.convolve(ssd_im, kernel)
    var_im_m = scindi.filters.convolve(var_im, kernel)

    plt.figure()
    plt.subplot(121), plt.imshow(ad_im, 'gray', interpolation='nearest'), plt.title('abs diff')
    plt.subplot(122), plt.imshow(ad_im_m, 'gray', interpolation='nearest'), plt.title('mean of abs diff')

    plt.figure()
    plt.subplot(121), plt.imshow(ssd_im, 'gray', interpolation='nearest'), plt.title('sum of squared diff')
    plt.subplot(122), plt.imshow(ssd_im_m, 'gray', interpolation='nearest'), plt.title('mean of sum of squared diff')

    plt.figure()
    plt.subplot(121), plt.imshow(var_im, 'gray', interpolation='nearest'), plt.title('var')
    plt.subplot(122), plt.imshow(var_im_m, 'gray', interpolation='nearest'), plt.title('mean of var')

    plt.show()


def run(data, rad=3, pt=None, windowing=False, show=False):
    if pt is None:
        pt = get_pt_interactively(data, windowing=windowing)
    ints, pts, mask = get_circle(data, pt, rad)

    int0 = data[pt[1], pt[0]]

    # print 'int0 =', int0, ', ints =', ints
    ad = np.abs(ints - int0)
    try:
        ssd = np.dot(ad, ad)
    except:
        pass
    var = np.var(ad)
    # print 'ssd =', ssd, ', absd =', ad

    if show:
        tools.show_slice(data, windowing=windowing, show=False)
        plt.hold(True)
        for i in range(len(pts[0])):
            plt.plot(pts[1][i], pts[0][i], 'ro')
        plt.show()

    return ad.mean(), ssd, var

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
                     [  9,  11,  18,   4,   7,  20,  18,  14,   9,   1]], dtype=np.float)

    # data_s = skifil.gaussian_filter(data, sigma=0.5)
    data_s = skires.denoise_tv_chambolle(data)

    plt.figure()
    plt.gray()
    plt.subplot(121), plt.imshow(data, interpolation='nearest'), plt.title('input')
    plt.subplot(122), plt.imshow(data_s, interpolation='nearest'), plt.title('gaussian filter')

    # run(data, rad=rad, windowing=False)
    lip_im(data_s, rad=rad)