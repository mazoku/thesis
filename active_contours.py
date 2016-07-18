from __future__ import division

import numpy as np
import scipy.stats as scista
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os

import cv2
import scipy.ndimage.filters as scindifil
import skimage.transform as skitra
import skimage.feature as skifea
import skimage.morphology as skimor
import skimage.measure as skimea
import skimage.segmentation as skiseg
import skimage.exposure as skiexp
from skimage import img_as_float

from collections import namedtuple

import lankton_lls

if os.path.exists('../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'Import error in active_contours.py. You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

if os.path.exists('../growcut/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../growcut/')
    from growcut import growcut
else:
    print 'You need to import package growcut: https://github.com/mazoku/growcut'
    sys.exit(0)


def initialize(data, dens_min=0, dens_max=255, prob_c=0.2, prob_c2=0.01, show=False, show_now=True):
    _, liver_rv = tools.dominant_class(data, dens_min=dens_min, dens_max=dens_max, show=show, show_now=show_now)

    liver_prob = liver_rv.pdf(data)

    prob_c = 0.2
    prob_c_2 = 0.01
    seeds1 = liver_prob > (liver_prob.max() * prob_c)
    seeds2 = liver_prob <= (np.median(liver_prob) * prob_c_2)
    seeds = seeds1 + 2 * seeds2

    plt.figure()
    plt.subplot(131), plt.imshow(data, 'gray')
    plt.subplot(132), plt.imshow(liver_prob, 'gray')
    plt.subplot(133), plt.imshow(seeds, 'gray')
    plt.show()

    gc = growcut.GrowCut(data, seeds, smooth_cell=False, enemies_T=0.7)
    gc.run()

    labs = gc.get_labeled_im().astype(np.uint8)
    labs = scindifil.median_filter(labs, size=3)

    return labs


def initialize_graycom(data, slice=None, distances=[1,], scale=0.5, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=False, c_t=5,
                       show=False, show_now=True):
    if scale != 1:
        data = tools.resize3D(data, scale, sliceId=0)

    # computing gray co-occurence matrix
    print 'Computing gray co-occurence matrix ...',
    # gcm = skifea.greycomatrix(data, distances, angles, symmetric=symmetric)
    # summing over distances and directions
    # gcm = gcm.sum(axis=3).sum(axis=2)
    gcm = tools.graycomatrix_3D(data, connectivity=1)
    print 'done'

    # plt.figure()
    # plt.subplot(121), plt.imshow(gcm, 'jet', vmax=10 * gcm.mean())
    # plt.subplot(122), plt.imshow(gcm2, 'jet', vmax=10 * gcm.mean())
    # plt.show()

    # ----
    # gcm2 = skifea.greycomatrix(data[10,...], distances, angles, symmetric=symmetric).sum(axis=3).sum(axis=2)
    # plt.figure()
    # plt.subplot(121), plt.imshow(gcm, 'jet', vmax=10 * gcm.mean()), plt.title('3D gcm')
    # plt.subplot(122), plt.imshow(gcm2, 'jet', vmax=10 * gcm2.mean()), plt.title('gcm of a slice')
    # plt.show()
    # ----

    # plt.figure()
    # thresh = np.mean(gcm)
    # ts = (thresh * np.array([1, 3, 6, 9, 12])).astype(int)
    # for i, t in enumerate(ts):
    #     plt.subplot(100 + 10 * len(ts) + i + 1)
    #     plt.imshow(gcm > t, 'gray')
    #     plt.title('t = %i' % t)
    # plt.show()

    # thresholding graycomatrix (GCM)
    thresh = c_t * np.mean(gcm)
    gcm_t = gcm > thresh
    gcm_to = skimor.binary_opening(gcm_t, selem=skimor.disk(3))

    # find peaks in the GCM and return them as random variables
    rvs = analyze_gcm(gcm_to)

    if slice is not None:
        data = data[slice,...]

    # deriving seed points
    seeds = np.zeros(data.shape, dtype=np.uint8)
    best_probs = np.zeros(data.shape)  # assign the most probable label if more than one are possible
    for i, rv in enumerate(rvs):
        probs = rv.pdf(data)
        s = probs > probs.mean()  # probability threshold
        s = np.where((probs * s) > (best_probs * s), i + 1, s)  # assign new label only if its probability is higher
        best_probs = np.where(s, probs, best_probs)  # update best probs
        seeds = np.where(s, i + 1, seeds)  # update seeds

    # running Grow cut
    gc = growcut.GrowCut(data, seeds, smooth_cell=False, enemies_T=0.7)
    gc.run()

    # postprocessing labels
    labs = gc.get_labeled_im().astype(np.uint8)[0,...]
    labs_f = scindifil.median_filter(labs, size=5)

    plt.figure()
    plt.subplot(131), plt.imshow(gcm, 'gray', vmax=gcm.mean()), plt.title('gcm')
    plt.subplot(132), plt.imshow(gcm_t, 'gray'), plt.title('thresholded')
    plt.subplot(133), plt.imshow(gcm_to, 'gray'), plt.title('opened')

    plt.figure()
    plt.subplot(121), plt.imshow(data, 'gray', interpolation='nearest'), plt.title('input')
    plt.subplot(122), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.title('seeds')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cax=cax, ticks=np.unique(seeds))

    plt.figure()
    plt.subplot(141), plt.imshow(data, 'gray'), plt.title('input')
    plt.subplot(142), plt.imshow(labs_f, 'gray'), plt.title('labels')
    # plt.subplot(143), plt.imshow(liver_blob, 'gray'), plt.title('liver blob')
    # plt.subplot(144), plt.imshow(init_mask, 'gray'), plt.title('init mask')

    # finding liver blob
    liver_blob = find_liver_blob(data, labs_f, show=show, show_now=show_now)

    # hole filling - adding (and then removing) a capsule of zeros, otherwise it'd fill holes touching image borders
    init_mask = np.zeros([x + 2 for x in liver_blob.shape], dtype=np.uint8)
    if liver_blob.ndim == 3:
        for i, im in enumerate(liver_blob):
            init_mask[i + 1, 1:-1, 1:-1] = im
    else:
        init_mask[1:-1, 1:-1] = liver_blob
    init_mask = skimor.remove_small_holes(init_mask, min_size=0.1 * liver_blob.sum(), connectivity=2)
    if liver_blob.ndim == 3:
        init_mask = init_mask[1:-1, 1:-1, 1:-1]
    else:
        init_mask = init_mask[1:-1, 1:-1]

    # visualization
    if show:
        plt.figure()
        plt.subplot(131), plt.imshow(gcm, 'gray', vmax=gcm.mean()), plt.title('gcm')
        plt.subplot(132), plt.imshow(gcm_t, 'gray'), plt.title('thresholded')
        plt.subplot(133), plt.imshow(gcm_to, 'gray'), plt.title('opened')

        plt.figure()
        plt.subplot(121), plt.imshow(data, 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(122), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.title('seeds')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax, ticks=np.unique(seeds))

        plt.figure()
        plt.subplot(141), plt.imshow(data, 'gray'), plt.title('input')
        plt.subplot(142), plt.imshow(labs_f, 'gray'), plt.title('labels')
        plt.subplot(143), plt.imshow(liver_blob, 'gray'), plt.title('liver blob')
        plt.subplot(144), plt.imshow(init_mask, 'gray'), plt.title('init mask')

        if show_now:
            plt.show()

    return init_mask


def find_liver_blob(data, labs_im, dens_min=120, dens_max=200, show=False, show_now=True):
    lbls = np.unique(labs_im)
    adepts = np.zeros_like(labs_im)
    for l in lbls:
        mask = labs_im == l
        mean_int = data[np.nonzero(mask)].mean()
        if not dens_min < mean_int < dens_max:
            print 'mean value: {:.1f} - SKIPPING'.format(mean_int)
            continue
        else:
            print 'mean value: {:.1f} - OK'.format(mean_int)
            adepts += l * mask

    adepts = skimor.binary_opening(adepts > 0, selem=skimor.disk(3))

    adepts_lbl, n_labels = skimea.label(adepts, connectivity=2, return_num=True)
    areas = [(adepts_lbl == l).sum() for l in range(1, n_labels)]
    winner = adepts_lbl == (np.argmax(areas) + 1)

    if show:
        plt.figure()
        plt.subplot(131), plt.imshow(labs_im, 'jet'), plt.title('labs_im')
        plt.subplot(132), plt.imshow(adepts_lbl, 'jet', vmax=labs_im.max()), plt.title('adepts')
        plt.subplot(133), plt.imshow(winner, 'jet', vmax=labs_im.max()), plt.title('liver blob')
        if show_now:
            plt.show()

    return winner


def analyze_gcm(gcm, area_t=200, ecc_t=0.35):
    labs_im = skimea.label(gcm, connectivity=2)

    # plt.figure()
    # plt.subplot(121), plt.imshow(gcm, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(labs_im, 'jet', interpolation='nearest')
    # plt.show()

    blobs = describe_blob(labs_im, area_t=area_t, ecc_t=ecc_t)
    means = [np.array(b.centroid).mean() for b in blobs]
    stds = 5 / np.array([b.eccentricity for b in blobs])

    rvs = [scista.norm(m, s) for m, s in zip(means, stds)]

    return rvs


def describe_blob(labs_im, area_t=200, ecc_t=0.25):
    # TODO: misto ecc kontrolovat jen major_axis?
    props = skimea.regionprops(labs_im)
    blobs = []
    blob = namedtuple('blob', ['label', 'area', 'centroid', 'eccentricity'])
    for i, prop in enumerate(props):
        label = prop.label
        area = int(prop.area)
        centroid = map(int, prop.centroid)
        major_axis = prop.major_axis_length
        minor_axis = prop.minor_axis_length
        # my_ecc = minor_axis / major_axis
        try:
            eccentricity = minor_axis / major_axis
        except ZeroDivisionError:
            eccentricity = 0
        print '#{}: area={}, centroid={}, eccentricity={:.2f}'.format(i, area, centroid, eccentricity),

        if (area > area_t) and (eccentricity > ecc_t):
            blobs.append(blob(label, area, centroid, eccentricity))
            print '... OK'
        elif area <= area_t:
            print '... TO SMALL - DISCARDING'
        elif eccentricity <= ecc_t:
            print '... SPLITTING'
            splitted = split_blob(labs_im == label, prop)
            for spl in splitted:
                spl_blob = describe_blob(spl)
                blobs += spl_blob
                pass

    return blobs


def split_blob(im, prop):
    # blobs = []
    # blob = namedtuple('blob', ['label', 'area', 'centroid', 'eccentricity'])
    centroid = tuple(map(int, np.round(prop.centroid)))
    major_axis = int(round(prop.major_axis_length / 2))
    minor_axis = int(round(prop.minor_axis_length / 2))
    angle = int(round(np.degrees(prop.orientation)))

    c1 = centroid[1] + major_axis * np.cos(prop.orientation)
    r1 = centroid[0] - major_axis * np.sin(prop.orientation)
    c2 = centroid[1] + major_axis * np.cos(prop.orientation + np.pi)
    r2 = centroid[0] - major_axis * np.sin(prop.orientation + np.pi)

    new_cent1 = ((centroid[0] + r1) / 2, (centroid[1] + c1) / 2)
    new_cent1 = tuple(map(int, map(round, new_cent1)))
    new_cent2 = ((centroid[0] + r2) / 2, (centroid[1] + c2) / 2)
    new_cent2 = tuple(map(int, map(round, new_cent2)))


    # imv = cv2.cvtColor(255 * im.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # cv2.ellipse(imv, centroid, (minor_axis, major_axis), angle, 0, 360, (0, 0, 255), thickness=2)
    # cv2.ellipse(imv, new_cent1, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, (255, 0, 0), thickness=2)
    # cv2.ellipse(imv, new_cent2, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, (255, 0, 0), thickness=2)
    #
    # plt.figure()
    # plt.imshow(imv, 'gray', interpolation='nearest')
    # plt.hold(True)
    # plt.plot(centroid[0], centroid[1], 'ro')
    # plt.plot(c1, r1, 'go')
    # plt.plot(c2, r2, 'go')
    # plt.plot(new_cent1[0], new_cent1[1], 'co')
    # plt.plot(new_cent2[0], new_cent2[1], 'yo')
    # plt.axis('image')
    # plt.show()

    im1 = np.zeros_like(im).astype(np.uint8)
    cv2.ellipse(im1, new_cent1, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, 1, thickness=-1)
    im1 *= im

    im2 = np.zeros_like(im).astype(np.uint8)
    cv2.ellipse(im2, new_cent2, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, 1, thickness=-1)
    im2 *= im

    # plt.figure()
    # plt.subplot(121), plt.imshow(im1, 'gray')
    # plt.subplot(122), plt.imshow(im2, 'gray')
    # plt.show()
    #
    # blob1 = blob(prop.label, prop.area, new_cent1, 0.5)
    # blob2 = blob(prop.label, prop.area, new_cent2, 0.5)

    # return [blob1, blob2]
    return (im1, im2)


def lankton_ls(im, mask, method='sfm', max_iters=1000):
    im = skitra.rescale(im, scale=0.5, preserve_range=True)
    mask = skitra.rescale(mask, scale=0.5, preserve_range=True).astype(np.bool)
    lankton_lls.run(im, mask, method='sfm', max_iter=1000)


def run(im, slice=None, mask=None, smoothing=True, show=False, show_now=True, save_fig=False, verbose=True):
    if smoothing:
        im = tools.smoothing(im, sigmaSpace=10, sigmaColor=10, sliceId=0)
    # init_mask = initialize(im, dens_min=50, dens_max=200, show=True, show_now=False)[0,...]
    if mask is None:
        # mask = initialize_graycom(im, [1, 2], scale=1, show=show, show_now=show_now)
        mask = initialize_graycom(im, slice, distances=[1,], scale=0.5, show=show, show_now=show_now)

    plt.show()
    # plt.figure()
    # plt.subplot(121), plt.imshow(im, 'gray'), plt.title('input')
    # plt.subplot(122), plt.imshow(mask, 'gray'), plt.title('init mask')
    # plt.show()

    lankton_ls(im, mask, method='sfm', max_iters=1000)

    # init_mask = np.zeros(im.shape, dtype=np.bool)
    # init_mask[37:213, 89:227] = 1
    # localized_seg(im, init_mask)
    # init = initialize(im, dens_min=10, dens_max=245)


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    verbose = True
    show = True
    show_now = False

    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    # 2D
    slice_ind = 10
    # data = data[slice_ind, :, :]
    # data = tools.windowing(data)

    # 3D
    data = tools.windowing(data)

    run(data, slice=slice_ind, mask=None, smoothing=True, save_fig=False, show=show, show_now=show_now, verbose=verbose)