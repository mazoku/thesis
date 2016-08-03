from __future__ import division

import numpy as np
import scipy.stats as scista
import scipy.ndimage.morphology as scindimor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os
import glob
import itertools

import pickle
import gzip
import xlsxwriter

import cv2
import scipy.ndimage.filters as scindifil
import skimage.transform as skitra
import skimage.feature as skifea
import skimage.morphology as skimor
import skimage.measure as skimea
import skimage.segmentation as skiseg
import skimage.color as skicol
import skimage.exposure as skiexp
from skimage import img_as_float

from collections import namedtuple

import lankton_lls

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in active_contours.py. You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

if os.path.exists('/home/tomas/projects/growcut/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '/home/tomas/projects/growcut/')
    from growcut import growcut
else:
    print 'You need to import package growcut: https://github.com/mazoku/growcut'
    sys.exit(0)

sys.path.append('/home/tomas/projects/morphsnakes/')
import morphsnakes


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


def initialize_graycom_deprecated(data, slice=None, distances=[1,], scale=0.5, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=False, c_t=5,
                       show=False, show_now=True):
    if scale != 1:
        data = tools.resize3D(data, scale, sliceId=0)

    # computing gray co-occurence matrix
    print 'Computing gray co-occurence matrix ...',
    if data.ndim == 2:
        gcm = skifea.greycomatrix(data, distances, angles, symmetric=symmetric)
        # summing over distances and directions
        gcm = gcm.sum(axis=3).sum(axis=2)
    else:
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
    rvs = analyze_glcm(gcm_to)

    if slice is not None:
        data = data[slice,...]

    # deriving seed points
    seeds = np.zeros(data.shape, dtype=np.uint8)
    best_probs = np.zeros(data.shape)  # assign the most probable label if more than one are possible
    for i, rv in enumerate(rvs):
        probs = rv.pdf(data)
        s = probs > probs.mean()  # probability threshold
        # sfh = skimor.remove_small_holes(s, min_size=0.1 * s.sum(), connectivity=2)
        s = tools.fill_holes_watch_borders(s)
        # s = skimor.binary_opening(s, selem=skimor.disk(3))
        # plt.figure()
        # plt.subplot(131), plt.imshow(s, 'gray')
        # plt.subplot(132), plt.imshow(sfh, 'gray')
        # plt.subplot(133), plt.imshow(sfh2, 'gray')
        # plt.show()
        s = np.where((probs * s) > (best_probs * s), i + 1, s)  # assign new label only if its probability is higher
        best_probs = np.where(s, probs, best_probs)  # update best probs
        seeds = np.where(s, i + 1, seeds)  # update seeds

    # # running Grow cut
    # gc = growcut.GrowCut(data, seeds, smooth_cell=False, enemies_T=0.7)
    # gc.run()
    #
    # postprocessing labels
    # labs = gc.get_labeled_im().astype(np.uint8)[0,...]
    # labs_f = scindifil.median_filter(labs, size=5)
    labs_f = scindifil.median_filter(seeds, size=3)

    # plt.figure()
    # plt.subplot(131), plt.imshow(gcm, 'gray', vmax=gcm.mean()), plt.title('gcm')
    # plt.subplot(132), plt.imshow(gcm_t, 'gray'), plt.title('thresholded')
    # plt.subplot(133), plt.imshow(gcm_to, 'gray'), plt.title('opened')
    #
    # plt.figure()
    # plt.subplot(121), plt.imshow(data, 'gray', interpolation='nearest'), plt.title('input')
    # plt.subplot(122), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.title('seeds')
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # plt.colorbar(cax=cax, ticks=np.unique(seeds))

    # plt.figure()
    # plt.subplot(141), plt.imshow(data, 'gray'), plt.title('input')
    # plt.subplot(142), plt.imshow(labs_f, 'gray'), plt.title('labels')
    # plt.subplot(143), plt.imshow(liver_blob, 'gray'), plt.title('liver blob')
    # plt.subplot(144), plt.imshow(init_mask, 'gray'), plt.title('init mask')

    # finding liver blob
    liver_blob = find_liver_blob(data, labs_f, slice=slice, show=show, show_now=show_now)

    # hole filling - adding (and then removing) a capsule of zeros, otherwise it'd fill holes touching image borders
    init_mask = tools.fill_holes_watch_borders(liver_blob)
    # init_mask = np.zeros([x + 2 for x in liver_blob.shape], dtype=np.uint8)
    # if liver_blob.ndim == 3:
    #     for i, im in enumerate(liver_blob):
    #         init_mask[i + 1, 1:-1, 1:-1] = im
    # else:
    #     init_mask[1:-1, 1:-1] = liver_blob
    # init_mask = skimor.remove_small_holes(init_mask, min_size=0.1 * liver_blob.sum(), connectivity=2)
    # if liver_blob.ndim == 3:
    #     init_mask = init_mask[1:-1, 1:-1, 1:-1]
    # else:
    #     init_mask = init_mask[1:-1, 1:-1]

    # visualization
    if show:
        if slice is None:
            slice = 0
        data_vis = data if data.ndim == 2 else data[slice,...]
        seeds_vis = seeds if data.ndim == 2 else seeds[slice, ...]
        labs_f_vis = labs_f if data.ndim == 2 else labs_f[slice,...]
        liver_blob_vis = liver_blob if data.ndim == 2 else liver_blob[slice,...]
        init_mask_vis = init_mask if data.ndim == 2 else init_mask[slice,...]

        plt.figure()
        plt.subplot(131), plt.imshow(gcm, 'gray', vmax=gcm.mean()), plt.title('gcm')
        plt.subplot(132), plt.imshow(gcm_t, 'gray'), plt.title('thresholded')
        plt.subplot(133), plt.imshow(gcm_to, 'gray'), plt.title('opened')

        plt.figure()
        plt.subplot(121), plt.imshow(data_vis, 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(122), plt.imshow(seeds_vis, 'jet', interpolation='nearest'), plt.title('seeds')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax, ticks=np.unique(seeds))

        plt.figure()
        plt.subplot(141), plt.imshow(data_vis, 'gray'), plt.title('input')
        plt.subplot(142), plt.imshow(labs_f_vis, 'gray'), plt.title('labels')
        plt.subplot(143), plt.imshow(liver_blob_vis, 'gray'), plt.title('liver blob')
        plt.subplot(144), plt.imshow(init_mask_vis, 'gray'), plt.title('init mask')

        if show_now:
            plt.show()

    return data, init_mask


def find_liver_blob(data, labs_im, dens_min=120, dens_max=200, slice=None, show=False, show_now=True):
    lbls = [l for l in np.unique(labs_im) if l != 0]
    adepts = np.zeros_like(labs_im)
    for l in lbls:
        mask = labs_im == l
        mean_int = data[np.nonzero(mask)].mean()
        if not dens_min < mean_int < dens_max:
            print 'mean value: {:.1f} - SKIPPING'.format(mean_int)
            continue
        else:
            print 'mean value: {:.1f} - OK'.format(mean_int)
            adepts += (l * mask).astype(np.uint8)
        # plt.figure()
        # plt.subplot(131), plt.imshow(labs_im, 'jet'), plt.title('labs_im')
        # plt.subplot(132), plt.imshow(mask, 'gray'), plt.title('mask')
        # plt.subplot(133), plt.imshow(adepts, 'jet', vmin=labs_im.min(), vmax=labs_im.max()), plt.title('adepts')
        # plt.show()

    # adepts = skimor.binary_opening(adepts > 0, selem=skimor.disk(3))
    adepts = tools.morph_ND(adepts > 0, 'erosion', selem=skimor.disk(3))

    adepts_lbl, n_labels = skimea.label(adepts, connectivity=2, return_num=True)
    areas = [(adepts_lbl == l).sum() for l in range(1, n_labels + 1)]
    winner = adepts_lbl == (np.argmax(areas) + 1)

    if show:
        if data.ndim == 3:
            if slice is None:
                slice = 0
            labs_vis = labs_im[slice,...]
            adepts_vis = adepts_lbl[slice,...]
            winner_vis = winner[slice,...]
        else:
            labs_vis = labs_im
            adepts_vis = adepts_lbl
            winner_vis = winner
        plt.figure()
        plt.subplot(131), plt.imshow(labs_vis, 'jet'), plt.title('labs_im')
        plt.subplot(132), plt.imshow(adepts_vis, 'jet', vmax=labs_im.max()), plt.title('adepts')
        plt.subplot(133), plt.imshow(winner_vis, 'jet', vmax=labs_im.max()), plt.title('liver blob')
        if show_now:
            plt.show()

    return winner


def analyze_glcm(gcm, area_t=200, ecc_t=0.35):
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


def lankton_ls(im, mask, method='sfm', slice=None, max_iters=1000, rad=10, alpha=0.1, scale=1., show=False, show_now=True):
    if scale != 1:
        # im = skitra.rescale(im, scale=scale, preserve_range=True).astype(np.uint8)
        # mask = skitra.rescale(mask, scale=scale, preserve_range=True).astype(np.bool)
        im = tools.resize_ND(im, scale=scale).astype(np.uint8)
        mask = tools.resize_ND(mask, scale=scale).astype(np.bool)

    seg = lankton_lls.run(im, mask, method='sfm', max_iter=1000, rad=rad, alpha=alpha,
                          # slice=slice, show=show, show_now=show_now)
                          slice=slice, show=True, show_now=True)

    if show:
        tools.visualize_seg(data, mask, seg, slice=slice, title='morph snakes', show_now=show_now)
    return im, mask, seg


def morph_snakes(data, mask, slice=None, scale=0.5, alpha=1000, sigma=1, smoothing_ls=1, threshold=0.3, balloon=1, max_iters=50, show=False, show_now=True):
    if scale != 1:
        # data = skitra.rescale(data, scale=scale, preserve_range=True).astype(np.uint8)
        # mask = skitra.rescale(mask, scale=scale, preserve_range=True).astype(np.bool)
        data = tools.resize3D(data, scale, sliceId=0)
        mask = tools.resize3D(mask, scale, sliceId=0)

    gI = morphsnakes.gborders(data, alpha=alpha, sigma=sigma)
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=smoothing_ls, threshold=threshold, balloon=balloon)
    mgac.levelset = mask
    mgac.run(iterations=max_iters)
    seg = mgac.levelset

    if show:
        tools.visualize_seg(data, seg, mask, slice=slice, title='morph snakes', show_now=show_now)
    return data, mask, seg


def calc_statistics(mask, gt):
    if mask.shape != gt.shape:
        mask = tools.resize_ND(mask, shape=gt.shape)
    precision = 100 * (mask * gt).sum() / mask.sum()  # how many selected items are relevant
    recall = 100 * (mask * gt).sum() / gt.sum()  # how many relevant items are selected
    f_measure = 2 * precision * recall / (precision + recall)

    return precision, recall, f_measure


def load_count_write(dir, gt_mask, xlsx_fname):
    files = glob.glob(os.path.join(dir, '*.pklz'))
    items = []
    print 'Calculating statistics ...',
    for f in files:
        f = gzip.open(f)
        datap = pickle.load(f)
        seg = datap['seg']
        params = datap['params']

        precision, recall, f_measure = calc_statistics(seg, gt_mask)
        elements = (f_measure, precision, recall, params['alpha'], params['sigma'], params['threshold'],
                 params['balloon'], params['scale'], params['init_scale'])
        items.append(elements)
        f.close()
    print 'done'

    print 'Writing the results to file ...',
    write_to_xlsx(xlsx_fname, items)
    print 'done'


def write_to_xlsx(fname, items):
    # fname = '/home/tomas/Dropbox/Data/liver_segmentation/morph_snakes/morph_snakes.xlsx'
    workbook = xlsxwriter.Workbook(fname)
    worksheet = workbook.add_worksheet()
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})
    # green = workbook.add_format({'bg_color': '#C6EFCE'})
    worksheet.write(0, 0, 'F-MEASURE', bold)
    worksheet.write(0, 1, 'PRECISION', bold)
    worksheet.write(0, 2, 'RECALL', bold)
    worksheet.write(0, 4, 'alpha', bold)
    worksheet.write(0, 5, 'sigma', bold)
    worksheet.write(0, 6, 'threshold', bold)
    worksheet.write(0, 7, 'ballon', bold)
    worksheet.write(0, 8, 'scale', bold)
    worksheet.write(0, 9, 'scale_init', bold)
    cols = [0, 1, 2, 4, 5, 6, 7, 8, 9]  # to be able to simply skip a column

    # saving results
    for i, item in enumerate(items):
        # item = (f_measure, precision, recall, alpha, sigma, threshold, balloon, params['scale'], params['init_scale'])
        for j, it in enumerate(item):
            worksheet.write(i + 1, cols[j], it)


def run(im_in, slice=None, mask=None, smoothing=True, method='sfm', max_iters=1000,
        rad=10, alpha=0.2, scale=1., init_scale=0.5,
        sigma=1, smoothing_ls=1, threshold=0.3, balloon=1,
        show=False, show_now=True, save_fig=False, verbose=True):
    im = im_in.copy()
    if smoothing:
        im = tools.smoothing(im, sigmaSpace=10, sigmaColor=10, sliceId=0)
    # init_mask = initialize(im, dens_min=50, dens_max=200, show=True, show_now=False)[0,...]
    if mask is None:
        # mask = initialize_graycom(im, [1, 2], scale=1, show=show, show_now=show_now)
        mask = tools.initialize_graycom(im, slice, distances=[1,], scale=init_scale, show=show, show_now=show_now)
    # else:
    #     im_init = im

    if method == 'morphsnakes':
        print ' Morph snakes ...',
        im, mask, seg = morph_snakes(im, mask, scale=scale, alpha=int(alpha), sigma=sigma, smoothing_ls=smoothing_ls,
                                     threshold=threshold, balloon=balloon, max_iters=max_iters,
                                     slice=slice, show=show, show_now=show_now)
        print 'done'
    elif method in ['sfm', 'lls']:
        print 'SFM ...',
        im, mask, seg = lankton_ls(im, mask, method=method, max_iters=max_iters, rad=rad, alpha=alpha, scale=scale,
                                   slice=slice, show=show, show_now=show_now)
        print 'done'
    return im, mask, seg


def run_param_tuning_sfm(im_in, gt_mask, slice_ind, smoothing=True, scale=0.5, init_scale=0.5):
    if smoothing:
        im_in = tools.smoothing(im_in, sigmaSpace=10, sigmaColor=10, sliceId=0)

    im = im_in.copy()
    mask_init = tools.initialize_graycom(im, slice, distances=[1, ], scale=init_scale)

    # parameters tuning ---------------------------------------------------------------
    alpha_v = (0.1, 0.3, 0.5, 0.8)
    rad_v = (1, 10, 20, 30)

    method = 'sfm'
    # create workbook for saving the resulting pracision, recall and f-measure
    workbook = xlsxwriter.Workbook('/home/tomas/Dropbox/Data/liver_segmentation/%s/%s.xlsx' % (method, method))
    worksheet = workbook.add_worksheet()
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})
    # green = workbook.add_format({'bg_color': '#C6EFCE'})
    worksheet.write(0, 0, 'F-MEASURE',bold)
    worksheet.write(0, 1, 'PRECISION', bold)
    worksheet.write(0, 2, 'RECALL', bold)
    worksheet.write(0, 4, 'alpha', bold)
    worksheet.write(0, 5, 'rad', bold)
    worksheet.write(0, 6, 'scale', bold)
    worksheet.write(0, 7, 'scale_init', bold)

    params_tune = []
    for p in itertools.product(alpha_v, rad_v):
        params_tune.append(p)
    # for i, (alpha, sigma, threshold, balloon) in enumerate(itertools.product(alpha_v, sigma_v, threshold_v, balloon_v)):
    for i in range(len(params_tune)):
        alpha, rad = params_tune[i]
        # threshold = 0.9
        # balloon = 4
        print '\n  --  it #%i/%i  --' % (i + 1, len(params_tune))
        print 'Working with alpha=%.1f, rad=%i' % (alpha, rad)
        params = {'alpha': alpha, 'rad': rad, 'max_iters': 1000, 'scale': scale, 'init_scale': init_scale,
                  'smoothing': False, 'save_fig': False, 'show': True, 'show_now': False}
        im, mask, seg = run(im_in, slice=slice_ind, mask=mask_init, method=method, **params)

        # print 'shapes 1: im={}, mask={}, seg={}'.format(im.shape, mask.shape, seg.shape)
        if im.shape != im_in.shape:
            im = tools.resize_ND(im, shape=im_in.shape)
        if mask.shape != gt_mask.shape:
            mask = tools.resize_ND(mask, shape=gt_mask.shape)
        if seg.shape != gt_mask.shape:
            seg = tools.resize_ND(seg, shape=gt_mask.shape)
        # print 'shapes 2: im={}, mask={}, seg={}'.format(im.shape, mask.shape, seg.shape)

        # calculating precision, recall and f_measure
        precision, recall, f_measure = calc_statistics(seg, gt_mask)
        # print 'shapes 3: im={}, mask={}, seg={}'.format(im.shape, mask.shape, seg.shape)

        # saving data
        datap = {'im': im, 'mask': mask, 'seg': seg, 'params': params}
        dirname = '/home/tomas/Dropbox/Data/liver_segmentation/%s/' % method
        dataname = '%s_al%s_rad%i.pklz' % (method, str(alpha).replace('.', ''), rad)
        f = gzip.open(os.path.join(dirname, dataname), 'w')
        pickle.dump(datap, f)
        f.close()

        # saving results
        cols = [0, 1, 2, 4, 5, 6, 7]
        items = (f_measure, precision, recall, alpha, rad, params['scale'], params['init_scale'])
        for j, it in enumerate(items):
            worksheet.write(i + 1, cols[j], it)

        # creating visualization
        fig = tools.visualize_seg(im, seg, mask, slice=slice_ind, title=method, for_save=True, show_now=False)
        fname = '/home/tomas/Dropbox/Data/liver_segmentation/%s/%s_al%s_rad%i.png' \
                % (method, method, str(alpha).replace('.', ''), rad)
        fig.savefig(fname)
        plt.close('all')

    workbook.close()
    # plt.show()


def run_param_tuning_morphsnakes(im_in, gt_mask, slice_ind, smoothing=True, scale=0.5, init_scale=0.5):
    if smoothing:
        im_in = tools.smoothing(im_in, sigmaSpace=10, sigmaColor=10, sliceId=0)

    im = im_in.copy()
    mask_init = tools.initialize_graycom(im, slice, distances=[1, ], scale=init_scale)

    # parameters tuning ---------------------------------------------------------------
    alpha_v = (10, 100, 500)
    sigma_v = (1, 5)
    threshold_v = (0.1, 0.5, 0.9)
    balloon_v = (1, 2, 4)

    method = 'morphsnakes'
    # create workbook for saving the resulting pracision, recall and f-measure
    workbook = xlsxwriter.Workbook('/home/tomas/Dropbox/Data/liver_segmentation/%s/%s.xlsx' % (method, method))
    worksheet = workbook.add_worksheet()
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})
    # green = workbook.add_format({'bg_color': '#C6EFCE'})
    worksheet.write(0, 0, 'F-MEASURE',bold)
    worksheet.write(0, 1, 'PRECISION', bold)
    worksheet.write(0, 2, 'RECALL', bold)
    worksheet.write(0, 4, 'alpha', bold)
    worksheet.write(0, 5, 'sigma', bold)
    worksheet.write(0, 6, 'threshold', bold)
    worksheet.write(0, 7, 'ballon', bold)
    worksheet.write(0, 8, 'scale', bold)
    worksheet.write(0, 9, 'scale_init', bold)

    params_tune = []
    for p in itertools.product(alpha_v, sigma_v, threshold_v, balloon_v):
        params_tune.append(p)
    # for i, (alpha, sigma, threshold, balloon) in enumerate(itertools.product(alpha_v, sigma_v, threshold_v, balloon_v)):
    for i in range(len(params_tune)):
        alpha, sigma, threshold, balloon = params_tune[i]
        # threshold = 0.9
        # balloon = 4
        print '\n  --  it #%i/%i  --' % (i + 1, len(params_tune))
        print 'Working with alpha=%i, sigma=%i, threshold=%.1f, balloon=%i' % (alpha, sigma, threshold, balloon)
        params = {'alpha': alpha, 'sigma': sigma, 'smoothing_ls': 1, 'threshold': threshold, 'balloon': balloon,
                  'max_iters': 1000, 'scale': scale, 'init_scale': init_scale,
                  'smoothing': False, 'save_fig': False, 'show': True, 'show_now': False}
        im, mask, seg = run(im_in, slice=slice_ind, mask=mask_init, method=method, **params)

        # print 'shapes 1: im={}, mask={}, seg={}'.format(im.shape, mask.shape, seg.shape)
        if im.shape != im_in.shape:
            im = tools.resize_ND(im, shape=im_in.shape)
        if mask.shape != gt_mask.shape:
            mask = tools.resize_ND(mask, shape=gt_mask.shape)
        if seg.shape != gt_mask.shape:
            seg = tools.resize_ND(seg, shape=gt_mask.shape)
        # print 'shapes 2: im={}, mask={}, seg={}'.format(im.shape, mask.shape, seg.shape)

        # calculating precision, recall and f_measure
        precision, recall, f_measure = calc_statistics(seg, gt_mask)
        # print 'shapes 3: im={}, mask={}, seg={}'.format(im.shape, mask.shape, seg.shape)

        # saving data
        datap = {'im': im, 'mask': mask, 'seg': seg, 'params': params}
        dirname = '/home/tomas/Dropbox/Data/liver_segmentation/%s/' % method
        dataname = '%s_al%i_si%i_th%s_ba%i.pklz' % (method, alpha, sigma, str(threshold).replace('.', ''), balloon)
        f = gzip.open(os.path.join(dirname, dataname), 'w')
        pickle.dump(datap, f)
        f.close()

        # saving results
        cols = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        items = (f_measure, precision, recall, alpha, sigma, threshold, balloon, params['scale'], params['init_scale'])
        for j, it in enumerate(items):
            worksheet.write(i + 1, cols[j], it)

        # creating visualization
        fig = tools.visualize_seg(im, seg, mask, slice=slice_ind, title=method, for_save=True, show_now=False)
        fname = '/home/tomas/Dropbox/Data/liver_segmentation/%s/%s_al%i_si%i_th%s_ba%i.png' \
                % (method, method, alpha, sigma, str(threshold).replace('.', ''), balloon)
        fig.savefig(fname)
        plt.close('all')

    workbook.close()
    # plt.show()

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    verbose = True
    show = True
    show_now = False
    save_fig = False
    smoothing = True
    init_scale = 0.25
    scale = 0.5

    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    # 2D
    # slice_ind = 17
    slice_ind = 15
    # data = data[slice_ind, :, :]
    data = tools.windowing(data)

    # 3D
    # data = tools.windowing(data)

    # SFM  ----
    # print 'SFM ...',
    method = 'sfm'
    # params = {'rad': 10, 'alpha': 0.6, 'scale': 0.5, 'init_scale': 0.5, 'smoothing': smoothing,
    #           'save_fig': save_fig, 'show':show, 'show_now': show_now, 'verbose': verbose}
    # im, mask, seg = run(data, slice=None, mask=None, method=method, **params)
    # print 'done'

    # MORPH SNAKES  ----
    # method = 'morphsnakes'
    # params = {'alpha': 100, 'sigma': 1, 'smoothing_ls': 1, 'threshold': 0.6, 'balloon': 1, 'max_iters': 1000,
    #           'scale': 0.5, 'init_scale': 0.25,
    #           'smoothing': smoothing, 'save_fig': save_fig, 'show': show, 'show_now': show_now}
    # params = {'alpha': 10, 'sigma': 1, 'smoothing_ls': 1, 'threshold': 0.9, 'balloon': 4, 'max_iters': 1000,
    #           'scale': scale, 'init_scale': init_scale,
    #           'smoothing': smoothing, 'save_fig': save_fig, 'show': show, 'show_now': show_now}
    # im, mask, seg = run(data, slice=slice_ind, mask=None, method=method, **params)

    # PARAM TUNING  ----
    if method == 'morphsnakes':
        run_param_tuning_morphsnakes(data, mask, slice_ind, smoothing=smoothing, scale=scale, init_scale=init_scale)
    else:
        run_param_tuning_sfm(data, mask, slice_ind, smoothing=smoothing, scale=scale, init_scale=init_scale)

    # CALCULATE STATISTICS FROM FILES  ----
    # xlsx_fname = '/home/tomas/Dropbox/Data/liver_segmentation/%s/%s.xlsx' % (method, method)
    # load_count_write('/home/tomas/Dropbox/Data/liver_segmentation/%s' % method, mask, xlsx_fname)

    # if show_now == False:
    #     plt.show()