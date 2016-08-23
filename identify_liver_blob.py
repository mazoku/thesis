from __future__ import division

import os
import sys

import numpy as np
import scipy.ndimage.measurements as scindimea
import io3d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

import skimage.morphology as skimor
import skimage.measure as skimea
import skimage.exposure as skiexp
import skimage.segmentation as skiseg

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in active_contours.py. You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


def calc_features(datadir):
    areas = []
    ints_art = []
    ints_ven = []
    eccs = []
    fnames = load_data(datadir)

    for i, fname in enumerate(fnames):
        data, mask, voxel_size = tools.load_pickle_data(fname, return_datap=False)
        mask = np.where(mask > 0, 1, 0)
        area = mask.sum()
        # mask_tmp = skimor.binary_erosion(mask, selem=skimor.disk(1))
        mask_tmp = tools.morph_ND(mask, 'erosion', selem=skimor.disk(1))
        mean_int = data[np.nonzero(mask_tmp)].mean()
        ecc = []
        for m in mask:
            if m.any():
                rp = skimea.regionprops(m)
                ecc.append(rp[0].eccentricity)
        ecc = np.array(ecc).mean()

        print 'area: %i, int: %i, ecc: %.2f' % (area, round(mean_int), ecc)
        areas.append(area)
        if 'arterial' in fname:
            ints_art.append(mean_int)
        elif 'venous' in fname:
            ints_ven.append(mean_int)
        eccs.append(ecc)

    mean_area = np.array(areas).mean()
    mean_int_art = np.array(ints_art).mean()
    mean_int_ven = np.array(ints_ven).mean()
    mean_ecc = np.array(eccs).mean()
    print '----'
    print 'mean area: %i, mean int art: %.1f, mean int ven: %.1f, mean ecc: %.2f' % (mean_area, mean_int_art, mean_int_ven, mean_ecc)


def load_data(dir, ext='pklz'):
    data = []
    # najdu vsechny pklz soubory
    for (rootDir, dirNames, filenames) in os.walk(dir):
        for filename in filenames:
            if filename.split('.')[-1] == ext:  # '.pklz':
                data.append(os.path.join(dir, filename))
    return data


def blob_from_hist(data, show=False, show_now=True, min_int=0, max_int=255):
    data = tools.windowing(data, sliceId=0)
    data = tools.resize_ND(data, scale=0.1).astype(np.uint8)
    data = tools.smoothing(data, sliceId=0)

    min_interval = int((0 + 125) / 350. * 255)
    max_interval = int((100 + 125) / 350. * 255)

    mask = np.ones_like(data)
    mask = np.where(data < 5, 0, mask)
    mask = np.where(data > 250, 0, mask)
    pts = data[np.nonzero(mask)]
    hist, bins = skiexp.histogram(pts)
    hist = tools.hist_smoothing(bins, hist)
    # bins = bins / 255 * 350 - 125

    hist_rec = np.zeros_like(hist)
    hist_rec[min_interval:max_interval] = hist[min_interval:max_interval]
    peak_idx = np.argmax(hist_rec)
    hist_ind = bins[peak_idx]

    hist_hu = hist_ind / 255 * 350 - 125
    print 'hist ind: %.1f = %i HU' % (hist_ind, int(hist_hu))

    if show:
        plt.figure()
        plt.plot(bins, hist, 'b-', lw=4)
        plt.plot((min_interval, min_interval), (0, 1.2 * hist_rec.max()), 'm-', lw=4)
        plt.plot((max_interval, max_interval), (0, 1.2 * hist_rec.max()), 'm-', lw=4)
        plt.plot(hist_ind, hist[peak_idx], 'go', markersize=12)

        img = data[21, ...]
        hist_diff = abs(img.astype(np.int64) - hist_ind)
        plt.figure()
        plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(122), plt.imshow(hist_diff, 'gray', interpolation='nearest'), plt.axis('off')

        if show_now:
            plt.show()

    return hist_ind


def blob_from_glcm(data, show=False, show_now=True, min_int=0, max_int=255):
    data = tools.windowing(data, sliceId=0)
    data_f = tools.resize_ND(data, scale=0.1).astype(np.uint8)
    data_f = tools.smoothing(data_f, sliceId=0)

    min_interval = int((0 + 125) / 350. * 255)
    max_interval = int((100 + 125) / 350. * 255)
    # min_int = 20
    # max_int = 80

    print 'calculating glcm ...',
    glcm = tools.graycomatrix_3D(data_f)
    print 'done'

    glcm_rec = glcm.copy()
    mask = np.zeros_like(glcm_rec)
    mask[min_interval:max_interval, min_interval:max_interval] = 1
    glcm_rec *= mask
    max_r, max_c = np.unravel_index(np.argmax(glcm_rec), glcm_rec.shape)
    glcm_ind = (max_r + max_c) / 2

    glcm_hu = glcm_ind / 255 * 350 - 125
    print 'glcm ind: %.1f = %i HU' % (glcm_ind, int(glcm_hu))

    img = data[21,...]

    if show:
        plt.figure()
        plt.subplot(121), plt.imshow(glcm, interpolation='nearest', vmax=5 * glcm.mean())
        plt.subplot(122), plt.imshow(glcm, interpolation='nearest', vmax=5 * glcm.mean())
        plt.plot((min_interval, min_interval), (min_interval, max_interval), 'm-', lw=5)
        plt.plot((min_interval, max_interval), (max_interval, max_interval), 'm-', lw=5)
        plt.plot((max_interval, max_interval), (max_interval, min_interval), 'm-', lw=5)
        plt.plot((max_interval, min_interval), (min_interval, min_interval), 'm-', lw=5)
        plt.plot(max_c, max_r, 'wo', markersize=12)
        plt.axis('image')

        glcm_diff = abs(img.astype(np.int64) - glcm_ind)
        plt.figure()
        plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(122), plt.imshow(glcm_diff, 'gray', interpolation='nearest'), plt.axis('off')

        if show_now:
            plt.show()

    return glcm_ind


def score_data(data, seg, peak_int=130, show=False, show_now=True, verbose=False):
    seg = skiseg.clear_border(seg)
    label_im, num_features = skimea.label(seg, connectivity=2, return_num=True)

    areas = []
    ints = []
    eccs = []
    dists = []
    lbls = []
    im_cr, im_cc = np.array(data.shape) / 2
    des_r = im_cr
    des_c = im_cc / 2
    for i in range(1, num_features):
        lbl = label_im == i
        area = lbl.sum()
        if area > 100:
            mean_int = data[np.nonzero(lbl)].mean()
            rp = skimea.regionprops(lbl.astype(np.uint8))[0]
            ecc = rp.eccentricity
            lab_cr, lab_cc = rp.centroid
            dist = np.sqrt((des_r - lab_cr) ** 2 + (des_c - lab_cc) ** 2)

            areas.append(area)
            ints.append(abs(peak_int - mean_int))
            eccs.append(ecc)
            dists.append(dist)
            lbls.append(i)
            if verbose:
                print 'area = %i, int = %i, ecc = %.2f, dist=%.1f' % (area, int(mean_int), ecc, dist)
    ints = max(ints) - np.array(ints)
    eccs = 1 - np.array(eccs)
    dists = max(dists) - np.array(dists)

    # rescaling
    areas = skiexp.rescale_intensity(np.array(areas).astype(np.float), out_range=(0, 1))
    ints = skiexp.rescale_intensity(np.array(ints).astype(np.float), out_range=(0, 1))
    eccs = skiexp.rescale_intensity(np.array(eccs).astype(np.float), out_range=(0, 1))
    dists = skiexp.rescale_intensity(np.array(dists).astype(np.float), out_range=(0, 1))

    score = areas + ints + eccs + dists
    # score = areas + ints + dists
    score_im = np.zeros(data.shape)

    area_sc_im = np.zeros(data.shape)
    ints_sc_im = np.zeros(data.shape)
    eccs_sc_im = np.zeros(data.shape)
    pos_sc_im = np.zeros(data.shape)
    for i, l in enumerate(lbls):
        lbl = label_im == l
        area_sc_im = np.where(lbl, areas[i], area_sc_im)
        ints_sc_im = np.where(lbl, ints[i], ints_sc_im)
        eccs_sc_im = np.where(lbl, eccs[i], eccs_sc_im)
        pos_sc_im = np.where(lbl, dists[i], pos_sc_im)
        score_im = np.where(lbl, score[i], score_im)

    max_sc = score_im.max()
    blob = score_im == max_sc

    if show:
        plt.figure()
        plt.subplot(231), plt.imshow(area_sc_im, 'jet', interpolation='nearest'), plt.axis('off')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)
        plt.subplot(232), plt.imshow(ints_sc_im, 'jet', interpolation='nearest'), plt.axis('off')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)
        plt.subplot(233), plt.imshow(eccs_sc_im, 'jet', interpolation='nearest'), plt.axis('off')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)
        plt.subplot(234), plt.imshow(pos_sc_im, 'jet', interpolation='nearest'), plt.axis('off')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)
        plt.subplot(235), plt.imshow(score_im, 'jet', interpolation='nearest'), plt.axis('off')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax)

        plt.figure()
        plt.subplot(131), plt.imshow(data, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(132), plt.imshow(seg, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(133), plt.imshow(blob, 'gray', interpolation='nearest'), plt.axis('off')

        if show_now:
            plt.show()

    return blob

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # gt_dir = '/home/tomas/Data/medical/dataset/gt'
    datadir = '/media/tomas/96C68A9BC68A7AEF/Data/20140124-data-other/183/46324212/6'

    dcr = io3d.DicomReader(datadir, gui=False)
    data = dcr.get_3Ddata()
    # data = tools.resize_ND(data, scale=0.1)

    # blob_from_glcm(data, show=True)
    blob_from_hist(data, show=True)

    # calc_features(gt_dir)


