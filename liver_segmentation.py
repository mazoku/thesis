from __future__ import division

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import itertools
import xlsxwriter

import cv2
import skimage.segmentation as skiseg
import skimage.morphology as skimor
import skimage.measure as skimea

import identify_liver_blob as ilb
import active_contours as ac

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools
else:
    print 'Import error in liver_segmentation.py. You need to import package imtools: https://github.com/mjirik/imtools'

if os.path.exists('/home/tomas/projects/growcut/'):
    sys.path.insert(0, '/home/tomas/projects/growcut/')
    from growcut.growcut import GrowCut
else:
    print 'Import error in liver_segmentation.py. You need to import package growcut: https://github.com/mzaoku/growcut'


def coarse_segmentation(img, show=False, show_now=True):
    seeds, peaks = tools.seeds_from_glcm_meanshift(img, min_int=0, max_int=255, show=True, show_now=False)
    # seeds, peaks = tools.seeds_from_hist(img, min_int=5, max_int=250, show=show, show_now=show_now)

    gc = GrowCut(img, seeds, maxits=100, enemies_T=0.7, smooth_cell=False)
    gc.run()
    labs = gc.get_labeled_im()
    seg = skiseg.clear_border(labs[0, ...])

    print 'identifying liver blob ...',
    blob = ilb.score_data(img, seg)
    print 'done'

    if show:
        plt.figure()
        plt.suptitle('input | seeds | segmentation | blob')
        plt.subplot(141), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(142), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(143), plt.imshow(seg, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(144), plt.imshow(blob, 'gray', interpolation='nearest'), plt.axis('off')

        if show_now:
            plt.show()

    return blob, seg


def segmentation_refinement(data, mask, smoothing=True, save_fig=False, show=False, show_now=True):
    method = 'morphsnakes'
    params = {'alpha': 10, 'sigma': 1, 'smoothing_ls': 1, 'threshold': 0.5, 'balloon': 1, 'max_iters': 1000,
              'smoothing': smoothing, 'save_fig': save_fig, 'show': False, 'show_now': show_now}
    im, mask, seg = ac.run(data, mask=mask, method=method, **params)

    # alpha_v = (10, 100, 500)
    # sigma_v = (1, 5)
    # threshold_v = (0.1, 0.5, 0.9)
    # balloon_v = (1, 2, 4)
    # params_tune = []
    # segs = []
    # for p in itertools.product(alpha_v, sigma_v, threshold_v, balloon_v):
    #     params_tune.append(p)
    # for i in range(len(params_tune)):
    #     alpha, sigma, threshold, balloon = params_tune[i]
    #     # threshold = 0.9
    #     # balloon = 4
    #     print '\n  --  it #%i/%i  --' % (i + 1, len(params_tune))
    #     print 'Working with alpha=%i, sigma=%i, threshold=%.1f, balloon=%i' % (alpha, sigma, threshold, balloon)
    #     params = {'alpha': alpha, 'sigma': sigma, 'smoothing_ls': 1, 'threshold': threshold, 'balloon': balloon,
    #               'max_iters': 1000, 'smoothing': False, 'save_fig': False, 'show': True, 'show_now': False}
    #     im, mask, seg = ac.run(data, mask=mask, method=method, **params)
    #     segs.append(seg)
    #
    #     fig = tools.visualize_seg(im, seg, mask=mask, title='{}: {}'.format(i, p), show_now=False, for_save=True)
    #     fname = '/home/tomas/Dropbox/Data/segmentation_refinement/al%i_si%i_th%s_ba%i.png' \
    #             % (alpha, sigma, str(threshold).replace('.', ''), balloon)
    #     fig.savefig(fname)
    #     plt.close('all')

    # seg_m = skimor.binary_closing(seg, selem=skimor.disk(1))
    # seg_m2 = skimor.binary_closing(seg, selem=skimor.disk(2))
    # seg_m3 = skimor.binary_closing(seg, selem=np.ones((3,3)))
    seg = skimor.binary_closing(seg, selem=skimor.disk(3))
    seg = skimor.binary_opening(seg, selem=skimor.disk(2))

    tools.visualize_seg(im, seg, mask=mask, title='0', show_now=False)
    # tools.visualize_seg(im, seg_m, mask=mask, title=' disk 1', show_now=False)
    # tools.visualize_seg(im, seg_m2, mask=mask, title='disk 3', show_now=False)
    # tools.visualize_seg(im, seg_m3, mask=mask, title='rect 3', show_now=False)
    plt.show()

    if show:
        plt.figure()
        plt.suptitle('input | initialization | segmentation')
        plt.subplot(131), plt.imshow(data, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(132), plt.imshow(mask, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(133), plt.imshow(seg, 'gray', interpolation='nearest'), plt.axis('off')
        if show_now:
            plt.show()

    return seg


def add_peripheral(img, seg, blob):
    seg_in = seg.copy()
    blob_in = blob.copy()

    blob_lbl = seg[np.nonzero(blob)][0]
    holes = skimor.remove_small_holes(blob, min_size=500) - blob
    blob = blob + holes
    seg = np.where(holes, blob_lbl, seg)
    caps = skimor.binary_dilation(blob, np.ones((3, 3))) - blob
    label_im, num_lbls = skimor.label(seg, return_num=True)

    nghb_lbl = np.unique(label_im[np.nonzero(caps)])[1:]
    print nghb_lbl
    relabeled = np.zeros_like(label_im)
    n_relabeled = 0
    for l in nghb_lbl:
        tmp = (label_im == l).astype(np.uint8)
        # rp = skimea.regionprops(tmp)[0]
        area = tmp.sum()
        comp = tools.compactness(tmp)
        if area > 50 and comp < 100:
            n_relabeled += 1
            relabeled = np.where(tmp, n_relabeled, relabeled)

    blob_comp = tools.compactness(blob)
    accepted = np.zeros_like(relabeled)
    for l in range(1, n_relabeled + 1):
        tmp = (relabeled == l).astype(np.uint8)
        tmp += blob.astype(np.uint8)
        comp = tools.compactness(tmp)
        if comp < blob_comp:
            accepted = np.where(relabeled == l, relabeled, accepted)
            print 'ACC: l=%i, comp: %.1f -> %.1f' % (l, blob_comp, comp)
    blob_acc = (blob + accepted) > 0
    seg_acc = np.where(blob_acc, blob_lbl, seg)
        # plt.figure()
        # plt.imshow(tmp, 'gray', interpolation='nearest')
        # plt.title('tmp=%.2f, blob=%.2f' % (comp, blob_comp))
    # plt.show()


    # plt.figure()
    # plt.suptitle('seg | holes | filled seg')
    # plt.subplot(131), plt.imshow(seg_in, interpolation='nearest'), plt.axis('off')
    # plt.subplot(132), plt.imshow(holes, 'gray', interpolation='nearest'), plt.axis('off')
    # plt.subplot(133), plt.imshow(seg, interpolation='nearest'), plt.axis('off')
    # plt.show()
    #
    plt.figure()
    plt.suptitle('seg | relabeled | accepted')
    plt.subplot(141), plt.imshow(seg, interpolation='nearest'), plt.axis('off')
    plt.subplot(142), plt.imshow(relabeled, interpolation='nearest'), plt.axis('off')
    plt.subplot(143), plt.imshow(accepted, interpolation='nearest', vmax=relabeled.max()), plt.axis('off')
    plt.subplot(144), plt.imshow(seg_acc, interpolation='nearest'), plt.axis('off')
    plt.show()

    # tmp = blob.copy()
    # tmp[np.nonzero(caps)] = 2
    # plt.figure()
    # plt.suptitle('blob | capsule | highlighted caps')
    # plt.subplot(131), plt.imshow(blob, 'gray', interpolation='nearest'), plt.axis('off')
    # plt.subplot(132), plt.imshow(caps, 'gray', interpolation='nearest'), plt.axis('off')
    # plt.subplot(133), plt.imshow(tmp, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.show()

    # plt.figure()
    # plt.suptitle('seg | label_im')
    # plt.subplot(121), plt.imshow(seg, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.subplot(122), plt.imshow(label_im, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.show()


def load_data(dir, ext='pklz'):
    data = []
    # najdu vsechny pklz soubory
    for (rootDir, dirNames, filenames) in os.walk(dir):
        for filename in filenames:
            if filename.split('.')[-1] == ext:  # '.pklz':
                data.append(os.path.join(dir, filename))
    return data


def segmentation_stats(datadir, scale=1, smoothing=True):
    # fnames = load_data(datadir)
    fnames = ['/home/tomas/Dropbox/Data/medical/dataset/180_arterial-.pklz',]

    # creating workbook
    workbook = xlsxwriter.Workbook('/home/tomas/Dropbox/Data/medical/dataset/liver_seg_stats.xlsx')
    worksheet = workbook.add_worksheet()
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})
    # green = workbook.add_format({'bg_color': '#C6EFCE'})
    col_gc_fmea = 3
    col_gc_prec = 6
    col_gc_rec = 9
    col_ls_fmea = 4
    col_ls_prec = 7
    col_ls_rec = 10
    cols_gc = (3, 6, 9)
    cols_ls = (4, 7, 10)
    start_row = 2
    worksheet.write(0, 0, 'DATA', bold)
    worksheet.write(0, 3, 'F-MEASURE', bold)
    worksheet.write(0, 6, 'PRECISION', bold)
    worksheet.write(0, 9, 'RECALL', bold)

    worksheet.write(1, col_gc_fmea, 'Grow Cut', bold)
    worksheet.write(1, col_ls_fmea, 'Level Sets', bold)
    worksheet.write(1, col_gc_prec, 'Grow Cut', bold)
    worksheet.write(1, col_ls_prec, 'Level Sets', bold)
    worksheet.write(1, col_gc_rec, 'Grow Cut', bold)
    worksheet.write(1, col_ls_rec, 'Level Sets', bold)

    # loading data
    for i, data_fname in enumerate(fnames):
        print 'data #%i/%i: ' % (i + 1, len(fnames)),
        data, gt_mask, voxel_size = tools.load_pickle_data(data_fname)

        data = data[17,...]
        gt_mask = gt_mask[17,...]
        print 'windowing ...',
        data = tools.windowing(data)
        shape_orig = data.shape

        # plt.figure()
        # plt.imshow(data, 'gray', vmin=0, vmax=255, interpolation='nearest')
        # plt.show()

        if scale != 1:
            print 'resizing ...',
            data = tools.resize_ND(data, scale=scale).astype(np.uint8)

        if smoothing:
            print 'smoothing ...',
            data = tools.smoothing(data)

        print 'gc ...',
        blob, seg_gc = coarse_segmentation(data, show=True)
        print 'ls ...',
        seg_ls = segmentation_refinement(data, blob, show=False)

        if seg_gc.shape != gt_mask.shape:
            seg_gc = tools.resize_ND(seg_gc, shape=shape_orig)
        if seg_ls.shape != gt_mask.shape:
            seg_ls = tools.resize_ND(seg_ls, shape=shape_orig)

        # computing statistics
        print 'stats ...',
        precision_gc, recall_gc, f_measure_gc = ac.calc_statistics(seg_gc, gt_mask)
        precision_ls, recall_ls, f_measure_ls = ac.calc_statistics(seg_ls, gt_mask)

        # writing statistics
        print 'writing ...',
        items = (f_measure_gc, precision_gc, recall_gc, f_measure_ls, precision_ls, recall_ls)
        for it, col in zip(items, cols_gc + cols_ls):
            worksheet.write(start_row + i, col, it)
        print 'done'


################################################################################
################################################################################
if __name__ == '__main__':
    segmentation_stats('', scale=0.5, smoothing=1)


    # data_fname = '/home/tomas/Dropbox/Work/Dizertace/figures/liver_segmentation/hypodenseTumor.png'
    # img = cv2.imread(data_fname, 0)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #
    # # blob, seg = coarse_segmentation(img, show=False, show_now=False)
    # # mask = np.zeros_like(blob)
    # # mask[118:127, 15:23] = 1
    # # mask[118, 18] = 0
    # # mask[126, 18] = 0
    # # seg = np.where(blob * mask, 2, seg)
    # # blob = np.where(blob * mask, 0, blob)
    #
    # blob = np.load('blob.npy')
    # seg = np.load('seg.npy')
    # add_peripheral(img, seg, blob)
    # # np.save('blob.npy', blob)
    # # np.save('seg.npy', seg)
    #
    # plt.figure()
    # plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest')
    # plt.subplot(132), plt.imshow(seg, 'jet', interpolation='nearest')
    # plt.subplot(133), plt.imshow(blob, 'gray', interpolation='nearest')
    # plt.show()

    # blob_init = blob.copy()
    # blob = skimor.remove_small_holes(blob, min_size=2000, connectivity=1)
    # blob = skimor.binary_opening(blob, selem=skimor.disk(1))
    # blob = skimor.binary_erosion(blob, selem=skimor.disk(1))
    #
    # seg = segmentation_refinement(img, blob, show=True, show_now=False)

    # tools.visualize_seg(img, blob_init, title='growcut init', show_now=False)
    # tools.visualize_seg(img, seg, title='morph snakes', show_now=False)

    plt.show()