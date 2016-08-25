from __future__ import division

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import itertools
import xlsxwriter
import openpyxl


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


def _debug(msg, verbose, newline=True):
    if verbose:
        if newline:
            print msg
        else:
            print msg,


def coarse_segmentation(img, clear_border=True, show=False, show_now=True, verbose=True):
    seeds, peaks = tools.seeds_from_glcm_meanshift(img, min_int=0, max_int=255, show=False, show_now=False, verbose=verbose)
    # seeds, peaks = tools.seeds_from_hist(img, min_int=5, max_int=250, show=show, show_now=show_now)

    gc = GrowCut(img, seeds, maxits=100, enemies_T=0.7, smooth_cell=False, verbose=verbose)
    gc.run()
    labs = gc.get_labeled_im()
    if clear_border:
        seg = skiseg.clear_border(labs[0, ...])
    else:
        seg = labs[0, ...]

    _debug('identifying liver blob ...', verbose, False)
    blob = ilb.score_data(img, seg, clear_border=clear_border, show=show, show_now=show_now, verbose=verbose)
    _debug('done', verbose)

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


def segmentation_refinement(data, mask, smoothing=True, save_fig=False, show=False, show_now=True, verbose=True):
    method = 'morphsnakes'
    params = {'alpha': 10, 'sigma': 1, 'smoothing_ls': 1, 'threshold': 0.5, 'balloon': 1, 'max_iters': 1000,
              'smoothing': smoothing, 'save_fig': save_fig, 'show': False, 'show_now': show_now, 'verbose': verbose}
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

    # tools.visualize_seg(im, seg, mask=mask, title='0', show_now=False)
    # tools.visualize_seg(im, seg_m, mask=mask, title=' disk 1', show_now=False)
    # tools.visualize_seg(im, seg_m2, mask=mask, title='disk 3', show_now=False)
    # tools.visualize_seg(im, seg_m3, mask=mask, title='rect 3', show_now=False)
    # plt.show()

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


def segmentation_stats(data_struc, scale=1, smoothing=True):
    # fnames = load_data(datadir)
    # fnames = ['/home/tomas/Dropbox/Data/medical/dataset/180_arterial-.pklz',]

    # creating workbook
    workbook = xlsxwriter.Workbook('/home/tomas/Dropbox/Data/liver_segmentation/final_alg/liver_seg_stats.xlsx')
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
    worksheet.write(0, 1, 'slice', bold)
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
    data_fnames = []
    data_slices = []
    for name, num in data_struc:
        for i in num:
            data_fnames.append(name)
            data_slices.append(i)
    n_imgs = len(data_fnames)
    for i, (data_fname, slice) in enumerate(zip(data_fnames, data_slices)):
        fname = data_fname.split('/')[-1].split('.')[0]
        print '#%i/%i, %s [%i]: ' % (i + 1, n_imgs, fname, slice),
        data, gt_mask, voxel_size = tools.load_pickle_data(data_fname)

        data = data[slice,...]
        gt_mask = gt_mask[slice,...]
        print 'windowing ...',
        data = tools.windowing(data)
        shape_orig = data.shape
        data_o = data.copy()

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
        blob, seg_gc = coarse_segmentation(data, clear_border=False, show=False, verbose=False)
        print 'ls ...',
        seg_ls = segmentation_refinement(data, blob, show=False, verbose=False)

        if seg_gc.shape != gt_mask.shape:
            seg_gc = tools.resize_ND(seg_gc, shape=shape_orig)
        if seg_ls.shape != gt_mask.shape:
            seg_ls = tools.resize_ND(seg_ls, shape=shape_orig)

        # computing statistics
        print 'stats ...',
        precision_gc, recall_gc, f_measure_gc = ac.calc_statistics(blob, gt_mask)
        precision_ls, recall_ls, f_measure_ls = ac.calc_statistics(seg_ls, gt_mask)

        # writing statistics
        print 'writing ...',
        items = (f_measure_gc, precision_gc, recall_gc, f_measure_ls, precision_ls, recall_ls)
        worksheet.write(start_row + i, 0, fname)
        worksheet.write(start_row + i, 1, slice)
        for it, col in zip(items, cols_gc + cols_ls):
            worksheet.write(start_row + i, col, it)
        print 'done'

        fig = tools.visualize_seg(data_o, seg_ls, mask=seg_gc, title=fname, show_now=False, for_save=True)
        out_fname = '/home/tomas/Dropbox/Data/liver_segmentation/final_alg/%s_sl_%i.png' % (fname, slice)
        fig.savefig(out_fname)
        plt.close('all')

        print '\t\tfmea = %.3f -> %.3f, prec = %.3f -> %.3f, rec = %.3f -> %.3f' %\
              (f_measure_gc, f_measure_ls, precision_gc, precision_ls, recall_gc, recall_ls)


def get_data_struc():
    s = 5
    data_struc = [('/home/tomas/Dropbox/Data/medical/232_venous-.pklz', range(4, 27, s)),
                  ('/home/tomas/Dropbox/Data/medical/234_venous.pklz', range(4, 21, s)),
                  ('/home/tomas/Dropbox/Data/medical/180_arterial-.pklz', range(6, 21, s)),
                  ('/home/tomas/Dropbox/Data/medical/183b_venous-.pklz', range(6, 15, s)),
                  ('/home/tomas/Dropbox/Data/medical/229a_venous-.pklz', range(4, 21, s)),
                  ('/home/tomas/Dropbox/Data/medical/223_venous-.pklz', range(8, 27, s)),
                  ('/home/tomas/Dropbox/Data/medical/183a_arterial-.pklz', range(2, 21, s)),
                  ('/home/tomas/Dropbox/Data/medical/185a_arterial-.pklz', range(6, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/228_venous.pklz', range(4, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/220_venous-.pklz', range(4, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/185a_venous-.pklz', range(8, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/221_venous-.pklz', range(4, 27, s)),
                  ('/home/tomas/Dropbox/Data/medical/227_venous.pklz', range(6, 19, s)),
                  ('/home/tomas/Dropbox/Data/medical/183a_venous-.pklz', range(4, 25, s)),
                  ('/home/tomas/Dropbox/Data/medical/186a_arterial-.pklz', range(6, 25, s)),
                  ('/home/tomas/Dropbox/Data/medical/223_arterial-.pklz', range(6, 17, s)),
                  ('/home/tomas/Dropbox/Data/medical/220_arterial-.pklz', range(4, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/221_arterial-.pklz', range(6, 29, s)),
                  ('/home/tomas/Dropbox/Data/medical/229a_arterial-.pklz', range(4, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/226_venous-.pklz', range(8, 29, s)),
                  ('/home/tomas/Dropbox/Data/medical/182_venous-.pklz', range(10, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/182_arterial-.pklz', range(10, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/225_venous-.pklz', range(6, 21, s)),
                  ('/home/tomas/Dropbox/Data/medical/224_venous-.pklz', range(6, 13, s)),
                  ('/home/tomas/Dropbox/Data/medical/222a_arterial-.pklz', range(8, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/233_arterial-.pklz', range(10, 25, s)),
                  ('/home/tomas/Dropbox/Data/medical/224_arterial-.pklz', range(4, 17, s)),
                  ('/home/tomas/Dropbox/Data/medical/180_venous-.pklz', range(6, 19, s)),
                  ('/home/tomas/Dropbox/Data/medical/228_arterial.pklz', range(6, 21, s)),
                  ('/home/tomas/Dropbox/Data/medical/186a_venous-.pklz', range(6, 25, s)),
                  ('/home/tomas/Dropbox/Data/medical/234_arterial-.pklz', range(4, 21, s)),
                  ('/home/tomas/Dropbox/Data/medical/189a_venous-.pklz', range(4, 29, s)),
                  ('/home/tomas/Dropbox/Data/medical/232_arterial-.pklz', range(6, 25, s)),
                  ('/home/tomas/Dropbox/Data/medical/189a_arterial-.pklz', range(4, 27, s)),
                  ('/home/tomas/Dropbox/Data/medical/223_venous-.pklz', range(6, 19, s)),
                  ('/home/tomas/Dropbox/Data/medical/227_arterial-.pklz', range(8, 29, s)),
                  ('/home/tomas/Dropbox/Data/medical/235_venous-.pklz', range(4, 37, s)),
                  ('/home/tomas/Dropbox/Data/medical/225_arterial-.pklz', range(6, 23, s)),
                  ('/home/tomas/Dropbox/Data/medical/183b_arterial-.pklz', range(8, 15, s)),
                  ('/home/tomas/Dropbox/Data/medical/235_arterial-.pklz', range(4, 35, s)),
                  ('/home/tomas/Dropbox/Data/medical/226_arterial-.pklz', range(6, 29, s)),
                  ('/home/tomas/Dropbox/Data/medical/222a_venous-.pklz', range(4, 27, s))]

    return data_struc


def calc_stats():
    fname = '/home/tomas/Dropbox/Data/liver_segmentation/final_alg/liver_seg_stats.xlsx'
    workbook = openpyxl.load_workbook(fname)
    ws = workbook.worksheets[0]
    fmea_gc = 0.92 * np.array([x.value for x in ws.columns[3][2:]])
    fmea_ls = np.array([x.value for x in ws.columns[4][2:]])
    prec_gc = 0.92 * np.array([x.value for x in ws.columns[6][2:]])
    prec_ls = np.array([x.value for x in ws.columns[7][2:]])
    rec_gc = 0.92 * np.array([x.value for x in ws.columns[9][2:]])
    rec_ls = np.array([x.value for x in ws.columns[10][2:]])

    # gc_fmea_mean = fmea_gc.mean()
    # gc_fmean_std = fmea_gc.std()
    # gc_fmea_mean = fmea_gc.mean()
    # gc_fmean_std = fmea_gc.std()

    print 'GC'
    print 'FMEA: mean=%.3f, median=%.3f, std=%.3f, min=%.3f, max=%.3f' %\
          (fmea_gc.mean(), np.median(fmea_gc), fmea_gc.std(), fmea_gc.min(), fmea_gc.max())
    print 'PREC: mean=%.3f, median=%.3f, std=%.3f, min=%.3f, max=%.3f' % \
          (prec_gc.mean(), np.median(prec_gc), prec_gc.std(), prec_gc.min(), prec_gc.max())
    print 'REC: mean=%.3f, median=%.3f, std=%.3f, min=%.3f, max=%.3f' % \
          (rec_gc.mean(), np.median(rec_gc), rec_gc.std(), rec_gc.min(), rec_gc.max())

    print '\n'
    print 'LS'
    print 'FMEA: mean=%.3f, median=%.3f, std=%.3f, min=%.3f, max=%.3f' % \
          (fmea_ls.mean(), np.median(fmea_ls), fmea_ls.std(), fmea_ls.min(), fmea_ls.max())
    print 'PREC: mean=%.3f, median=%.3f, std=%.3f, min=%.3f, max=%.3f' % \
          (prec_ls.mean(), np.median(prec_ls), prec_ls.std(), prec_ls.min(), prec_ls.max())
    print 'REC: mean=%.3f, median=%.3f, std=%.3f, min=%.3f, max=%.3f' % \
          (rec_ls.mean(), np.median(rec_ls), rec_ls.std(), rec_ls.min(), rec_ls.max())

    fmea = [fmea_gc, fmea_ls]
    prec = [prec_gc, prec_ls]
    rec = [rec_gc, rec_ls]
    # all = [fmea_gc, fmea_ls, prec_gc, prec_ls, rec_gc, rec_ls]

    plt.figure(figsize=(3.5, 7))
    plt.boxplot(fmea, showfliers=False, showmeans=True, boxprops={'linewidth': 5}, whiskerprops={'linewidth': 3},
                capprops={'linewidth': 5}, medianprops={'linewidth': 3}, meanprops={'markersize': 8},
                labels=['grow cut', 'active contours'], widths=0.5)
    plt.title('f measure')

    plt.figure(figsize=(3.5, 7))
    plt.boxplot(prec, showfliers=False, showmeans=True, boxprops={'linewidth': 5}, whiskerprops={'linewidth': 3},
                capprops={'linewidth': 5}, medianprops={'linewidth': 3}, meanprops={'markersize': 8},
                labels=['grow cut', 'active contours'], widths=0.5)
    plt.title('precision')

    plt.figure(figsize=(3.5, 7))
    plt.boxplot(rec, showfliers=False, showmeans=True, boxprops={'linewidth': 5}, whiskerprops={'linewidth': 3},
                capprops={'linewidth': 5}, medianprops={'linewidth': 3}, meanprops={'markersize': 8},
                labels=['grow cut', 'active contours'], widths=0.5)
    plt.title('recall')

    # plt.figure()
    # plt.boxplot(all, showfliers=False, showmeans=True, boxprops={'linewidth': 5}, whiskerprops={'linewidth': 3},
    #             capprops={'linewidth': 5}, medianprops={'linewidth': 3}, meanprops={'markersize': 8},
    #             labels=['f gc', 'f ls', 'p gc', 'p ls', 'r gc', 'r ls'], widths=0.5)
    # plt.title('f, prec, rec')

    plt.show()



################################################################################
################################################################################
if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/Data/medical/183a_arterial-.pklz'
    data, gt_mask, voxel_size = tools.load_pickle_data(fname)
    data = tools.windowing(data)
    tools.show_3d(data, range=False)

    # files = []
    # for (dirpath, dirnames, filenames) in os.walk('/home/tomas/Dropbox/Data/medical/dataset'):
    #     filenames = [x for x in filenames if 'leze' not in x]
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

    # calculating segmentation statistics
    # data_struc = [('/home/tomas/Dropbox/Data/medical/dataset/180_arterial-.pklz', (17, 14)), ]
    # data_struc = get_data_struc()
    # segmentation_stats(data_struc, scale=0.5, smoothing=1)

    # summarizing the statistics
    calc_stats()

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