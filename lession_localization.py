from __future__ import division

import os
import sys

import cPickle as pickle
import gzip
import xlsxwriter
import openpyxl

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
import skimage.exposure as skiexp
import skimage.morphology as skimor
import skimage.segmentation as skiseg
import skimage.color as skicol

import saliency_fish as sf
import mrf_segmentation as mrf_seg

from PyQt4 import QtGui

import datetime

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in %s. You need to import package imtools: https://github.com/mjirik/imtools'\
          % os.path.basename(__file__)
    sys.exit(0)

if os.path.exists('/home/tomas/projects/data_viewers/'):
    sys.path.insert(0, '/home/tomas/projects/data_viewers/')
    from dataviewers.seg_viewer import SegViewer
else:
    print 'Import error in %s. You need to import package imtools: https://github.com/mazoku/dataviewers'\
          % os.path.basename(__file__)
    sys.exit(0)

verbose = False  # whether to write debug comments or not


def set_verbose(val):
    global verbose
    verbose = val


def _debug(msg, msgType="[INFO]", new_line=True):
    if verbose:
        if new_line:
            print '{} {} | {}'.format(msgType, msg, datetime.datetime.now())
        else:
            print '{} {} | {}'.format(msgType, msg, datetime.datetime.now()),


def calc_saliencies(data, mask, smoothing=True, save=False, save_fig=False, types='dhgstcb', out_fname='out.pklz', show=False, show_now=True):
    use_sigmoid = True
    morph_proc = True

    mean_v = int(data[np.nonzero(mask)].mean())
    data = np.where(mask, data, mean_v)

    if smoothing:
        data = tools.smoothing(data)

    data_struc = [(data, 'input'),]
    data_f = img_as_float(data)
    print 'calculating:',
    if 'c' in types:
        print 'circloids ...',
        circ = sf.conspicuity_calculation(data_f, sal_type='circloids', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True)
        data_struc.append((circ, 'circloids'))
    if 'd' in types:
        print 'int diff ...',
        id = sf.conspicuity_calculation(data_f, sal_type='int_diff', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
        data_struc.append((id, 'int diff'))
    if 'h' in types:
        print 'int hist ...',
        ih = sf.conspicuity_calculation(data_f, sal_type='int_hist', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
        data_struc.append((ih, 'int hist'))
    if 'g' in types:
        print 'int glcm ...',
        ig = sf.conspicuity_calculation(data, sal_type='int_glcm', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=morph_proc)
        data_struc.append((ig, 'int glcm'))
    if 's' in types:
        print 'int sliwin ...',
        isw = sf.conspicuity_calculation(data_f, sal_type='int_sliwin', mask=mask, calc_features=False,
                                         use_sigmoid=False, morph_proc=morph_proc)
        data_struc.append((isw, 'int sliwin'))
    if 't' in types:
        print 'texture ...',
        t = sf.conspicuity_calculation(data_f, sal_type='texture', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=False)
        data_struc.append((t, 'texture'))
    if 'b' in types:
        print 'blobs ...',
        blobs = sf.conspicuity_calculation(data_f, sal_type='blobs', mask=mask, calc_features=False, use_sigmoid=False, morph_proc=True, verbose=False)
        data_struc.append((blobs, 'blobs'))
    print 'done'

    if save:
        f = gzip.open(out_fname, 'wb', compresslevel=1)
        pickle.dump(data_struc, f)
        f.close()

    if show or save_fig:
        fig = plt.figure()
        n_imgs = len(data_struc)
        for i, (im, tit) in enumerate(data_struc):
            # r = 1 + int(i > (n_imgs / 2))
            plt.subplot(201 + 10 * (np.ceil(n_imgs / 2)) + i)
            if i == 0:
                cmap = 'gray'
            else:
                cmap = 'jet'
            plt.imshow(im, cmap, interpolation='nearest')
            plt.title(tit)
        if save_fig:
            fig.savefig(out_fname.replace('pklz', 'png'))
        if show and show_now:
            plt.show()
        plt.close('all')


def localize(data, mask, show=False, show_now=True):
    data, mask = tools.crop_to_bbox(data, mask>0)

    sals = calc_saliencies(data, mask, show=show, show_now=show_now)


def get_data_struc():
    '''
    datastruc: [(venous data, arterial data, slices, offset), ...]
    datastruc: [(venous data, arterial data, slices), ...]
    '''
    dirpath = '/home/tomas/Dropbox/Data/medical/dataset/gt/'
    # s = 4
    # data_struc = [('180_venous-GT.pklz', '180_arterial-GT-registered.pklz', range(5, 15, s)),
    #               ('183a_venous-GT.pklz', '183a_arterial-GT-registered.pklz', range(11, 23, s)),
    #               ('185a_venous-GT.pklz', '185a_arterial-GT-registered.pklz', range(9, 24, s)),  # spatne registrovany
    #               ('186a_venous-GT.pklz', '186a_arterial-GT-registered.pklz', range(1, 7, s)),
    #               ('189a_venous-GT.pklz', '189a_arterial-GT-registered.pklz', range(12, 26, s)),
    #               ('221_venous-GT.pklz', '221_arterial-GT-registered.pklz', range(17, 23, s)),
    #               ('222a_venous-GT.pklz', '222a_arterial-GT-registered.pklz', range(1, 9, s) + range(23, 30, s)),
    #               ('232_venous-GT.pklz', '232_arterial-GT-registered.pklz', range(1, 12, s) + range(17, 25, s)),
    #               # ('234_venous-GT.pklz', '234_arterial-GT.pklz', range(0, 0, s)),
    #               ('235_venous-GT.pklz', '235_arterial-GT-registered.pklz', range(11, 42, s))]

    data_struc = [('180_venous-GT.pklz', '180_arterial-GT.pklz', [(x, y) for x, y in zip(range(5, 16), range(5, 16))]),
                  ('183a_venous-GT.pklz', '183a_arterial-GT.pklz', [(x, y) for x, y in zip(range(9, 23), range(7, 21))]),
                  ('185a_venous-GT.pklz', '185a_arterial-GT.pklz', [(x, y) for x, y in zip([9,] + range(10, 24), [10,] + range(10, 24))]),
                  ('186a_venous-GT.pklz', '186a_arterial-GT.pklz', [(x, y) for x, y in zip(range(1, 8), range(1, 8))]),
                  ('189a_venous-GT.pklz', '189a_arterial-GT.pklz', [(x, y) for x, y in zip(range(12, 26), range(11, 25))]),
                  ('221_venous-GT.pklz', '221_arterial-GT.pklz', [(x, y) for x, y in zip(range(17, 22), range(17, 22))]),
                  ('222a_venous-GT.pklz', '222a_arterial-GT.pklz', [(x, y) for x, y in zip(range(2, 7) + range(24, 30), range(1, 7) + range(21, 27))]),
                  ('232_venous-GT.pklz', '232_arterial-GT.pklz', [(x, y) for x, y in zip(range(1, 12) + range(17, 25), range(1, 12) + range(16, 24))]),
                  # ('234_venous-GT.pklz', '234_arterial-GT.pklz', range(0, 0, s)),
                  ('235_venous-GT.pklz', '235_arterial-GT.pklz', [(x, y) for x, y in zip(range(4, 42), range(4, 42))])] # fujky

    # data_struc = [(dirpath + x[0], dirpath + 'registered/' + x[1], x[2]) for x in data_struc]
    data_struc = [(dirpath + x[0], dirpath + x[1], x[2]) for x in data_struc]

    return data_struc


def mrfing(im, mask, salmap, salmap_name='unknown', smoothing=True, show=False, show_now=True, save_fig=False, verbose=False):
    '''
    salmaps ... [(salmap_1, name_1), (salmap_2, name_2), ...]
    '''
    figdir = '/home/tomas/Dropbox/Work/Dizertace/figures/mrf'
    set_verbose(verbose)
    im_bb, mask_bb = tools.crop_to_bbox(im, mask)
    if smoothing:
        im_bb = tools.smoothing(im_bb, sliceId=0)

    salmap, __ = tools.crop_to_bbox(salmap, mask)

    _debug('Creating MRF object...')
    alpha = 1
    beta = 1
    scale = 0
    mrf = mrf_seg.MarkovRandomField(im_bb, mask=mask_bb, models_estim='hydohy', alpha=alpha, beta=beta, scale=scale,
                                    verbose=False)
    mrf.params['unaries_as_cdf'] = 1
    mrf.params['perc'] = 30
    mrf.params['domin_simple_estim'] = 0
    mrf.params['prob_w'] = 0.1

    unary_domin = mrf.get_unaries()[:, :, 1]
    max_prob = unary_domin.max()
    max_int = max_prob
    # saliencies_inv = []
    # for i, (im, tit) in enumerate(salmaps):
    #     salmaps[i] = skiexp.rescale_intensity(im, out_range=(0, max_int)).astype(unary_domin.dtype)
    #     saliencies_inv.append(skiexp.rescale_intensity(im, out_range=(max_int, 0)).astype(unary_domin.dtype))
    salmap = skiexp.rescale_intensity(salmap, out_range=(0, max_int)).astype(unary_domin.dtype)
    salmap_inv = skiexp.rescale_intensity(salmap, out_range=(max_int, 0)).astype(unary_domin.dtype)

    unaries_domin_sal = np.dstack((unary_domin, salmap_inv.reshape(-1, 1)))

    alphas = [1, 10, 100]
    beta = 1
    for alpha in alphas:
        mrf = mrf_seg.MarkovRandomField(im_bb, mask=mask_bb, models_estim='hydohy', alpha=alpha, beta=beta, scale=scale,
                                        verbose=False)
        _debug('Optimizing MRF with unary term: %s' % salmap_name)
        # mrf.set_unaries(unary.astype(np.int32))
        mrf.alpha = alpha
        mrf.beta = beta
        mrf.models = []
        mrf.set_unaries(unaries_domin_sal)
        res = mrf.run(resize=False)
        res = res[0, :, :]
        res = np.where(mask_bb, res, -1)

        # morphology
        lbl_obj = 1
        lbl_dom = 0
        rad = 5
        res_b = res == lbl_obj
        res_b = skimor.binary_opening(res_b, selem=skimor.disk(rad))
        res = np.where(res_b, lbl_obj, lbl_dom)
        res = np.where(mask_bb, res, -1)

        if show or save_fig:
            if save_fig:
                fig = plt.figure(figsize=(24, 14))
            else:
                plt.figure()
            plt.subplot(141), plt.imshow(im, 'gray', interpolation='nearest'), plt.title('input')
            plt.subplot(142), plt.imshow(res, interpolation='nearest'), plt.title('result - %s' % tit)
            plt.subplot(143), plt.imshow(unaries_domin_sal[:, 0, 0].reshape(im_bb.shape), 'gray', interpolation='nearest')
            plt.title('unary #1')
            plt.subplot(144), plt.imshow(unaries_domin_sal[:, 0, 1].reshape(im_bb.shape), 'gray', interpolation='nearest')
            plt.title('unary #2')

            if save_fig:
                figname = 'unary_%s_alpha_%i_rad_%i.png' % (tit, alpha, rad)
                fig.savefig(os.path.join(figdir, figname), dpi=100, bbox_inches='tight', pad_inches=0)
            if show_now:
                plt.show()

        return res, unaries_domin_sal[:, 0, 0].reshape(im_bb.shape), unaries_domin_sal[:, 0, 1].reshape(im_bb.shape)


def seg_acc_to_xlsx(acc):
    workbook = xlsxwriter.Workbook('/home/tomas/Dropbox/Data/liver_segmentation/final_alg/liver_seg_stats.xlsx')
    worksheet = workbook.add_worksheet()
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})
    col_fmea = 3
    col_prec = 4

    worksheet.write(0, 0, 'DATA', bold)
    worksheet.write(0, 2, 'F-MEASURE', bold)
    worksheet.write(0, 6, 'PRECISION', bold)
    worksheet.write(0, 9, 'RECALL', bold)

    worksheet.write(1, col_gc_fmea, 'Grow Cut', bold)
    worksheet.write(1, col_ls_fmea, 'Level Sets', bold)
    worksheet.write(1, col_gc_prec, 'Grow Cut', bold)
    worksheet.write(1, col_ls_prec, 'Level Sets', bold)
    worksheet.write(1, col_gc_rec, 'Grow Cut', bold)
    worksheet.write(1, col_ls_rec, 'Level Sets', bold)

################################################################################
################################################################################
if __name__ == '__main__':
    # TODO 4: MRF experimenty
    # TODO 5: vybrat idealni kombinaci salmap
    # TODO 6: porovnat data

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # mrfing
    # datapath = '/home/tomas/Dropbox/Data/medical/dataset/gt/salmaps'
    # files = []
    # for (dirpath, dirnames, filenames) in os.walk(datapath):
    #     filenames = [x for x in filenames if x.split('.')[-1] == 'pklz']
    #     for fname in filenames:
    #         files.append(os.path.join(dirpath, fname))
    #     break

    files = ['/home/tomas/Dropbox/Data/medical/dataset/gt/salmaps/vyber/183a_venous-GT-sm-17.pklz',]

    sheet_res = []
    fig = plt.figure(figsize=(24, 14))
    for i, fname in enumerate(files):
        # f = files[i]
        # print i, f
        with gzip.open(fname, 'rb') as f:
            fcontent = f.read()
        data = pickle.loads(fcontent)

        n_imgs = len(data)
        im = data[0][0]
        mask = data[1][0]
        # plt.figure()
        # plt.imshow(mask), plt.colorbar()
        # plt.show()
        data_res = []
        for i in range(2, len(data)):
            salmap, tit = data[i]
            res, unary_bgd, unary_obj = mrfing(im, mask>0, salmap, salmap_name=tit, smoothing=True, show=False, show_now=False)
            res = np.where(res == -1, 0, res).astype(np.uint8)

            im_bb, mask_bb = tools.crop_to_bbox(im, mask>0)
            salmap, __ = tools.crop_to_bbox(salmap, mask>0)

            results = (im_bb, salmap, res, unary_bgd, unary_obj, tit)

            gt, __ = tools.crop_to_bbox((mask == 1).astype(np.uint8), mask > 0)
            precision, recall, f_measure = tools.segmentation_accuracy(res, gt)
            accuracy = (f_measure, precision, recall)
            data_res.append((tit, f_measure, precision, recall))

            # fig = plt.figure(figsize=(24, 14))
            im_overlay = res + 3 * gt
            im_overlay = skicol.label2rgb(im_overlay, colors=['magenta', 'yellow', 'green', 'white'], bg_label=0, alpha=1)
            plt.suptitle('im | %s | res | gt || res bounds | gt bounds | overlay' % tit)
            plt.subplot(241), plt.imshow(im_bb * mask_bb , 'gray', interpolation='nearest'), plt.axis('off')
            plt.subplot(242), plt.imshow(salmap, 'jet', interpolation='nearest'), plt.axis('off')
            plt.subplot(243), plt.imshow(res, 'gray', interpolation='nearest'), plt.axis('off')
            plt.subplot(244), plt.imshow(gt, 'gray', interpolation='nearest'), plt.axis('off')
            plt.subplot(245), plt.imshow(skiseg.mark_boundaries(im_bb * mask_bb, res, color=(1, 0, 0), mode='thick'), interpolation='nearest'), plt.axis('off')
            plt.subplot(246), plt.imshow(skiseg.mark_boundaries(im_bb * mask_bb, gt, color=(1, 0, 0), mode='thick'), interpolation='nearest'), plt.axis('off')
            plt.subplot(247), plt.imshow(im_overlay, interpolation='nearest'), plt.axis('off')
            fig_fname = fname.replace('vyber/', 'vyber/res/').replace('.pklz', '-%s-seg.png' % tit.replace(' ', '_'))
            fig.savefig(fig_fname)
            # plt.close('all')
            fig.clf()

        sheet_res.append(data_res)

        # for (im, salmap, res, unary_bgd, unary_obj, tit), (f_measure, precision, recall) in zip(results, accuracy):
        #     plt.figure()
        #     # f_measure, precision, recall = acc
        #     plt.suptitle('f=%.3f, p=%.3f, r=%.3f' % (f_measure, precision, recall))
        #     plt.subplot(231), plt.imshow(im, 'gray', interpolation='nearest'), plt.title('input')
        #     plt.subplot(232), plt.imshow(salmap, 'jet', interpolation='nearest'), plt.title('salmap - %s' % tit)
        #     plt.subplot(233), plt.imshow(res, 'gray', interpolation='nearest'), plt.title('result')
        #     plt.subplot(234), plt.imshow(unary_bgd, 'jet', interpolation='nearest'), plt.title('unary bgd')
        #     plt.subplot(235), plt.imshow(unary_obj, 'jet', interpolation='nearest'), plt.title('unary obj')
        #     plt.subplot(236), plt.imshow(tum_mask_bb, 'jet', interpolation='nearest'), plt.title('gt')
        # plt.show()

        # fig = plt.figure()
        # n_imgs = len(data)
        # for i, (im, tit) in enumerate(data):
        #     # r = 1 + int(i > (n_imgs / 2))
        #     plt.subplot(201 + 10 * (np.ceil(n_imgs / 2)) + i)
        #     if i == 0:
        #         cmap = 'gray'
        #     else:
        #         cmap = 'jet'
        #     plt.imshow(im, cmap, interpolation='nearest')
        #     plt.title(tit)
        # plt.show()

    # f = gzip.open('/home/tomas/Dropbox/Data/medical/dataset/gt/salmaps/180_venous-GT-sm-5.pklz', 'rb')
    # fcontent = f.read()
    # salmaps = pickle.loads(fcontent)
    # f.close()
    #
    # plt.figure()
    # n_imgs = len(salmaps)
    # for i, (im, tit) in enumerate(salmaps):
    #     # r = 1 + int(i > (n_imgs / 2))
    #     plt.subplot(201 + 10 * (np.ceil(n_imgs / 2)) + i)
    #     if i == 0:
    #         cmap = 'gray'
    #     else:
    #         cmap = 'jet'
    #     plt.imshow(im, cmap, interpolation='nearest')
    #     plt.title(tit)
    # plt.show()

    # ds = get_data_struc()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # vypocet salmap --------------------
    # for i, (data_ven_fn, data_art_fn, slics) in enumerate(ds):
    #     if i < 4:
    #         continue
    #     print '\n#%i/%i: ' % (i + 1, len(ds))
    #     data_ven, mask_ven, vs = tools.load_pickle_data(data_ven_fn)
    #     data_art, mask_art, vs = tools.load_pickle_data(data_art_fn)
    #     data_ven = tools.windowing(data_ven)
    #     data_art = tools.windowing(data_art)
    #     for j, (s_ven, s_art) in enumerate(slics):
    #         # if j < 2:
    #         #     continue
    #         im_ven = data_ven[s_ven,...]
    #         im_art = data_art[s_art,...]
    #         mask_im_ven = mask_ven[s_ven,...] > 0
    #         mask_im_art = mask_art[s_art,...] > 0
    #
    #         # venous data
    #         print '    ven #%i/%i' % (j + 1, len(slics)),
    #         dirs = data_ven_fn.split('/')
    #         fname = dirs[-1]
    #         salmap_fname = fname.replace('.pklz', '-sm-%i.pklz' % s_ven)
    #         dirp = '/'.join(dirs[:-1])
    #         out_ven = os.path.join(dirp, 'salmaps', salmap_fname)
    #         salmaps_ven = calc_saliencies(im_ven, mask_im_ven, save=True, save_fig=True, out_fname=out_ven)
    #
    #         # arterial data
    #         print '    art #%i/%i' % (j + 1, len(slics)),
    #         dirs = data_art_fn.split('/')
    #         fname = dirs[-1]
    #         salmap_fname = fname.replace('.pklz', '-sm-%i.pklz' % s_art)
    #         dirp = '/'.join(dirs[:-1])
    #         out_art = os.path.join(dirp, 'salmaps', salmap_fname)
    #         salmaps_art = calc_saliencies(im_art, mask_im_art, save=True, save_fig=True, out_fname=out_art)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # zoomovani tak aby mela data stejny pocet rezu
    # d1, m1, vs1 = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/222a_venous-GT.pklz')
    # d2, m2, vs2 = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/222a_arterial-GT.pklz')
    # d3 = tools.match_size(d2, d1.shape)
    # m3 = tools.match_size(m2, m1.shape)
    #
    # # datap = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/222a_arterial-GT.pklz', return_datap=True)
    #
    # slab = {'none': 0, 'liver': 1, 'lesions': 6}
    # datap_1 = {'data3d': d1, 'segmentation': m1, 'voxelsize_mm': vs1, 'slab': slab}
    # datap_2 = {'data3d': d3, 'segmentation': m3, 'voxelsize_mm': vs2, 'slab': slab}
    # app = QtGui.QApplication(sys.argv)
    # le = SegViewer(datap1=datap_1, datap2=datap_2)
    # le.show()
    # app.exec_()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # vizualizace dat
    # for i, (vfn, afn, slcs, off) in enumerate(ds):
    # i = 8
    # vfn, afn, slcs = ds[i]
    #
    # print i, ' ... ', vfn.split('/')[-1], afn.split('/')[-1]
    #
    # datap_1 = tools.load_pickle_data(vfn, return_datap=True)
    # datap_2 = tools.load_pickle_data(afn, return_datap=True)
    # # d2, m2, vs2 = tools.load_pickle_data(afn)
    # # d2 = tools.match_size(d2, datap_1['data3d'].shape)
    # # m2 = tools.match_size(m2, datap_1['segmentation'].shape)
    # # slab = {'none': 0, 'liver': 1, 'lesions': 6}
    # # datap_2 = {'data3d': d2, 'segmentation': m2, 'voxelsize_mm': vs2, 'slab': slab}
    #
    # app = QtGui.QApplication(sys.argv)
    # le = SegViewer(datap1=datap_1, datap2=datap_2)
    # le.show()
    # app.exec_()

    # files = []
    # for (dirpath, dirnames, filenames) in os.walk('/home/tomas/Dropbox/Data/medical/dataset/gt'):
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

    # nejaka vizualizace
    # data, gt_mask, voxel_size = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/234_arterial-.pklz')
    # data, gt_mask, voxel_gtsize = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/234_venous-GT.pklz')
    # data = tools.windowing(data)
    # tools.show_3d(gt_mask, range=False)

    # data, mask, _ = tools.load_pickle_data('/home/tomas/Dropbox/Data/medical/dataset/gt/183a_venous-GT.pklz')
    # slice_ind = 17
    # data = data[slice_ind,...]
    # mask = mask[slice_ind,...]
    # data = tools.windowing(data)
    #
    # localize(data, mask, show=True)

    # tools.show_3d(data, range=False)
    # plt.figure()
    # plt.imshow(data, 'gray', interpolation='nearest')
    # plt.show()