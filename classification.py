from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import lession_localization as lesloc

import os
import sys
import gzip
import cPickle as pickle

import skimage.measure as skimea
import skimage.exposure as skiexp

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


def find_file(q_fname, q_sl, files):
    base_fn = q_fname.split('/')[-1].split('.')[0]
    fnames = [x for x in files if base_fn in x and '-%i-' % q_sl in x]

    if len(fnames) > 1:
        raise ValueError('More than one file found.')
    return fnames[0]


def filter_out_fp(im, gt):
    lab_im = skimea.label(im)
    lab_gt = skimea.label(gt)

    plt.figure()
    plt.subplot(121), plt.imshow(im, 'gray')
    plt.subplot(122), plt.imshow(lab_im, 'gray')
    plt.show()

    filtered = np.zeros_like(lab_im)

    for i in range(1, lab_im.max() + 1):
        inters = (lab_im == i) * lab_gt
        if (inters > 0).sum() > 0:
            labs = np.unique(inters)
            cnts = []
            for j in labs:
                cnts.append((inters == j).sum())
            cnts[0] = 0  # zeroing label 0 which is background
            lbl = labs[np.argmax(cnts)]
            filtered = np.where(inters, lbl, filtered)

    plt.figure()
    plt.subplot(131), plt.imshow(gt, 'gray', interpolation='nearest'), plt.title('gt')
    plt.subplot(132), plt.imshow(lab_im, 'gray', interpolation='nearest'), plt.title('lab im')
    plt.subplot(133), plt.imshow(filtered, 'gray', interpolation='nearest'), plt.title('filtered lab im')
    plt.show()

    return filtered


def identify_tumors(im1, im2, gt1, gt2):
    im1 = filter_out_fp(im1, gt1)


def pair_up_tumors(res1_in, res2_in, gt1_in, gt2_in):
    # olabelovat res1 a res2
    # iterovat pres labely res1
    #   je obj v gt1?
    #   ne -> zahodit, inkrementovat FP
    #   ano -> je na stejne pozici obj v res2?
    #       ano -> vytvorit par a ulozit
    #           -> clearnout prislusny obj v res2
    #       ne -> pouzit k vytvoreni paru oblast odpovidajici obj z res1
    res1 = res1_in.copy()
    res2 = tools.resize_ND(res2_in, shape=res1.shape)
    gt1 = gt1_in.copy()
    gt2 = tools.resize_ND(gt2_in.astype(np.uint8), shape=gt1.shape)
    lab_res1 = skimea.label(res1)
    lab_res2 = skimea.label(res2)
    # lab_res2 = tools.resize_ND(lab_res2, shape=lab_res1.shape)

    fp = 0
    pairs = []
    for l in range(1, lab_res1.max() + 1):
        obj1 = lab_res1 == l
        if (obj1 * gt1).sum() > 0:  # prekryvaji se -> jedna se o TP
            if (obj1 * lab_res2).sum() > 0:  # tumor detekovan i v druhych datech
                lbls = lab_res2[np.nonzero(obj1 * lab_res2)]
                uni_lbls = np.unique(lbls)
                if len(uni_lbls) > 1:  # v pripade vice objektu vyberu ten nejcetnejsi
                    hist, bins = skiexp.histogram(lbls)
                    lbl = bins[np.argmax(hist)]
                else:
                    lbl = uni_lbls[0]  # jinak vezmu ten jediny
                obj2 = lab_res2 == lbl
                lab_res2 = np.where(obj2, 0, lab_res2)  # vynuluji - na konci se bude analyzovat zbytek v lab_res2
                pair_type = 1  # v obou datech byly objekty
            else:  # v druhych datech tumor nedetekovan
                obj2 = obj1.copy()  # pouziji masku z res1
                pair_type = 2  # druhy objekt nenalezen
            pairs.append((obj1, obj2, pair_type))  # ulozim korespondujici par lozisek
        else:
            fp += 1

    # zbyva zanalyzovat objekty v druhych datech, ktere nebyly sparovany s objekty v prvnich datech (nebyly vynulovany)
    res_labs = np.unique(lab_res2)
    for l in res_labs[1:]:  # 0 neni objekt ale pozadi
        obj2 = lab_res2 == l
        if (obj2 * gt2).sum() > 0:  # prekryvaji se -> jedna se o TP
            obj1 = obj2.copy()  # prislusny obj1 musi byt neolabelovana oblast, jinak by byl jiz sparovan
            pair_type = 3  # prvni objekt nenalezen
            pairs.append((obj1, obj2, pair_type))

    relab1 = np.zeros_like(lab_res1)
    relab2 = np.zeros_like(lab_res2)
    for i, (o1, o2, __) in enumerate(pairs):
        relab1 = np.where(o1, i + 1, relab1)
        relab2 = np.where(o2, i + 1, relab2)
    # plt.figure()
    # plt.subplot(121), plt.imshow(relab1, interpolation='nearest')
    # plt.subplot(122), plt.imshow(relab2, interpolation='nearest')
    # plt.show()

    return pairs


################################################################################
################################################################################
if __name__ == '__main__':
    datas = lesloc.get_data_struc(reshape=True)
    # datas = reshape_struc(datas)

    files = []
    # datadir = '/home/tomas/Dropbox/Data/medical/dataset/gt/salmaps/vyber/hypo/res_pklz'
    datadir = '/home/tomas/Dropbox/Data/medical/dataset/gt/salmaps/res/best'
    for (dirpath, dirnames, filenames) in os.walk(datadir):
        filenames = [x for x in filenames if x.split('.')[-1] == 'pklz']
        for fname in filenames:
            files.append(os.path.join(dirpath, fname))
        break

    for i, (im1_fn, im2_fn, s1, s2) in enumerate(datas):
        print 'data # %i/%i ...' % (i + 1, len(datas))
        # getting ground-truth data
        __, gt_mask1, __ = tools.load_pickle_data(im1_fn)
        gt_mask1 = gt_mask1[s1, ...]
        gt_mask1, __ = tools.crop_to_bbox(gt_mask1, gt_mask1 > 0)
        gt_mask1 = gt_mask1 == 1

        __, gt_mask2, __ = tools.load_pickle_data(im2_fn)
        gt_mask2 = gt_mask2[s2, ...]
        gt_mask2, __ = tools.crop_to_bbox(gt_mask2, gt_mask2 > 0)
        gt_mask2 = gt_mask2 == 1

        # finding corresponding files
        fname1 = find_file(im1_fn, s1, files)
        fname2 = find_file(im2_fn, s2, files)

        # processing
        if fname1 and fname2:
            # data1, gt_mask, voxel_size = tools.load_pickle_data(f)
            with gzip.open(fname1, 'rb') as f:
                fcontent = f.read()
            im1, sm1, res1 = pickle.loads(fcontent)

            with gzip.open(fname2, 'rb') as f:
                fcontent = f.read()
            im2, sm2, res2 = pickle.loads(fcontent)

            # identify_tumors(res1, res2, gt_mask1, gt_mask2)
            pairs = pair_up_tumors(res1, res2, gt_mask1, gt_mask2)
            for j, (obj1, obj2, pair_type) in enumerate(pairs):
                kw = fname1.split('/')[-1].split('.')[0].split('-')
                nmb = kw[0].split('_')[0]
                sl1 = kw[-2]
                sl2 = fname2.split('/')[-1].split('.')[0].split('-')[-2]
                outfn = '/'.join(fname1.split('/')[:-1]) + '/pairs/%s_sl-%s-%s_tp-%s.pklz' % (nmb, sl1, sl2, pair_type)
                with gzip.open(outfn, 'wb', compresslevel=1) as f:
                    pickle.dump(pairs, f)

            # plt.figure()
            # plt.subplot(241), plt.imshow(im1, 'gray', interpolation='nearest'), plt.axis('off')
            # plt.subplot(242), plt.imshow(sm1, interpolation='nearest'), plt.axis('off')
            # plt.subplot(243), plt.imshow(res1, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('res')
            # plt.subplot(244), plt.imshow(gt_mask1, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('gt')
            #
            # plt.subplot(245), plt.imshow(im2, 'gray', interpolation='nearest'), plt.axis('off')
            # plt.subplot(246), plt.imshow(sm2, interpolation='nearest'), plt.axis('off')
            # plt.subplot(247), plt.imshow(res2, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('res')
            # plt.subplot(248), plt.imshow(gt_mask2, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('gt')
            # plt.show()