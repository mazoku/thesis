from __future__ import division

import numpy as np
import scipy.stats as scista
import matplotlib.pyplot as plt
import cv2

import lession_localization as lesloc
import fuzzy as fuz

import os
import sys
import gzip
import cPickle as pickle

import skimage.measure as skimea
import skimage.exposure as skiexp
import skimage.morphology as skimor
import skimage.segmentation as skiseg
from sklearn.cluster import KMeans

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

    if not fnames:  # no tumors found
        return None
    elif len(fnames) > 1:
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


def extract_and_save_pairs():
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
        # gt_mask1 = gt_mask1 == 1

        __, gt_mask2, __ = tools.load_pickle_data(im2_fn)
        gt_mask2 = gt_mask2[s2, ...]
        gt_mask2, __ = tools.crop_to_bbox(gt_mask2, gt_mask2 > 0)
        # gt_mask2 = gt_mask2 == 1

        # finding corresponding files
        fname1 = find_file(im1_fn, s1, files)
        fname2 = find_file(im2_fn, s2, files)

        # processing
        if fname1 is not None and fname2 is not None:
            # data1, gt_mask, voxel_size = tools.load_pickle_data(f)
            with gzip.open(fname1, 'rb') as f:
                fcontent = f.read()
            im1, sm1, res1 = pickle.loads(fcontent)

            with gzip.open(fname2, 'rb') as f:
                fcontent = f.read()
            im2, sm2, res2 = pickle.loads(fcontent)

            # identify_tumors(res1, res2, gt_mask1, gt_mask2)
            pairs = pair_up_tumors(im1, res1, gt_mask1, im2, res2, gt_mask2)

            # ukladani na disk
            for j, pair in enumerate(pairs):
                __, __, pair_type = pair
                kw = fname1.split('/')[-1].split('.')[0].split('-')
                nmb = kw[0].split('_')[0]
                sl1 = kw[-2]
                sl2 = fname2.split('/')[-1].split('.')[0].split('-')[-2]
                outfn = '/'.join(fname1.split('/')[:-1]) + '/pairs/%s_sl-%s-%s_p-%i_tp-%s.pklz' % (
                nmb, sl1, sl2, j, pair_type)
                with gzip.open(outfn, 'wb', compresslevel=1) as f:
                    pickle.dump(pair, f)


def pair_up_tumors(im1, res1_in, gt1_in, im2, res2_in, gt2_in):
    # olabelovat res1 a res2
    # iterovat pres labely res1
    #   je obj v gt1?
    #   ne -> zahodit, inkrementovat FP
    #   ano -> je na stejne pozici obj v res2?
    #       ano -> vytvorit par a ulozit
    #           -> clearnout prislusny obj v res2
    #       ne -> pouzit k vytvoreni paru oblast odpovidajici obj z res1
    im2_res = tools.resize_ND(im2, shape=im1.shape)
    res1 = res1_in.copy()
    res2 = tools.resize_ND(res2_in, shape=res1.shape)
    # gt1 = gt1_in.copy()
    gt1_liver = gt1_in > 0
    gt1_tum = gt1_in == 1
    gt2_liver = tools.resize_ND((gt2_in > 0).astype(np.uint8), shape=gt1_liver.shape)
    gt2_tum = tools.resize_ND((gt2_in == 1).astype(np.uint8), shape=gt1_tum.shape)
    # gt2 = tools.resize_ND(gt2_in.astype(np.uint8), shape=gt1.shape)
    lab_res1 = skimea.label(res1)
    lab_res2 = skimea.label(res2)
    # lab_res2 = tools.resize_ND(lab_res2, shape=lab_res1.shape)

    fp = 0
    pairs = []
    for l in range(1, lab_res1.max() + 1):
        obj1 = lab_res1 == l
        if (obj1 * gt1_tum).sum() > 0:  # prekryvaji se -> jedna se o TP
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
            pairs.append(((im1, gt1_liver, obj1), (im2_res, gt2_liver, obj2), pair_type))  # ulozim korespondujici par lozisek
        else:
            fp += 1

    # zbyva zanalyzovat objekty v druhych datech, ktere nebyly sparovany s objekty v prvnich datech (nebyly vynulovany)
    res_labs = np.unique(lab_res2)
    for l in res_labs[1:]:  # 0 neni objekt ale pozadi
        obj2 = lab_res2 == l
        if (obj2 * gt2_tum).sum() > 0:  # prekryvaji se -> jedna se o TP
            obj1 = obj2.copy()  # prislusny obj1 musi byt neolabelovana oblast, jinak by byl jiz sparovan
            pair_type = 3  # prvni objekt nenalezen
            pairs.append(((im1, gt1_liver, obj1), (im2_res, gt2_liver, obj2), pair_type))

    # relab1 = np.zeros_like(lab_res1)
    # relab2 = np.zeros_like(lab_res2)
    # for i, (o1, o2, __) in enumerate(pairs):
    #     relab1 = np.where(o1, i + 1, relab1)
    #     relab2 = np.where(o2, i + 1, relab2)
    # plt.figure()
    # plt.subplot(121), plt.imshow(relab1, interpolation='nearest')
    # plt.subplot(122), plt.imshow(relab2, interpolation='nearest')
    # plt.show()

    return pairs


def classify(pair):
    '''
    pair ... ((im1, liver_mask1, tum_mask1), (im2, liver_mask2, mask2), pair_type)
    '''
    (im1, liver_mask1, tum_mask1), (im2, liver_mask2, tum_mask2), pair_type = pair

    healthy_m1 = liver_mask1 - tum_mask1
    liver_mean1 = im1[np.nonzero(healthy_m1)].mean()
    tum_mean1 = im1[np.nonzero(tum_mask1)].mean()
    eps = 5
    if tum_mean1 < (liver_mean1 - eps):
        type1 = 'hypo'
    elif (liver_mean1 - eps) < tum_mean1 < (liver_mean1 + eps):
        type1 = 'iso'
    else:
        type1 = 'hyper'

    healthy_m2 = liver_mask2 - tum_mask2
    # plt.figure()
    # plt.subplot(131), plt.imshow(liver_mask2, 'gray')
    # plt.subplot(132), plt.imshow(tum_mask2, 'gray')
    # plt.subplot(133), plt.imshow(healthy_m2, 'gray')
    # plt.show()
    liver_mean2 = im2[np.nonzero(healthy_m2)].mean()
    tum_mean2 = im2[np.nonzero(tum_mask2)].mean()
    if tum_mean2 < (liver_mean2 - eps):
        type2 = 'hypo'
    elif (liver_mean2 - eps) < tum_mean2 < (liver_mean2 + eps):
        type2 = 'iso'
    else:
        type2 = 'hyper'

    print 'means: h1=%.1f, t1=%.1f -> %s, h2=%.1f, t2=%.1f -> %s' % (liver_mean1, tum_mean1, type1, liver_mean2, tum_mean2, type2)

    plt.figure()
    plt.subplot(231), plt.imshow(im1, 'gray')
    plt.subplot(232), plt.imshow(liver_mask1, 'gray')
    plt.subplot(233), plt.imshow(tum_mask1, 'gray')
    plt.subplot(234), plt.imshow(im2, 'gray')
    plt.subplot(235), plt.imshow(liver_mask2, 'gray')
    plt.subplot(236), plt.imshow(tum_mask2, 'gray')
    plt.show()


def dominant_colors(img, mask=None, k=3):
    if mask is None:
        mask = np.ones_like(img)
    clt = KMeans(n_clusters=k)
    data = img[np.nonzero(mask)]
    clt.fit(data.reshape(len(data), 1))
    cents = clt.cluster_centers_

    return cents


def color_clustering(img, mask=None, colors=None, k=3):
    if mask is None:
        mask = np.ones(img.shape[:2])
    if colors is None:
        colors = dominant_colors(img, mask=mask, k=k)
    diffs = np.array([np.abs(img - x) for x in colors])

    clust = np.argmin(diffs, 0)
    clust = np.where(mask, clust, -1)

    print colors
    plt.figure()
    plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest')
    plt.subplot(122), plt.imshow(clust, 'jet', interpolation='nearest')
    plt.show()

    return clust, colors


def analyze_im_for_features(fn, show=True, show_now=True):
    mask_fn = fn.replace('.%s' % (fn.split('.')[-1]), '_mask.%s' % (fn.split('.')[-1]))
    im = cv2.imread(fn, 0)
    ims = tools.smoothing(im, d=10, sigmaColor=50, sigmaSpace=10)
    mask = cv2.imread(mask_fn, 0) > 0
    pts = ims[np.nonzero(mask)]

    # fuz.fcm(ims, mask=mask, n_clusters=2, show=True)
    # color_clustering(ims, mask=mask, k=2)

    hist, bins = skiexp.histogram(pts)
    glcm = tools.graycomatrix_3D(ims, mask=mask)
    # glcm *= glcm > 3

    # glcm_p = glcm > (0.4 * glcm.max())
    glcm_p = glcm > 2 * (glcm[np.nonzero(glcm)].mean())
    # glcm_p = glcm > (np.median(glcm[np.nonzero(glcm)]))
    glcm_pts = np.nonzero(glcm_p)
    width = glcm_pts[1].max() - glcm_pts[1].min()

    plt.figure()
    plt.subplot(121), plt.imshow(glcm)
    plt.subplot(122), plt.imshow(glcm_p, 'gray')
    # plt.show()

    mu, std = scista.norm.fit(pts.flatten())
    # b = scista.norm.pdf(bins, mu, std)
    # k = hist.max() / b.max()
    # plt.figure()
    # plt.plot(bins, hist, 'b-')
    # plt.plot(bins, k * b, 'r-')
    # plt.show()

    if show:
        plt.figure()
        plt.suptitle('std=%.2f, width=%i' % (std, width))
        plt.subplot(131), plt.imshow(skiseg.mark_boundaries(im, mask, color=(1,0,0), mode='thick'), 'gray', interpolation='nearest')
        # plt.subplot(142), plt.imshow(mask, 'gray', interpolation='nearest')
        plt.subplot(132), plt.plot(bins, hist, 'b-')
        plt.xlim([0, 255])
        plt.subplot(133), plt.imshow(glcm, interpolation='nearest')#, vmax=5 * glcm.mean())
        if show_now:
            plt.show()

    return bins, hist, glcm


def detect_capsule(fn):
    mask_fn = fn.replace('.%s' % (fn.split('.')[-1]), '_mask.%s' % (fn.split('.')[-1]))
    im = cv2.imread(fn, 0)
    mask = cv2.imread(mask_fn, 0)

    # plt.figure()
    # plt.subplot(131), plt.imshow(skiseg.mark_boundaries(im, mask, color=(1,0,0), mode='thick')), plt.axis('off')
    # plt.show()

    mask_in = mask > 254 # 220
    mask_in = skimor.remove_small_holes(mask_in)
    mask_in = skimor.binary_opening(mask_in, selem=skimor.disk(5))
    # mask_caps = mask > 20 * mask < 100
    mask_caps = (20 < mask) * (mask < 150)
    # mask_caps = (mask > 20) - mask_in

    mask_caps_c = skimor.remove_small_holes(mask_caps, min_size=500)

    mask_caps_e = skimor.binary_erosion(mask_caps_c, selem=skimor.disk(3)) * mask_caps

    inside = 2 * mask_caps_c - mask_in

    plt.figure()
    plt.suptitle('in, detected objs, filled caps, inside test')
    plt.subplot(141), plt.imshow(im, 'gray', interpolation='nearest')
    plt.subplot(142), plt.imshow(2 * mask_caps_e + mask_in, 'gray', interpolation='nearest')
    # plt.subplot(141), plt.imshow(mask_in, 'gray', interpolation='nearest')
    # plt.subplot(142), plt.imshow(mask_caps, 'gray', interpolation='nearest')
    plt.subplot(143), plt.imshow(mask_caps_c, 'gray', interpolation='nearest')
    plt.subplot(144), plt.imshow(inside, 'gray', interpolation='nearest')
    # plt.subplot(236), plt.imshow(mask_in + mask_caps, 'gray', interpolation='nearest')
    plt.show()


def hemorr_vs_capsule():
    hem_fn = '/home/tomas/Dropbox/Work/Dizertace/figures/lesion_local/feature_hemorr.png'
    hem_mask_fn = '/home/tomas/Dropbox/Work/Dizertace/figures/lesion_local/feature_hemorr_mask2.png'
    hem = cv2.imread(hem_fn, 0)
    hem_mask = cv2.imread(hem_mask_fn, 0) > 0

    caps_fn = '/home/tomas/Dropbox/Data/medical/features/imgTum_zoom.png'
    caps_mask_fn = '/home/tomas/Dropbox/Data/medical/features/imgTum_zoom_mask.png'
    caps = cv2.imread(caps_fn, 0)
    caps_mask = cv2.imread(caps_mask_fn, 0) > 0

    hem_comp = tools.compactness(hem_mask)
    caps_comp = tools.compactness(caps_mask)

    print 'hem comp = %.3f, caps comp = %.3f' % (hem_comp, caps_comp)

    #TODO: compactness
    plt.figure()
    plt.subplot(141), plt.imshow(skiseg.mark_boundaries(hem, hem_mask, color=(1,0,0), mode='thick'), 'gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(142), plt.imshow(hem_mask, 'gray', interpolation='nearest'), plt.axis('off')
    plt.subplot(143), plt.imshow(skiseg.mark_boundaries(caps, caps_mask, color=(1,0,0), mode='thick'), 'gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(144), plt.imshow(caps_mask, 'gray', interpolation='nearest'), plt.axis('off')
    # plt.subplot(141), plt.imshow(hem, 'gray', interpolation='nearest')
    # plt.subplot(142), plt.imshow(hem_mask, 'gray', interpolation='nearest')
    # plt.subplot(143), plt.imshow(caps, 'gray', interpolation='nearest')
    # plt.subplot(144), plt.imshow(caps_mask, 'gray', interpolation='nearest')
    plt.show()


def features():
    # HOMO
    # print 'homo ...',
    # fn_homo = '/home/tomas/Dropbox/Data/medical/features/cyst_venous.png'
    # analyze_im_for_features(fn_homo, show_now=False)
    # print 'done'

    # HETERO
    # print 'hetero ...',
    # fn_hetero = '/home/tomas/Dropbox/Data/medical/features/flk_arterial.png'
    # analyze_im_for_features(fn_hetero)
    # print 'done'

    # SCAR
    # print 'scar ...',
    # # fn = '/home/tomas/Dropbox/Data/medical/features/fnh_arterial.png'
    # fn = '/home/tomas/Dropbox/Data/medical/features/scar1.png'
    # analyze_im_for_features(fn)
    # print 'done'

    # CALCIFICATION
    # print 'calcification ...',
    # fn = '/home/tomas/Dropbox/Data/medical/features/flk_nect.png'
    # analyze_im_for_features(fn)
    # print 'done'

    # CAPSULE HYPO
    # fn =

    # CAPSULE HYPER
    # print 'capsule ...',
    # fn = '/home/tomas/Dropbox/Data/medical/features/imgTum.png'
    # detect_capsule(fn)
    # print 'done'

    # HEMORRHAGE
    # print 'hemorrhage ...',
    # fn = '/home/tomas/Dropbox/Work/Dizertace/figures/lesion_local/feature_hemorr.png'
    # analyze_im_for_features(fn)
    # print 'done'

    # HEMORR vs CAPSULE
    hemorr_vs_capsule()


################################################################################
################################################################################
if __name__ == '__main__':
    # pair up tumors ------
    # extract_and_save_pairs()
    # ------

    # features
    features()
    sys.exit(0)

    # classification ------
    files = []
    datadir = '/home/tomas/Dropbox/Data/medical/dataset/gt/salmaps/res/best/pairs/'
    for (dirpath, dirnames, filenames) in os.walk(datadir):
        for fname in filenames:
            files.append(os.path.join(dirpath, fname))
    for f in files:
        with gzip.open(f, 'rb') as f:
            fcontent = f.read()
        pair = pickle.loads(fcontent)
        classify(pair)
    # ------