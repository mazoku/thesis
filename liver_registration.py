# from __future__ import division

import SimpleITK as sitk
from skimage import data
import skimage.transform as skitra

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PyQt4 import QtGui
import cv2

import cPickle as pickle
import gzip

from itertools import product

# import warnings
# warnings.filterwarnings("ignore")

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in liver_registration.py. You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

if os.path.exists('/home/tomas/projects/data_viewers/'):
    sys.path.insert(0, '/home/tomas/projects/data_viewers/')
    from dataviewers.seg_viewer import SegViewer
else:
    print 'Import error in liver_registration.py. You need to import package imtools: https://github.com/mazoku/dataviewers'
    sys.exit(0)

prubeh = []
metric_vals = []
iteration = 0


def show_initialization(fixed, moving):
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 / 2. + simg2 / 2.)

    fixed_im = sitk.GetArrayFromImage(simg1)
    moving_im = sitk.GetArrayFromImage(simg2)
    cimg_im = sitk.GetArrayFromImage(cimg)

    plt.figure()
    plt.suptitle('initialization: fixed, moving, composed')
    plt.subplot(131), plt.imshow(fixed_im, 'gray', interpolation='nearest'), plt.axis('off')
    plt.subplot(132), plt.imshow(moving_im, 'gray', interpolation='nearest'), plt.axis('off')
    plt.subplot(133), plt.imshow(cimg_im, 'jet', interpolation='nearest'), plt.axis('off')
    # plt.show()


def command_iteration(method, metric, ttype):
    global iteration, prubeh, metric_vals
    #print method.GetOptimizerPosition()
    prubeh.append(list(method.GetOptimizerPosition()))
    if metric == 'mi':
        val = - method.GetMetricValue()
    else:
        val = method.GetMetricValue()
    metric_vals.append(val)

    iteration += 1
    # if (iteration % 10) == 0:
    #     print '%s, %s: it #%i, metric=%.4f' % (metric, ttype, iteration, method.GetMetricValue())


def register(fixed, moving, learning_rate=1., min_step=1e-4, max_iters=500, metric='ms', transform_type='spline',
             show=False, show_now=True):
    R = sitk.ImageRegistrationMethod()
    if metric == 'mi':
        R.SetMetricAsMattesMutualInformation(50)
    elif metric == 'ms':
        R.SetMetricAsMeanSquares()
    else:
        raise ValueError('Wrong metric type, only  \'mi\' and \'ms\' are allowed.')

    # R.SetOptimizerAsRegularStepGradientDescent(learningRate=learning_rate, minStep=min_step, numberOfIterations=max_iters, gradientMagnitudeTolerance=1e-8)
    # R.SetOptimizerAsGradientDescentLineSearch(1.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, maximumNumberOfIterations=max_iters, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000, costFunctionConvergenceFactor=1e+7)

    transfromDomainMeshSize = [8] * moving.GetDimension()
    if transform_type == 'spline':
        tx = sitk.BSplineTransformInitializer(fixed, transfromDomainMeshSize)
    elif transform_type == 'affine':
        tx = sitk.AffineTransform(fixed.GetDimension())
    elif transform_type == 'simil':
        tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
    else:
        raise ValueError('Wrong transform type, only  \'spline\', \'affine\' and \'simil\' are allowed.')

    # tx.SetCenter([0,0])
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)

    # R.SetShrinkFactorsPerLevel([6,2,1])                                                                 # navic
    # R.SetSmoothingSigmasPerLevel([6,2,1])

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R, metric, transform_type))

    outTx = R.Execute(fixed, moving)

    # print("-------")
    # print(outTx)
    # print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    # print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    # print(" Metric value: {0}".format(R.GetMetricValue()))
    #
    # sitk.WriteTransform(outTx, "vystup_BS.txt")

    if "SITK_NOSHOW" not in os.environ:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        # resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)
        out = resampler.Execute(moving)

        simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1 / 2. + simg2 / 2.)

        # array = sitk.GetArrayFromImage(cimg)
        fixed_im = sitk.GetArrayFromImage(simg1)
        moving_im = sitk.GetArrayFromImage(simg2)
        cimg_im = sitk.GetArrayFromImage(cimg)

        if show:
            plt.figure()
            plt.suptitle('%s, %s: fixed, moving, composed' % (transform_type, metric))
            plt.subplot(131), plt.imshow(fixed_im, 'gray', interpolation='nearest'), plt.axis('off')
            plt.subplot(132), plt.imshow(moving_im, 'gray', interpolation='nearest'), plt.axis('off')
            plt.subplot(133), plt.imshow(cimg_im, 'jet', interpolation='nearest'), plt.axis('off')

            plt.figure()
            plt.plot(range(len(metric_vals)), metric_vals, 'b-')
            plt.title('%s, %s' % (transform_type, metric))

            if show_now:
                plt.show()

def get_data_struc():
    dirpath = '/home/tomas/Dropbox/Data/medical/'
    s = 4
    data_struc = [('232_venous-.pklz', '232_arterial-.pklz', range(9, 24, s), -2),
                  ('234_venous-.pklz', '234_arterial-.pklz', range(5, 25, s), 0),
                  ('183b_venous-.pklz', '183b_arterial-.pklz', range(5, 19, s), -1),
                  ('182_venous-.pklz', '182_arterial-.pklz', range(5, 24, s), 0),
                  ('229a_venous-.pklz', '229a_arterial-.pklz', range(5, 23, s), 1),
                  ('233_venous-.pklz', '233_arterial-.pklz', range(12, 29, s), -1),
                  ('228_venous-.pklz', '228_arterial-.pklz', range(5, 20, s), 0),
                  # ('220_venous-.pklz', '220_arterial-.pklz', range(0, 0, s), 0),
                  ('185a_venous-.pklz', '185a_arterial-.pklz', range(6, 24, s), 1),
                  ('221_venous-.pklz', '221_arterial-.pklz', range(4, 30, s), 0),
                  ('227_venous-.pklz', '227_arterial-.pklz', range(4, 26, s), 0),
                  ('183a_venous-.pklz', '183a_arterial-.pklz', range(6, 20, s), -1),
                  ('226_venous-.pklz', '226_arterial-.pklz', range(5, 26, s), -1),
                  ('225_venous-.pklz', '225_arterial-.pklz', range(6, 20, s), 0),
                  ('224_venous-.pklz', '224_arterial-.pklz', range(4, 18, s), 1),
                  ('180_venous-.pklz', '180_arterial-.pklz', range(6, 24, s), 0),
                  ('186a_venous-.pklz', '186a_arterial-.pklz', range(8, 27, s), 0),
                  ('189a_venous-.pklz', '189a_arterial-.pklz', range(6, 25, s), -1),
                  ('223_venous-.pklz', '223_arterial-.pklz', range(4, 16, s), -1),
                  ('235_venous-.pklz', '235_arterial-.pklz', range(5, 40, s), 0),
                  ('222a_venous-.pklz', '222a_arterial-.pklz', range(12, 22, s),-2)]

    data_struc = [(dirpath + x[0], dirpath + x[1], x[2], x[3]) for x in data_struc]

    return data_struc


def find_datapairs(data_dir='/home/tomas/Dropbox/Data/medical/dataset/', ext='pklz'):
    venous_fnames = []
    arterial_fnames = []

    # rozdelim soubory do dvou skupin: obsahujici leze a obsahujici jatra
    for (rootDir, dirNames, filenames) in os.walk(data_dir):
        for filename in filenames:
            if filename.split('.')[-1] == ext and 'leze' not in filename:
                if 'venous' in filename:
                    venous_fnames.append(data_dir + filename)
                elif 'arterial' in filename:
                    arterial_fnames.append(data_dir + filename)
        break

    # dam dohromady prislusne liver a lesion soubory
    venous_aligned = []
    arterial_aligned = []
    for v_fname in venous_fnames:
        a_fname = v_fname.replace('venous', 'arterial')
        if a_fname in arterial_fnames:
            venous_aligned.append(v_fname)
            arterial_aligned.append(a_fname)
        else:
            print 'Cannot find arterial data for \'%s\'' % v_fname

    return venous_aligned, arterial_aligned


def examine_data_pairs():
    venous_fnames, arterial_fnames = find_datapairs()
    tmp = [(x.split('/')[-1], y.split('/')[-1]) for x, y in zip(venous_fnames, arterial_fnames)]
    for i in tmp:
        print i

    for i, (v, a) in enumerate(zip(venous_fnames, arterial_fnames)):
        if i < 10:
            continue
        print i, v.split('/')[-1], a.split('/')[-1]
        datap_1 = tools.load_pickle_data(v, return_datap=True)
        datap_2 = tools.load_pickle_data(a, return_datap=True)

        app = QtGui.QApplication(sys.argv)
        le = SegViewer(datap1=datap_1, datap2=datap_2)
        le.show()
        app.exec_()


def count_stats(vals, metrics, ttypes, show_now=True):
    metric_vals = np.zeros((len(metrics), len(ttypes)))
    metric_cnts = np.zeros((len(metrics), len(ttypes)))
    for i, mtype in enumerate(metrics):
        for j, ttype in enumerate(ttypes):
            if mtype == 'mi':
                vals[mtype][ttype] = [x if x < 1 else 0.99 for x in vals[mtype][ttype]]
                # print (m_vals > 1).sum(),
                # m_vals = np.array([x if x < 1 else 0.99 for x in m_vals])
                # print (m_vals > 1).sum(),
            m_vals = np.array(vals[mtype][ttype])
            # print mtype, ttype, m_vals
            metric_vals[i, j] = m_vals.sum()
            metric_cnts[i, j] = len(m_vals)
    metric_means = metric_vals / metric_cnts

    print '\n\n---------------------------------------------------------'
    print metric_means
    print metric_cnts

    for mtype in metrics:
        data = [vals[mtype][x] for x in ttypes]
        plt.figure(figsize=(5, 7))
        plt.boxplot(data, showfliers=False, showmeans=True, boxprops={'linewidth': 5}, whiskerprops={'linewidth': 3},
                    capprops={'linewidth': 5}, medianprops={'linewidth': 3}, meanprops={'markersize': 8},
                    labels=ttypes, widths=0.5)
        plt.title(mtype)
    if show_now:
        plt.show()


################################################################################
################################################################################
if __name__ == '__main__':
    # data_struc =  get_data_struc()
    # fixed_fnames = []
    # moving_fnames = []
    # slices = []
    # offsets = []
    # for fixed_name, moving_name, num, off in data_struc:
    #     for i in num:
    #         fixed_fnames.append(fixed_name)
    #         moving_fnames.append(moving_name)
    #         slices.append(i)
    #         offsets.append(off)
    # n_imgs = len(fixed_fnames)
    #
    # max_num = 60
    # metrics = ['ms', 'mi']
    # ttypes = ['simil', 'affine', 'spline']
    # metric_all_vals = {x: {y: [] for y in ttypes} for x in metrics}
    # for i, (fixed_fn, moving_fn, slice, offset) in enumerate(zip(fixed_fnames, moving_fnames, slices, offsets)):
    #     print '%i/%i ...' % (i + 1, max_num),
    #     data_v, gt_mask, voxel_size = tools.load_pickle_data(fixed_fn)
    #     data_v = data_v[slice,...]
    #     data_v = tools.windowing(data_v)
    #
    #     data_a, gt_mask, voxel_size = tools.load_pickle_data(moving_fn)
    #     data_a = data_a[slice, ...]
    #     data_a = tools.windowing(data_a)
    #
    #     fixed_arr = data_v.astype(np.float)
    #     moving_arr = tools.resize_ND(data_a, shape=data_v.shape).astype(fixed_arr.dtype)
    #
    #     for mtype in metrics:
    #         for ttype in ttypes:
    #             # print '\n    ------    \n%s  --  %s' % (mtype, ttype)
    #             print '(%s, %s)' % (mtype, ttype),
    #             fixed = sitk.GetImageFromArray(fixed_arr.copy())
    #             moving = sitk.GetImageFromArray(moving_arr.copy())
    #             try:
    #                 register(fixed, moving, transform_type=ttype, metric=mtype, max_iters=300, show=False)
    #                 metric_all_vals[mtype][ttype].append(metric_vals[-1])
    #             except:
    #                 pass
    #             iteration = 0
    #             prubeh = []
    #             metric_vals = []
    #     print 'done'
    #
    #     if i == max_num - 1:
    #         break
    #
    # f = gzip.open('metrics.pklz', 'wb', compresslevel=1)
    # pickle.dump(metric_all_vals, f)
    # f.close()
    # count_stats(metric_all_vals, metrics, ttypes, show_now=False)

    # f = gzip.open('metrics.pklz', 'rb')
    # fcontent = f.read()
    # metric_all_vals = pickle.loads(fcontent)
    # count_stats(metric_all_vals, metrics, ttypes, show_now=True)

    # --------------------------------------------------------------------------
    mtype = 'ms'
    # mtype = 'mi'
    ttype = 'simil'
    fixed_arr = cv2.imread('/home/tomas/Dropbox/Work/Dizertace/figures/liver_registration/fixed_img.png', 0)
    moving_arr = cv2.imread('/home/tomas/Dropbox/Work/Dizertace/figures/liver_registration/moving_img.png', 0)
    fixed = sitk.GetImageFromArray(fixed_arr.astype(np.float))
    moving = sitk.GetImageFromArray(moving_arr.astype(np.float))

    register(fixed, moving, transform_type=ttype, metric=mtype, max_iters=300, show=True)