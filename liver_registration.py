# from __future__ import division

import SimpleITK as sitk
from skimage import data
import skimage.transform as skitra

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import time

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in active_contours.py. You need to import package imtools: https://github.com/mjirik/imtools'
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


def command_iteration(method):
    global iteration, prubeh, metric_vals
    #print method.GetOptimizerPosition()
    prubeh.append(list(method.GetOptimizerPosition()))
    metric_vals.append(method.GetMetricValue())

    iteration += 1
    if (iteration % 10) == 0:
        print 'it #%i, metric=%.4f' % (iteration, method.GetMetricValue())


def register(fixed, moving, learning_rate=1., min_step=1e-4, max_iters=500):
    R = sitk.ImageRegistrationMethod()
    # R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricAsMeanSquares()

    #    R.SetOptimizerAsGradientDescent()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=learning_rate, minStep=min_step, numberOfIterations=max_iters, gradientMagnitudeTolerance=1e-8)
    # R.SetOptimizerAsGradientDescentLineSearch(1.0, 100,convergenceMinimumValue=1e-4,convergenceWindowSize=5)
    # R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,maximumNumberOfIterations=100,maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,costFunctionConvergenceFactor=1e+7)

    transfromDomainMeshSize = [8] * moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transfromDomainMeshSize)
    # tx = sitk.AffineTransform(ctverec.GetDimension())
    # tx.SetCenter([0,0])
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)

    # R.SetShrinkFactorsPerLevel([6,2,1])                                                                 # navic
    # R.SetSmoothingSigmasPerLevel([6,2,1])

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    sitk.WriteTransform(outTx, "vystup_BS.txt")

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

        plt.figure()
        plt.suptitle('results: fixed, moving, composed')
        plt.subplot(131), plt.imshow(fixed_im, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(132), plt.imshow(moving_im, 'gray', interpolation='nearest'), plt.axis('off')
        plt.subplot(133), plt.imshow(cimg_im, 'jet', interpolation='nearest'), plt.axis('off')

        plt.figure()
        plt.plot(range(len(metric_vals)), metric_vals, 'b-')

        plt.show()


################################################################################
################################################################################
if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/Data/medical/183a_venous-.pklz'
    data_v, gt_mask, voxel_size = tools.load_pickle_data(fname)
    data_v = data_v[17,...]
    data_v = tools.windowing(data_v)

    fname = '/home/tomas/Dropbox/Data/medical/183a_arterial-.pklz'
    data_a, gt_mask, voxel_size = tools.load_pickle_data(fname)
    data_a = data_a[15, ...]
    data_a = tools.windowing(data_a)

    # imgs_v = [tools.windowing(data_v[16,...]), tools.windowing(data_v[17,...]), tools.windowing(data_v[18,...])]
    # imgs_a = [tools.windowing(data_a[14,...]), tools.windowing(data_a[15,...]), tools.windowing(data_a[16,...])]
    # plt.figure()
    # for i, im in enumerate(imgs_v + imgs_a):
    #     plt.subplot(231 + i), plt.imshow(im, 'gray', interpolation='nearest')
    # plt.show()

    # fixed = sitk.ReadImage('ctverec.png', sitk.sitkFloat32)
    # moving = sitk.ReadImage('ctverec_T.png', sitk.sitkFloat32)
    fixed = data_v.astype(np.float)
    moving = tools.resize_ND(data_a, shape=data_v.shape).astype(fixed.dtype)
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)
    print fixed.GetSize(), moving.GetSize()

    show_initialization(fixed, moving)
    register(fixed, moving, learning_rate=1., max_iters=100)