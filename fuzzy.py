from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

import sys
import os
import datetime

if os.path.exists('../imtools/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '../imtools/')
    from imtools import tools, misc
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


verbose = True


def set_verbose(val):
    global verbose
    verbose = val


def _debug(msg, msgType="[INFO]"):
    if verbose:
        print '{} {} | {}'.format(msgType, msg, datetime.datetime.now())


def fcm(img, mask=None, n_clusters=2, show=False, show_now=True, verbose=True):
    set_verbose(verbose)

    if img.ndim == 2:
        is_2D = True
    else:
        is_2D = False

    if mask is None:
        mask = np.ones_like(img)

    coords = np.nonzero(mask)
    data = img[coords]
    # alldata = np.vstack((coords[0], coords[1], data))
    alldata = data.reshape((1, data.shape[0]))

    _debug('Computing FCM...')
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, n_clusters, 2, error=0.005, maxiter=1000, init=None)

    # sorting the results in ascending order
    idx = np.argsort(cntr[:, 0])
    cntr = cntr[idx, :]
    u = u[idx, :]

    cm = np.argmax(u, axis=0) + 1  # cluster membership
    labels = np.zeros(img.shape)
    labels[coords] = cm

    # calculating cluster memberships
    mems = np.zeros(((n_clusters,) + img.shape))
    for i in range(n_clusters):
        cp = u[i, :] + 1  # cluster membership
        mem = np.zeros(img.shape)
        mem[coords] = cp
        mems[i,...] = mem

    if show:
        if is_2D:
            plt.figure()
            plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('input')
            plt.subplot(132), plt.imshow(labels, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('fcm')
            plt.subplot(133), plt.imshow(labels, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('fcm')
        # else:
            # py3DSeedEditor.py3DSeedEditor(labels).show()

            u0 = np.zeros(img.shape)
            u0[coords] = u[0]
            u1 = np.zeros(img.shape)
            u1[coords] = u[1]
            d0 = np.zeros(img.shape)
            d0[coords] = d[0]
            d1 = np.zeros(img.shape)
            d1[coords] = d[1]
            plt.figure()
            plt.subplot(231), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input')
            plt.subplot(232), plt.imshow(u0, 'gray', interpolation='nearest'), plt.title('u0 - partititon matrix'), plt.colorbar()
            plt.subplot(233), plt.imshow(u1, 'gray', interpolation='nearest'), plt.title('u1 - partititon matrix'), plt.colorbar()
            plt.subplot(235), plt.imshow(d0, 'gray', interpolation='nearest'), plt.title('d0 - dist matrix'), plt.colorbar()
            plt.subplot(236), plt.imshow(d1, 'gray', interpolation='nearest'), plt.title('d1 - dist matrix'), plt.colorbar()


        if show_now:
            plt.show()

    return mems, u, d, cntr, fpc


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    show = True
    pyr_scale = 1.5
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'  # slice_id = 17
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_185_48441644_venous_5.0_B30f-.pklz'  # slice_id = 14
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_186_49290986_venous_5.0_B30f-.pklz'  # slice_id = 5
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    # tools.show_3d(tools.windowing(data))

    slice_ind = 17
    data_s = data[slice_ind, :, :]
    data_s = tools.windowing(data_s)
    mask_s = mask[slice_ind, :, :]

    fcm(data_s, mask=mask_s, n_clusters=2, show=True)