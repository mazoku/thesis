from __future__ import division

import os
import numpy as np
import skimage.segmentation as skiseg
import skimage.morphology as skimor
import cv2

import tools


def run(data, mask, voxel_size, return_vis=False):
    data_w = tools.windowing(data).astype(np.uint8)
    data_s = tools.smoothing(data_w, sliceId=0) * (mask > 0)
    data_rgb = np.zeros((data_s.shape[1], data_s.shape[2], 3, data_s.shape[0]))
    for i, s in enumerate(data_s):
        data_rgb[:, :, :, i] = cv2.cvtColor(s, cv2.COLOR_BAYER_GR2RGB)
    data_rgb = np.swapaxes(data_rgb, 2, 3)

    slics = skiseg.slic(data_rgb, spacing=np.roll(voxel_size, 2),
                        multichannel=True, n_segments=10)
    slics = np.swapaxes(np.swapaxes(slics, 1, 2), 0, 1) * (mask > 0)

    slics_big = np.zeros_like(slics)

    areaT = 6000
    for i in np.unique(slics):
        slic = tools.opening3D(slics == i, selem=skimor.disk(3))
        area = slic.sum()
        if area > areaT:
            slics_big = np.where(slic, slics, slics_big)

    # tools.show_3d((data, slics, slics_big))
    if not return_vis:
        return slics_big

    vis = []
    for i, (im, slic, slic_b) in enumerate(zip(data_w, slics, slics_big)):
        vis_s = []
        vis_s.append(('input', im))
        vis_s.append(('slic', slic))
        vis_s.append(('thresholded slic', slic_b))
        vis_s.append(('boundaries', skiseg.mark_boundaries(im, slic_b, color=(1, 0, 1))))
        vis.append(vis_s)

    # ranges for visualization
    ranges = []
    ranges.append((0, 255))
    ranges.append((slics.min(), slics.max()))
    ranges.append((slics_big.min(), slics_big.max()))
    ranges.append((0, 255))

    # colormaps
    cmaps = []
    cmaps.append('gray')
    cmaps.append('gray')
    cmaps.append('jet')
    cmaps.append('gray')

    return slics_big, vis, ranges, cmaps


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation//org-exp_183_46324212_arterial_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_180_49509315_venous_5.0_B30f-.pklz'
    config_fname = 'config.ini'

    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    print 'Processing file -- %s' % data_fname[data_fname.rfind('/') + 1:]
    run(data, mask, voxel_size)