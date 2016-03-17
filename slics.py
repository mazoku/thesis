from __future__ import division

#TODO:
    # Zaplneni der po odstraneni malych objektu vis http://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array

import os
import numpy as np
import skimage.segmentation as skiseg
import skimage.morphology as skimor
import skimage.color as skicol
import cv2

import matplotlib.pyplot as plt

# import tools
import os
import sys
if os.path.exists('../imtools/'):
    sys.path.append('../imtools/')
    from imtools import tools
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

import progressbar


# def run_2D(data, mask, return_vis=False):
#     data_w = tools.windowing(data).astype(np.uint8)
#     data_s = tools.smoothing(data_w) * (mask > 0)
#     data_rgb = cv2.cvtColor(data_s, cv2.COLOR_GRAY2RGB)
#
#     slics = skiseg.slic(data_rgb, multichannel=True, n_segments=10)
#
#     slics_big = np.zeros_like(slics)
#     areaT = 600
#     for i in np.unique(slics):
#         slic = skimor.binary_opening(slics == i, skimor.disk(3))
#         area = slic.sum()
#         if area > areaT:
#             slics_big = np.where(slic, slics, slics_big)
#
#     # tools.show_3d((data, slics, slics_big))
#     if not return_vis:
#         return slics_big
#
#     vis = [('input', data_w), ('slic', slics), ('thresholded slic', slics_big),
#            ('boundaries', skiseg.mark_boundaries(data_w, slics_big, color=(1, 0, 1)))]
#
#     # ranges for visualization
#     ranges = [(0, 255), (slics.min(), slics.max()), (slics_big.min(), slics_big.max()), (0, 255)]
#
#     # colormaps
#     cmaps = ['gray', 'gray', 'jet', 'gray']
#
#     return slics_big, vis, ranges, cmaps


def run(data, mask, voxel_size, is_2D=False, bboxing=False, return_vis=False, show=False, show_now=True, save_fig=False):
    data_w = tools.windowing(data).astype(np.uint8)
    data_s = tools.smoothing(data_w, sliceId=0) * (mask > 0)
    data_rgb = np.zeros((data_s.shape[1], data_s.shape[2], 3, data_s.shape[0]))
    for i, s in enumerate(data_s):
        data_rgb[:, :, :, i] = cv2.cvtColor(s, cv2.COLOR_BAYER_GR2RGB)
    data_rgb = np.swapaxes(data_rgb, 2, 3)

    slics = skiseg.slic(data_rgb, spacing=np.roll(voxel_size, 2), multichannel=True, n_segments=10)
    # slics = skiseg.slic(data_rgb, spacing=np.roll(voxel_size, 2), multichannel=True, n_segments=100,
    #                     slic_zero=True, enforce_connectivity=True)
    slics = np.swapaxes(np.swapaxes(slics, 1, 2), 0, 1) * (mask > 0)

    # tools.show_3d(slics)

    slics_big = np.zeros_like(slics)
    if is_2D:
        areaT = 500
    else:
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
    for i, (im, slic, slic_b, mask_s) in enumerate(zip(data_w, slics, slics_big, mask)):
        vis_s = []
        if bboxing:
            im, mask_bb = tools.crop_to_bbox(im, mask_s)
            slic, mask_bb = tools.crop_to_bbox(slic, mask_s)
            slic_b, mask_bb = tools.crop_to_bbox(slic_b, mask_s)
        vis_s.append(('input', im))
        vis_s.append(('slic', slic))
        vis_s.append(('thresholded slic', slic_b))
        vis_s.append(('boundaries', skiseg.mark_boundaries(im, slic_b, color=(1, 0, 1))))
        vis.append(vis_s)

    # ranges for visualization
    ranges = [(0, 255), (slics.min(), slics.max()), (slics_big.min(), slics_big.max()), (0, 255)]

    # colormaps
    cmaps = ['gray', 'gray', 'jet', 'gray']

    if show or save_fig:
        n_imgs = len(vis[0])
        # pbar = progressbar.ProgressBar(maxval=len(vis), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        # widgets = ["Seting up figures: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        # pbar = progressbar.ProgressBar(maxval=len(vis), widgets=widgets)
        # pbar.start()
        for slice_id, (v, mask_s) in enumerate(zip(vis, mask)):
            fig = plt.figure(figsize=(24, 14))
            for i, ((tit, im), rg, cm) in enumerate(zip(v, ranges, cmaps)):
                # tit = v[slice_id][i][0]
                # im = v[slice_id][i][1]
                # im_bb, mask_bb = tools.crop_to_bbox(im, mask_s)
                plt.subplot(1, n_imgs, i + 1)
                plt.imshow(im, cmap=cm, vmin=rg[0], vmax=rg[1], interpolation='nearest')
                plt.title(tit)

            if save_fig:
                fig_dir = '/home/tomas/Dropbox/Work/Dizertace/figures/slics/'
                dirs = fig_dir.split('/')
                for i in range(2, len(dirs)):
                    subdir = '/'.join(dirs[:i])
                    if not os.path.exists(subdir):
                        os.mkdir(subdir)
                if is_2D:
                    dirname = os.path.join(fig_dir, 'slics_vis_2D_slice_%i.png' % (slice_id + 1))
                else:
                    dirname = os.path.join(fig_dir, 'slics_vis_slice_%i.png' % (slice_id + 1))
                fig.savefig(dirname, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close('all')
            # pbar.update(slice_id)
        # pbar.finish()
        if show_now:
            plt.show()

    return slics_big, vis, ranges, cmaps


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation//org-exp_183_46324212_arterial_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_180_49509315_venous_5.0_B30f-.pklz'
    config_fname = 'config.ini'

    show = True
    show_now = False
    save_fig = False

    is_2D = False

    data, mask, voxel_size = tools.load_pickle_data(data_fname)
    slice_id = 17
    data_s = data[slice_id, :, :]
    mask_s = mask[slice_id, :, :]
    data = np.expand_dims(data_s, axis=0)
    mask = np.expand_dims(mask_s, axis=0)
    is_2D = True

    print 'Processing file -- %s' % data_fname[data_fname.rfind('/') + 1:]
    slics_big, vis, ranges, cmaps = run(data, mask, voxel_size, return_vis=True, is_2D=is_2D, show=show, show_now=show_now, save_fig=save_fig)

    # n_imgs = len(vis[0])
    # for slice, v in enumerate(vis):
    #     fig = plt.figure(figsize=(24, 14))
    #     for i, ((tit, im), rg, cm) in enumerate(zip(v, ranges, cmaps)):
    #         # tit = v[slice_id][i][0]
    #         # im = v[slice_id][i][1]
    #         plt.subplot(1, n_imgs, i + 1)
    #         plt.imshow(im, cmap=cm, vmin=rg[0], vmax=rg[1], interpolation='nearest')
    #         plt.title(tit)
    plt.show()