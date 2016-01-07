from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as matpat

import skimage.exposure as skiexp
import skimage.morphology as skimor
import skimage.measure as skimea

# import tools
import os
import sys
if os.path.exists('../imtools/'):
    sys.path.append('../imtools/')
    from imtools import tools
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


# def save_figures(data_fname, subdir, data, mask, imgs):
#     dirs = data_fname.split('/')
#     dir = dirs[-1]
#     patient_id = dir[8:11]
#     if 'venous' in dir or 'ven' in dir:
#         phase = 'venous'
#     elif 'arterial' in dir or 'art' in dir:
#         phase = 'arterial'
#     else:
#         phase = 'phase_unknown'
#
#     fig_dir = os.path.join(os.sep.join(dirs[:-1]), subdir)
#     if not os.path.exists(fig_dir):
#         os.mkdir(fig_dir)
#     fig_patient_dir = os.path.join(fig_dir, 'figs_%s_%s' % (patient_id, phase))
#     if not os.path.exists(fig_patient_dir):
#         os.mkdir(fig_patient_dir)
#
#     # fig_fname = os.path.join(fig_dir, 'figs_%s_%s' % (patient_id, phase))
#
#     fig = plt.figure(figsize=(20, 5))
#     for s in range(len(imgs)):
#         if imgs[s] is None:
#             continue
#         # print 'Saving figure #%i/%i ...' % (s + 1, len(res)),
#         print tools.get_status_text('\tSaving figures', iter=s, max_iter=len(imgs)),
#         for i, (title, im) in enumerate(imgs[s]):
#             plt.subplot(1, len(imgs[s]), i + 1)
#             if title == 'contours':
#                 plt.imshow(imgs[s][0][1], 'gray', interpolation='nearest'), plt.title(title)
#                 plt.hold(True)
#                 for n, contour in enumerate(im):
#                     plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
#                 plt.axis('image')
#             else:
#                 if i == 0:
#                     coords = np.nonzero(mask[s, :, :])
#                     r_min = max(0, min(coords[0]))
#                     r_max = min(mask.shape[1], max(coords[0]))
#                     c_min = max(0, min(coords[1]))
#                     c_max = min(mask.shape[2], max(coords[1]))
#                     plt.imshow(data[s,:,:], 'gray', interpolation='nearest', vmin=50 - 175, vmax=50 + 175), plt.title(title)
#                     plt.hold(True)
#                     bbox = matpat.Rectangle((c_min, r_min), (c_max - c_min + 1), (r_max - r_min + 1), color='red', fill=False, lw=2)
#                     plt.gca().add_patch(bbox)
#                     plt.axis('image')
#                 else:
#                     plt.imshow(im, 'gray', interpolation='nearest'), plt.title(title)
#
#         fig.savefig(os.path.join(fig_patient_dir, 'slice_%i.png' % s))
#         fig.clf()
#         # print 'done'
#     print tools.get_status_text('\tSaving figures', iter=-1, max_iter=len(imgs))


def process_slice(img_o, mask_o, proc, disp=False, show=False, t=0.2):
    if not mask_o.any():
        return np.zeros_like(img_o), None
    n_rows, n_cols = [1, len(proc) + 1]

    img_bbox, mask = tools.crop_to_bbox(img_o, mask_o)

    # write mean value outside the mask
    mean_val = np.mean(img_bbox[np.nonzero(mask)])
    img = np.where(mask, img_bbox, mean_val)

    # rescale to UINT8
    img = skiexp.rescale_intensity(img, in_range=(50 - 175, 50 + 175), out_range=np.uint8).astype(np.uint8)

    vis = []
    img_pipe = img.copy()
    seg = None  # final segmentation

    # saving input and input with liver borders
    vis.append(('input', img_o))
    liver_contours = skimea.find_contours(mask_o, 0.9)
    vis.append(('liver contours', (img_o, liver_contours)))

    # smoothing
    if 'smo' in proc or 'smoothing' in proc:
        img_s = tools.smoothing(img)
        img_pipe = img_s.copy()
        # img_pipe_o = tools.put_bbox_back(np.zeros_like(img_pipe_o), img_pipe, mask=mask_o)
        vis.append(('smoothing', img_s * mask))

    # # write mean value outside the mask
    # mean_val = np.mean(img_pipe[np.nonzero(mask)])
    # img_pipe = np.where(mask, img_pipe, mean_val)

    # Equalization
    if 'equ' in proc or 'equalization' in proc:
        img_h = skiexp.equalize_hist(img_pipe)
        img_pipe = img_h.copy()
        # img_pipe_o = tools.put_bbox_back(np.zeros_like(img_pipe_o), img_pipe, mask=mask_o)
        vis.append(('equalization', img_h * mask))

    # canny edge detector
    # canny = tools.auto_canny(skiexp.rescale_intensity(img_eq, out_range=np.uint8).astype(np.uint8), sigma=0.1)

    # contrast stretch
    if 'str' in proc or 'stretch' in proc:
        p = (20, 40)
        pmin, pmax = np.percentile(img_pipe, p)
        img_rescale = skiexp.rescale_intensity(img_pipe, in_range=(pmin, pmax))
        img_pipe = img_rescale.copy()
        vis.append(('stretch {}'.format(p), img_rescale * mask))

    # morphology
    if 'clo' in proc or 'closing' in proc:
        r = 7
        img_morph = skimor.closing(img_pipe, selem=skimor.disk(r))
        img_pipe = img_morph.copy()
        seg = tools.put_bbox_back(np.zeros(img_o.shape), img_pipe < t, mask=mask_o)
        vis.append(('closing r=%i'%r, img_morph * mask))

    # contours
    if 'con' in proc or 'contours' in proc:
        contours = skimea.find_contours(img_pipe, t)
        vis.append(('contours', (img_bbox, contours)))

    # plt.figure()
    # plt.subplot(121), plt.imshow(img_morph, 'gray'), plt.colorbar()
    # plt.subplot(122), plt.imshow(seg, 'gray'), plt.colorbar()
    # plt.show()

    if disp:
        plt.figure()
        for i, (title, im) in enumerate(vis):
            plt.subplot(n_rows, n_cols, i + 1)
            if title == 'contours':
                plt.imshow(vis[0][1], 'gray', interpolation='nearest'), plt.title(title)
                plt.hold(True)
                for n, contour in enumerate(im):
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.axis('image')
            else:
                plt.imshow(im, 'gray', interpolation='nearest'), plt.title(title)

        # if 'con' in proc or 'contours' in proc:
        #     for n, contour in enumerate(contours):
        #         plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #     plt.axis('image')

        if show:
            plt.show()

    return seg, vis


def run(data, mask, proc=['smoo', 'equa', 'clos', 'cont'], disp=False, show=False):
    # data, mask = tools.crop_to_bbox(data, mask)
    segs = []
    vis = []

    for i in range(data.shape[0]):
        # print 'Processing slice #%i/%i' % (i + 1, data.shape[0]),
        print tools.get_status_text('\tProcessing slices', iter=i, max_iter=data.shape[0]),
        seg_s, vis_s = process_slice(data[i, :, :], mask[i, :, :], proc=['smo', 'equ', 'clo', 'con'], disp=disp, show=show)
        segs.append(seg_s)
        vis.append(vis_s)
        # print 'done'
    print tools.get_status_text('\tProcessing slice', iter=-1, max_iter=data.shape[0])

    return np.array(segs), vis


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation//org-exp_183_46324212_arterial_5.0_B30f-.pklz'
    # data_fname = '/home/tomas/Data/liver_segmentation/org-exp_180_49509315_venous_5.0_B30f-.pklz'
    config_fname = 'config.ini'

    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    print 'Processing file -- %s' % data_fname[data_fname.rfind('/') + 1:]
    run(data, mask, proc=['smoo', 'equa', 'clos', 'cont'], disp=True, show=True)