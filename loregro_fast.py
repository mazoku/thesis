from __future__ import division

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.morphology as skimor
import skimage.color as skicol
import scipy.ndimage.morphology as scindimor

if os.path.exists('/home/tomas/projects/imtools/'):
    sys.path.insert(0, '/home/tomas/projects/imtools/')
    from imtools import tools, misc
else:
    print 'Import error in active_contours.py. You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)


class LoReGro:
    def __init__(self, im, seeds, im_path=None, energy=None, ball_r_out=1, ball_r=3, min_diff=0.1, alpha=1.,
                 scale=0.5, max_iters=1000, smoothing=True, show=False, show_now=True):
        if isinstance(im, str):
            self.im_path = im
            self.im = cv2.imread(self.im_path, 0)
        else:
            self.im = im
        self.seeds = seeds
        if energy is not None:
            self.energy = energy
        else:
            self.energy = self.im.copy()

        if self.im.ndim == 2:
            self.im = np.expand_dims(self.im, 0)
        if self.energy.ndim == 2:
            self.energy = np.expand_dims(self.energy, 0)
        if self.seeds.ndim == 2:
            self.seeds= np.expand_dims(self.seeds, 0)

        if (self.im.shape != self.energy.shape) and (self.im.shape != self.seeds.shape):
            raise IOError('Shape of image, energy and seeds didn\'t match.')
        # n_slices, n_rows, n_cols = self.im.shape
        self.n_slices = self.im.shape[0]
        self.segmentation = self.seeds.copy()
        self.newbies = None  # nove pridane pixely

        self.scale = scale
        self.orig_shape = self.im.shape
        self.ndim = self.im.ndim
        self.ball_r_out = ball_r_out
        self.ball_r = ball_r
        self.min_diff = min_diff
        self.alpha = alpha
        self.max_iters = max_iters
        self.smoothing = smoothing
        self.show = show
        self.show_now = show_now

    def run_segmentation(self, display=False):

        # mask = self.data_term(find_all=False)
        # mask = self.seeds.copy()

        # strel = skimor.disk(2)

        # ballM = np.zeros((2 * self.ball_r + 1, 2 * self.ball_r + 1, 2), dtype=np.int)
        # ballM[:, :, 1] = np.tile(np.arange(-self.ball_r, self.ball_r + 1), [2 * self.ball_r + 1, 1])
        # ballM[:, :, 0] = ballM[:, :, 1].conj().transpose()
        #
        # ballMOut = np.zeros((2 * self.ball_r_out + 1, 2 * self.ball_r_out + 1, 2), dtype=np.int)
        # ballMOut[:, :, 1] = np.tile(np.arange(-self.ball_r_out, self.ball_r_out + 1), [2 * self.ball_r_out + 1, 1])
        # ballMOut[:, :, 0] = ballMOut[:, :, 1].conj().transpose()

        # self.segmentation = self.segmentInnerBoundary(self.energy, mask, ballM, strel, self.min_diff, self.alpha, display)
        # self.membraneMask = skimor.binary_closing(self.membraneMask, strel)

        if self.smoothing:
            self.im = tools.smoothing(self.im, sigmaSpace=5, sigmaColor=5, sliceId=0)

        if scale != 1:
            self.im = tools.resize3D(self.im, self.scale, sliceId=0)
            self.seeds = tools.resize3D(self.seeds, self.scale, sliceId=0)
            self.segmentation = tools.resize3D(self.segmentation, self.scale, sliceId=0)

        self.shape = self.im.shape

        print 'Creating nghb matrix ...',
        self.nghbm = self.create_balls_nghbm()
        print 'done.'

        newbies = self.seeds.copy()
        curr_it = 0
        changed = True
        while changed and curr_it < self.max_iters:
            curr_it += 1
            print 'iteration #%i/%i' % (curr_it, self.max_iters)
            #            mask, accepted, refused = self.iterationIB( im, mask, ballM, strel, minDiff, alpha )
            mask_new, accepted, refused = self.iteration_IB(newbies)
            # plt.figure()
            # plt.subplot(121), plt.imshow(self.segmentation[0,...].copy(), 'gray')
            # plt.subplot(122), plt.imshow(mask_new[0,...].copy(), 'gray')
            # if (curr_it % 2) == 0:
            #     plt.show()
            # newbies = mask_new - self.segmentation
            # mask = mask_new
            self.segmentation = mask_new
            #            self.newbies = np.zeros( self.newbies.shape, dtype=np.bool )
            #            self.newbies[accepted[0],accepted[1]] = True

            if not accepted.any():
                changed = False

        for i, im in enumerate(self.segmentation):
            self.segmentation[i,...] = skimor.binary_closing(im, np.ones((3, 3)))
            # self.segmentation[i,...] = scindimor.binary_closing(self.segmentation, np.ones((3, 3, 3)))

        # rescale the segmentation to original shape
        if scale != 1:
            tmp = np.zeros(self.orig_shape)
            for i, im in enumerate(self.segmentation):
                tmp[i,...] = cv2.resize(im.astype(np.uint8), (self.orig_shape[2], self.orig_shape[1]))
            self.segmentation = tmp

    def create_balls_nghbm(self):

        # creating ball mask
        strel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        tmp = np.zeros((2 * self.ball_r + 1,) * self.ndim)
        tmp[(self.ball_r,) * self.ndim] = 1
        ball_im = scindimor.binary_dilation(tmp, strel, self.ball_r - 1)
        n_ball_pts = ball_im.sum()

        coords = np.nonzero(ball_im)
        center = (self.ball_r,) * self.ndim
        # move the coords so that the center lies in 0
        coords = tuple(coords[i] - center[i] for i in range(self.ndim))
        coords = np.array(coords).T  # coords in array [npts, dim]

        # creating nghb matrix [n_pts, n_nghbs = # of points in the ball]
        n_pts = np.prod(self.shape)
        neighbors_m = np.zeros((n_pts, n_ball_pts))
        for i in range(n_pts):
            s, r, c = np.unravel_index(i, self.shape)
            for j in range(n_ball_pts):
                s_ball, r_ball, c_ball = coords[j,:]
                sn = s + s_ball
                rn = r + r_ball
                cn = c + c_ball

                slice_ko = sn < 0 or sn > (self.shape[0] - 1)
                row_ko = rn < 0 or rn > (self.shape[1] - 1)
                col_ko = cn < 0 or cn > (self.shape[2] - 1)
                if row_ko or col_ko or slice_ko:
                    neighbors_m[i, j] = np.NaN
                else:
                    index = np.ravel_multi_index((sn, rn, cn), self.shape)
                    neighbors_m[i, j] = index

        return neighbors_m

    def iteration_IB(self, mask):
        dilm = np.zeros_like(mask)
        for i, im in enumerate(mask):
            dilm[i,...] = skimor.binary_dilation(im, skimor.rectangle(3, 3))
        newbies = dilm - mask
        newbiesL = np.argwhere(newbies)  # .tolist()

        accepted = list()
        refused = list()

        #        labels = mt.fuseImages( (mask, newbies2), 'wg', type = 'imgs')
        #        plt.figure()
        #        plt.imshow( im ), plt.hold( True )
        #        plt.imshow( labels )
        #        plt.show()

        # maskV = np.argwhere(self.segmentation)
        # maskV = im[maskV[:, 0], maskV[:, 1]]
        mean_mask = self.im[np.nonzero(mask)].mean()

        for i in range(newbiesL.shape[0]):
            newbie = tuple(newbiesL[i])
            inners, outers = self.mask_newbie(newbie, self.segmentation)

            dist_in, dist_out = self.get_distances(newbie, inners, outers)
            dist_mask = np.absolute(mean_mask - self.im[newbie])

            weight_dist_in = self.alpha * dist_in + (1 - self.alpha) * dist_mask

            if weight_dist_in < dist_out or np.absolute(dist_in - dist_out) < self.min_diff:
                mask[newbie] = True
                accepted.append(newbie)
            else:
                refused.append(newbie)

        accepted = np.array(accepted)
        refused = np.array(refused)

        # acc = (accepted[:, 0], accepted[:, 1], accepted[:, 2])
        # ref = (refused[:, 0], refused[:, 1], refused[:, 2])
        # im_vis = self.segmentation.copy()
        # im_vis[acc] = 2
        # im_vis[ref] = 3
        # im_vis = skicol.label2rgb(im_vis[0,...], colors=((0, 0, 255), (0, 255, 0), (255, 0, 0)), bg_label=0)
        # plt.figure()
        # plt.imshow(im_vis.astype(np.uint8))
        # plt.show()

        return mask, accepted, refused

    def mask_newbie(self, newbie, mask):
        ball_pts = [int(x) for x in self.nghbm[np.ravel_multi_index(newbie, self.shape), :] if not np.isnan(x)]
        # inners_ind = [x for x in ball_pts if mask[np.unravel_index(x, self.shape)]]
        inners_ind = []
        outers_ind = []
        for i in ball_pts:
            if mask[np.unravel_index(i, self.shape)]:
                inners_ind.append(i)
            else:
                outers_ind.append(i)

        inners = np.unravel_index(inners_ind, self.shape)
        outers = np.unravel_index(outers_ind, self.shape)

        # tmp = np.zeros_like(mask)
        # tmp[newbie] = 1
        # strel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        #                   [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        #                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        # ball_im = scindimor.binary_dilation(tmp, strel, self.ball_r - 1)
        # # ball_im = cv2.circle(tmp, (newbie[0], newbie[2], newbie[1]), self.ball_r, 255, -1)
        #
        # inners = np.nonzero(ball_im * mask)
        # outers = np.nonzero(ball_im * np.logical_not(mask))

        return inners, outers

    def get_distances(self, newbie, inners, outers):
        # ints_in = self.im[inners[:, 0], inners[:, 1]]
        # ints_out = self.im[outers[:, 0], outers[:, 1]]
        # int_new = self.im[newbie[0], newbie[1]]
        ints_in = self.im[inners]
        ints_out = self.im[outers]
        int_new = self.im[newbie[0], newbie[1], newbie[2]]

        mean_in = ints_in.mean()
        mean_out = ints_out.mean()

        dist_in = np.absolute(mean_in - int_new)
        dist_out = np.absolute(mean_out - int_new)

        return dist_in, dist_out


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_fname = '/home/tomas/Data/medical/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    data, mask, voxel_size = tools.load_pickle_data(data_fname)

    # 2D
    # slice_ind = 17
    slice_ind = 15
    # data = data[slice_ind, :, :]
    data = tools.windowing(data)

    ball_r_out = 1
    ball_r = 5
    min_diff = 1
    alpha = 0.9
    scale = 0.25
    max_iters = 100
    smoothing = True
    show = True
    show_now = False

    _, seeds = tools.initialize_graycom(data, slice=slice_ind, show=show, show_now=show_now)
    # seeds = tools.eroding3D(seeds, np.ones((3, 3))).astype(np.uint8)
    seeds = tools.morph_ND(seeds, 'erosion', selem=np.ones((3, 3)), slicewise=True, sliceId=0).astype(np.uint8)

    plt.figure()
    plt.subplot(121), plt.imshow(data[slice_ind,...], 'gray')
    plt.subplot(122), plt.imshow(seeds[slice_ind,...], 'jet')
    # plt.show()

    lrg = LoReGro(data, seeds, ball_r_out=ball_r_out, ball_r=ball_r, min_diff=min_diff, alpha=alpha,
                  scale=scale, max_iters=max_iters, smoothing=smoothing, show=show, show_now=show_now)
    lrg.run_segmentation()

    seg = lrg.segmentation

    # plt.figure()
    # plt.suptitle('LoReGro segmentation')
    # plt.subplot(131), plt.imshow(data, 'gray'), plt.title('input')
    # plt.subplot(132), plt.imshow(seeds, 'jet'), plt.title('seeds')
    # plt.subplot(133), plt.imshow(seg[0,...].astype(np.uint8), 'gray'), plt.title('segmentation')
    # plt.show()
    # tools.visualize_seg(data, seg[0,...], mask=seeds, title='loregro')
    tools.visualize_seg(data, seg, mask=seeds, slice=slice_ind, title='loregro')