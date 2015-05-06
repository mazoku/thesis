__author__ = 'tomas'


import numpy as np
import matplotlib.pyplot as plt


def run(data, params, n_steps=100, mask=None):

    debug = True
    # vmin = params['win_level'] - params['win_width'] / 2
    # vmax = params['win_level'] + params['win_width'] / 2

    if mask is None:
        mask = np.ones(data.shape, dtype=np.bool)

    nghb_idxs, nghb_ints = nghb_matrix(data)

    t_probs = trans_probs(data, nghb_idxs, nghb_ints)

    # visits = walk(t_probs, nghb_idxs, n_steps, start)
    vis_1 = walk(t_probs, nghb_idxs, n_steps, 0).reshape(data.shape)
    vis_2 = walk(t_probs, nghb_idxs, n_steps, data.shape[1]-1).reshape(data.shape)
    vis_3 = walk(t_probs, nghb_idxs, n_steps, (data.shape[0])*(data.shape[1]-1)).reshape(data.shape)
    vis_4 = walk(t_probs, nghb_idxs, n_steps, np.prod(data.shape)-1).reshape(data.shape)

    visits = (vis_1 + vis_2 + vis_3 + vis_4) / 4.

    plt.figure()
    plt.subplot(221), plt.imshow(vis_1, 'gray', interpolation='nearest')
    plt.subplot(222), plt.imshow(vis_2, 'gray', interpolation='nearest')
    plt.subplot(223), plt.imshow(vis_3, 'gray', interpolation='nearest')
    plt.subplot(224), plt.imshow(vis_4, 'gray', interpolation='nearest')

    plt.figure()
    plt.subplot(121)
    plt.imshow(data, 'gray', interpolation='nearest')
    plt.title('input')
    plt.subplot(122)
    plt.imshow(visits.reshape(data.shape), 'gray', interpolation='nearest')
    plt.title('visits')
    #plt.colorbar()
    plt.show()

    # print 'data:'
    # print data
    # print 'nghb idxs:'
    # print nghb_idxs
    # print 'nghb ints:'
    # print nghb_ints
    # print 't_probs:'
    # print t_probs



def walk(t_probs, nghb_idxs, n_steps, start):
    visits = np.zeros(t_probs.shape[1], dtype=np.uint16)
    curr_pt = start

    for i in range(n_steps):
        dir = get_direction(t_probs[:,curr_pt])
        curr_pt = nghb_idxs[dir, curr_pt]
        visits[curr_pt] += 1

    return visits


def get_direction(probs):
    prob = np.random.rand(1)[0]
    idx = -1
    probs = np.cumsum(probs)
    for i in range(len(probs)):
        if prob < probs[i]:
            idx = i
            break
    return idx


def nghb_matrix(data):
    if data.ndim == 2:
        n_nghbs = 4
        n_cols = data.shape[1]
    data = data.flatten()
    n_pts = len(data)
    nghb_m = np.mgrid[0:n_nghbs, 0:n_pts][1]

    #deriving the upper neighbor
    nghb_m[0,:] = np.roll(nghb_m[0,:], n_cols)
    nghb_m[0,0:n_cols] = -1
    #deriving the right neighbor
    nghb_m[1,:] = np.roll(nghb_m[1,:], -1)
    nghb_m[1,range(n_cols-1, n_pts, n_cols)] = -1
    #deriving the bottom neighbor
    nghb_m[2,:] = np.roll(nghb_m[2,:], -n_cols)
    nghb_m[2,-n_cols:] = -1
    #deriving the left neighbor
    nghb_m[3,:] = np.roll(nghb_m[3,:], 1)
    nghb_m[3,range(0, n_pts, n_cols)] = -1

    # creating matrix of neighbors' intensities
    nghb_ints = np.where(nghb_m >= 0, data[nghb_m], -1)

    return nghb_m, nghb_ints


def trans_probs(data, nghb_idxs, nghb_ints):
    data = data.ravel()
    n_pts = np.prod(data.shape)
    n_nghbs = 4
    t_probs = nghb_ints.copy()
    nghbs_count = np.sum(nghb_idxs >= 0, 0)  # number of neighbors of current point

    # subtracting the intensity of the 'central' pixel
    diff = np.where(nghb_idxs >= 0, np.abs(t_probs - np.tile(data, (n_nghbs, 1))), -1)

    # sum - diff, the addition is due to the -1 where there's no neighbor
    sum_diff = np.sum(diff, 0) + (n_nghbs - nghbs_count)
    recip = np.where(nghb_idxs >= 0, np.tile(sum_diff, (n_nghbs, 1)) - diff, -1)
    sum_recip = np.sum(recip, 0) + (n_nghbs - nghbs_count)

    # dividing by the sum -> probabilities
    t_probs = np.where(nghb_idxs >= 0, recip.astype(np.float) / sum_recip, 0)

    return t_probs


################################################################################
################################################################################
if __name__ == '__main__':

    params = dict()
    params['win_level'] = 50
    params['win_width'] = 350
    # data = np.array([[2, 14, 8], [32, 40, 12], [19, 22, 6]])
    data = np.array([[  5,   8,  12,   4,   8,   6,   9,   0,   5,   8],
                     [  1,  10,  16,   5,   2,   0,   2,   8,   7,   5],
                     [  9,   2,   6,   8,   2,  10,   7,   1,   9,  11],
                     [  7,   6,  10,  15,  58,  90,  12,   9,  15,   6],
                     [ 12,   8,  11, 130, 200, 110,  96,  12,  18,   1],
                     [  8,   9,  26,  99, 120,  80,  78,  20,   5,  11],
                     [ 20,  12,  14, 150, 160,  90, 100,  16,  11,  15],
                     [ 10,  13,   9,  12, 136, 120, 110,  20,   8,   8],
                     [ 11,  13,  11,  10,  76,  60,   9,   0,  10,  36],
                     [  9,  11,  18,   4,   7,  20,  18,  14,   9,   1]])

    n_steps = 10000
    run(data, params, n_steps)