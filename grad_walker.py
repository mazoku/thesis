__author__ = 'tomas'


import numpy as np


def run(data, params, mask=None):

    debug = True
    # vmin = params['win_level'] - params['win_width'] / 2
    # vmax = params['win_level'] + params['win_width'] / 2

    if mask is None:
        mask = np.ones(data.shape, dtype=np.bool)

    nghbs_ints, nghbs_idxs = nghb_matrix(data)

    t_probs = trans_probs(data)


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

    #TODO: nghb_m jsou indexy, jeste udelat pole s intenzitami

    return nghb_m


def trans_probs(data):
    n_pts = np.prod(data.shape)
    t_probs = np.zeros()

    return t_probs


################################################################################
################################################################################
if __name__ == '__main__':

    params = dict()
    params['win_level'] = 50
    params['win_width'] = 350
    data = np.array([[2, 14, 8], [32, 40, 12], [19, 22, 6]])
    run(data, params)