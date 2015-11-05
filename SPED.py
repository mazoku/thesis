from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import graph_tools as gt


def cost_of_path(G, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += G.edge[path[i]][path[i + 1]]['weight']
    return cost

def run(data, p0, pts, show=False):
    '''

    Parameters
    ----------
    data ... input image
    p0 ... source point (x, y)
    pts ... target points [(x1, y1), (x2, y2), ... , (xn, yn)]
    show ... whether to show results or not

    Returns
    -------

    '''
    G = gt.create_graph(data, wtype=1)

    p0_lin = np.ravel_multi_index(p0, data.shape)
    pts_lin = [np.ravel_multi_index(x, data.shape) for x in pts]
    n_pts = len(pts_lin)

    paths = list()
    costs = list()
    for i in range(n_pts):
        path = nx.shortest_path(G, p0_lin, pts_lin[i], 'weight')
        paths.append(path)
        cost = cost_of_path(G, path)
        costs.append(cost)

    # cityblock distance
    dists = list()
    for i in range(n_pts):
        dist = np.abs(np.array(p0) - np.array(pts[i])).sum()
        dists.append(dist)

    # SPED value
    speds = np.array(dists) / np.array(costs)

    for i in range(n_pts):
        print 's: ', p0, ', t: ', pts[i], ', path: ', paths[i], ', cost: ', costs[i]
        print '\tdist: %.2f, sp: %.2f, sped = %.2f' % (dists[i], costs[i], speds[i])

    if show:
        for i in range(n_pts):
            plt.figure()
            plt.imshow(data, 'gray', interpolation='nearest')
            plt.hold(True)
            for j in range(len(paths[i])):
                xs = [np.unravel_index(x, data.shape)[1] for x in paths[i]]
                ys = [np.unravel_index(x, data.shape)[0] for x in paths[i]]
                # pt = np.unravel_index(paths[i][j], data.shape)
                plt.plot(xs, ys, 'ro-')
            plt.plot(p0[1], p0[0], 'go')  # source point
            plt.plot(pts[i][1], pts[i][0], 'bo')  # target point
            plt.axis('image')
            plt.title('sped = %.2f' % speds[i])

        plt.show()

if __name__ == '__main__':
    show = True
    c = 10

    data = np.zeros((10, 10), dtype=np.uint8)
    data[5:8, 1:6] = c

    p0 = [0, 1]
    p1 = [6, 3]
    p2 = [1, 7]
    p3 = [8, 4]
    pts = list((p1, p2, p3))

    run(data, p0, pts, show)

    # plt.figure()
    # plt.imshow(data, 'gray', interpolation='nearest')
    # plt.show()