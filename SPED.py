from __future__ import division

import sys

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from PyQt4.QtGui import QApplication

import graph_tools as gt
from simple_viewer import SimpleViewer


def cost_of_path(G, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += G.edge[path[i]][path[i + 1]]['weight']
    return cost

def single_source(data):
    plt.figure()
    plt.imshow(data, 'gray', interpolation='nearest')
    plt.axis('image')
    pt0 = plt.ginput(1, timeout=0)  # returns list of tuples
    plt.close('all')

    if len(pt0) != 1:
        print 'Need exactly one points.'
        return

    pt0 = (int(round(pt0[0][0])), int(round(pt0[0][1]))) #[(int(round(x[0])), int(round(x[1]))) for x in pts]
    pts = np.indices(data.shape).reshape((2, np.prod(data.shape)))
    pts = [(x[0], x[1]) for x in pts.transpose()]

    speds = run(data, pt0, pts)
    speds_im = speds.reshape(data.shape)

    plt.figure()
    plt.imshow(speds_im, 'gray', interpolation='nearest')
    plt.show()

def interactive(data):
    # app = QApplication(sys.argv)
    # sv = SimpleViewer()
    # sv.view_widget.setSlice(data)
    # sv.view_widget.mouseClickSignal.connect(mouse_press)
    # sv.show()
    # sys.exit(app.exec_())

    plt.figure()
    plt.imshow(data, 'gray', interpolation='nearest')
    plt.axis('image')
    pts = plt.ginput(-1, timeout=0)  # returns list of tuples
    plt.close('all')

    pts = [(int(round(x[0])), int(round(x[1]))) for x in pts]

    if len(pts) >= 2:
        run(data, pts[0], pts[1:], show=True)
    else:
        print 'Need at least two points.'


def mouse_press(coords, intensity):
    print 'pressed at', coords, ', intensity:', intensity

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

    # graph creation
    G = gt.create_graph(data, wtype=1)

    # deriving linear indices of the points
    p0_lin = np.ravel_multi_index(p0, data.shape)
    pts_lin = [np.ravel_multi_index(x, data.shape) for x in pts]
    n_pts = len(pts_lin)

    # getting intensities of the points
    int0 = data[p0]
    ints = [data[x] for x in pts]

    # calculating shortest paths and their costs
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
    # speds = np.exp(np.absolute(ints - int0)) * np.array(dists) / np.array(costs)

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

    return speds

if __name__ == '__main__':
    show = True

    # c = 10
    # data = np.zeros((10, 10), dtype=np.uint8)
    # data[5:8, 1:6] = c

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

    p0 = (0, 1)
    p1 = (6, 3)
    p2 = (1, 7)
    p3 = (8, 4)
    pts = list((p1, p2, p3))

    # run(data, p0, pts, show)
    # interactive(data)
    single_source(data)

    # plt.figure()
    # plt.imshow(data, 'gray', interpolation='nearest')
    # plt.show()