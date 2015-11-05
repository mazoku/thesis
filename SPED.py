import numpy as np
import matplotlib.pyplot as plt

import graph_tools as gt

def run(data, p0, pts):
    G = gt.create_graph(data, wtype=1)


if __name__ == '__main__':
    c = 10

    data = np.zeros((10, 10), dtype=np.uint8)
    data[5:8, 1:6] = c

    p0 = [0, 1]
    p1 = [6, 3]
    p2 = [1, 7]
    p3 = [8, 4]
    pts = list((p1, p2, p3))

    run(data, p0, pts)

    plt.figure()
    plt.imshow(data, 'gray', interpolation='nearest')
    plt.show()