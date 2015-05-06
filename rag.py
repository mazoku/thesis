__author__ = 'tomas'

import matplotlib.pyplot as plt
from matplotlib import colors

from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    weight : float
        The absolute difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


################################################################################
################################################################################
if __name__ == '__main__':

    img = io.imread('/home/tomas/Dropbox/images/medicine/hypodenseTumor_box.jpg')
    labels = segmentation.slic(img, compactness=30, n_segments=100)
    # labels = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[:-1])
    g = graph.rag_mean_color(img, labels)


    # labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False,
    #                                    in_place_merge=True,
    #                                    merge_func=merge_mean_color,
    #                                    weight_func=_weight_mean_color)

    # g2 = graph.rag_mean_color(img, labels2)
    #
    # out = color.label2rgb(labels2, img, kind='avg')
    # out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

    # plt.figure()
    # plt.subplot(221), plt.imshow(img, 'gray', interpolation='nearest')
    # plt.subplot(222), plt.imshow(labels, interpolation='nearest')
    # plt.subplot(224), plt.imshow(labels2, interpolation='nearest')
    # plt.show()

    cmap = colors.ListedColormap(['blue', 'red'])
    # out = segmentation.mark_boundaries(img, labels)
    # out = graph.draw_rag(labels, g, img, thresh=20, node_color='yellow', colormap=cmap)
    out = graph.draw_rag(labels, g, np.zeros_like(img), thresh=50, node_color='yellow', colormap='jet')

    plt.figure()
    plt.imshow(out)
    plt.colorbar()
    plt.show()

    # io.imshow(out)
    # io.show()