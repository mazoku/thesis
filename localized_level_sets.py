from __future__ import division

# import sys
# sys.path.append('localized_seg/localized_seg_demo.m')

import matlab.engine


def run(im, init_mask, max_iter=100):
    eng = matlab.engine.start_matlab()
    eng.addpath('localized_seg')
    # eng.localized_seg_demo(nargout=0)

    seg = eng.localized_seg(im, init_mask, max_iter, nargout=1)