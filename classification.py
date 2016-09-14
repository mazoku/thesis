from __future__ import division

import lession_localization as lesloc


def reshape_struc(data):
    data_out = []
    for im1, im2, slics in data:
        for s1, s2 in slics:
            data_out.append((im1, im2, s1, s2))
    return data_out


################################################################################
################################################################################
if __name__ == '__main__':
    data = lesloc.get_data_struc()
    data = reshape_struc(data)
