from __future__ import division

import cPickle as pickle
import gzip

fname = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-_old.pklz'
fname2 = '/home/tomas/Data/liver_segmentation/org-exp_183_46324212_venous_5.0_B30f-.pklz'
f = gzip.open(fname, 'rb')
fcontent = f.read()
f.close()

data_dict = pickle.loads(fcontent)

data_dict['data3d'] = data_dict['data3d'][1:, 1:-1, 1:-1]
data_dict['segmentation'] = data_dict['segmentation'][1:, 1:-1, 1:-1]

f = gzip.open(fname2, 'wb', compresslevel=1)
pickle.dump(data_dict, f)
f.close()