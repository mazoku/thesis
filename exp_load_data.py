from imtools import tools
import matplotlib.pyplot as plt

import skimage.segmentation as skiseg

fnames = ['/home/tomas/Dropbox/Data/medical/dataset/gt/235_arterial-GT.pklz',
              '/home/tomas/Dropbox/Data/medical/dataset/gt/183a_arterial-GT.pklz',
              '/home/tomas/Dropbox/Data/medical/dataset/gt/232_venous-GT.pklz',
              '/home/tomas/Dropbox/Data/medical/dataset/gt/234_venous-GT.pklz',
              '/home/tomas/Dropbox/Data/medical/dataset/gt/180_arterial-GT.pklz',
              '/home/tomas/Dropbox/Data/medical/dataset/gt/185a_venous-GT.pklz',
              ]
slices = [13, 12, 8, 6, 11, 17]

for f, s in zip(fnames, slices):
    data, gt_mask, voxel_size = tools.load_pickle_data(f)
    data = tools.windowing(data)
    data = data[s,...]
    gt_mask = gt_mask[s,...]

    plt.figure()
    plt.subplot(121), plt.imshow(data, 'gray', interpolation='nearest'), plt.axis('off')
    # plt.subplot(122), plt.imshow(gt_mask, 'gray', interpolation='nearest')
    plt.subplot(122), plt.imshow(skiseg.mark_boundaries(data, gt_mask==1, color=(1, 0, 0), mode='thick'), interpolation='nearest'), plt.axis('off')
    plt.suptitle(f.split('/')[-1] + ', ' + str(s))
plt.show()