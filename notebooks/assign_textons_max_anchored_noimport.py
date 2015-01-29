# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import itertools
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
    
centroids = dm.load_pipeline_result('textons', 'npy')
features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

# <codecell>

# reconstruct textons

# from skimage.util import pad
# padded_kernels = [None] * dm.n_kernel
# for i, kern in enumerate(dm.kernels):
#     ksize = kern.shape[0]
#     a = (dm.max_kern_size - ksize)/2
#     padded_kern = pad(kern, [a, a], mode='constant', constant_values=0)
#     padded_kernels[i] = padded_kern
    
    
# F = np.vstack([k.flatten() for k in padded_kernels])

# from numpy.linalg import lstsq

# for f in centroids:
#     b, r, _, _ = lstsq(F, f)

#     c = b.reshape((dm.max_kern_size,dm.max_kern_size))

#     plt.imshow(c[dm.max_kern_size/2-30:dm.max_kern_size/2+30,
#                  dm.max_kern_size/2-30:dm.max_kern_size/2+30], cmap=plt.cm.Greys_r)
    
#     plt.show()

# <codecell>

n_texton = len(centroids)
# n_texton = 10

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000, init=centroids)
kmeans.fit(features_rotated)
final_centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# <codecell>

# try:
#     textonmap = dm.load_pipeline_result('texMap', 'npy')
    
# except Exception as e:
    
textonmap = -1 * np.ones_like(dm.image, dtype=np.int)
textonmap[dm.mask] = labels

dm.save_pipeline_result(textonmap, 'texMap', 'npy')

# <codecell>

hc_colors = np.loadtxt('../visualization/100colors.txt')

vis = label2rgb(textonmap, colors=hc_colors, alpha=1.)

# <codecell>

# display(vis)

# <codecell>

dm.save_pipeline_result(vis, 'texMap', 'png')

# <codecell>

# plt.hist(textonmap.flat, bins=np.arange(n_texton+1))
# plt.xlabel('texton')
# plt.xticks(np.arange(n_texton))
# plt.show()

# <codecell>

# def overlay_labels(image, lbp, labels):
    
#     mask = np.logical_or.reduce([lbp == each for each in labels])
#     return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

# def visualize_textonmap_layered(textonmap, cropped_img):
    
#     for s in range(n_texton):
#         print s
#         overlayed = overlay_labels(cropped_img, textonmap, [s])
#         cv2.imwrite('overlayed_pca_texton%d.png'%s, img_as_ubyte(overlayed)[..., ::-1])
#     return 

