# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 0

from preamble import *

# <codecell>

pattern = cv2.imread('/home/yuncong/cheetah.png', 0)

# <codecell>

from joblib import Parallel, delayed
from scipy.signal import fftconvolve

def convolve_per_proc(i):
    return fftconvolve(pattern, dm.kernels[i], 'same')

filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                        for i in range(dm.n_kernel))

features = np.empty((pattern.shape[0], pattern.shape[1], dm.n_kernel))
for i in range(dm.n_kernel):
    features[...,i] = filtered[i]

del filtered

# <codecell>

# valid_features = features.reshape((pattern.shape[0] * pattern.shape[1], -1))

# <codecell>

# def rotate_features(fs):
#     features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
#     max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
#     features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
#                                for i, ai in enumerate(max_angle_indices)], (fs.shape[0], dm.n_freq * dm.n_angle))
    
#     return features_rotated

# from joblib import Parallel, delayed

# n_splits = 1000
# features_rotated_list = Parallel(n_jobs=16)(delayed(rotate_features)(fs) for fs in np.array_split(valid_features, n_splits))
# features_rotated = np.vstack(features_rotated_list)

# del valid_features

# <codecell>

features.reshape((-1, 99)).shape

# <codecell>

n_texton = 100
# n_texton = 10

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
# kmeans.fit(features_rotated_pca)
# kmeans.fit(features_rotated)
kmeans.fit(features.reshape((-1, 99)))
centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

from scipy.cluster.hierarchy import fclusterdata
cluster_assignments = fclusterdata(centroids, 1.15, method="complete")
# cluster_assignments = fclusterdata(centroids, .8, method="complete", criterion="distance")


#########  NOTE that multiple runs of kmeans gives different reduced cluster assignments !!! ########3

# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import average, fcluster, leaders, complete, single, dendrogram
# dists = pdist(centroids)
# lkg = complete(dists)
# cluster_ids = fcluster(lkg, 1.15)

reduced_centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])

n_reduced_texton = len(reduced_centroids)
print n_reduced_texton

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_reduced_texton, batch_size=1000, init=reduced_centroids)
# kmeans.fit(features_rotated_pca)
# kmeans.fit(features_rotated)
kmeans.fit(features.reshape((-1, 99)))

final_centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# <codecell>

# hc_colors = np.loadtxt('hc_colors.txt', delimiter=',')/ 255.
# hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt')/ 255.
hc_colors = np.loadtxt('../visualization/100colors.txt')

# hc_colors = np.random.random((n_texton, 3))
# np.savetxt('../visualization/100colors.txt', hc_colors)

# <codecell>

def visualize_features(centroids, n_freq, n_angle, colors=None):
    """
    if colors is not None, colorcodes are plotted below feature matrices
    """

    import itertools
    from matplotlib import gridspec
    
    n_cols = min(10, len(centroids))
    n_rows = int(np.ceil(n_texton/n_cols))
        
    vmin = centroids.min()
    vmax = centroids.max()

    fig = plt.figure(figsize=(20,20), facecolor='white')
        
    if colors is None:
        gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[1]*n_rows)
        for r, c in itertools.product(range(n_rows), range(n_cols)):
            i = r * n_cols + c
            if i == len(centroids): break
            ax_mat = fig.add_subplot(gs[r*n_cols+c])
            ax_mat.set_title('texton %d'%i)
            ax_mat.matshow(centroids[i].reshape(n_freq, n_angle), vmin=vmin, vmax=vmax)
            ax_mat.set_xticks([])
            ax_mat.set_yticks([])
    else:
        gs = gridspec.GridSpec(2*n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[4,1]*n_rows)
        for r, c in itertools.product(range(n_rows), range(n_cols)):
            i = r * n_cols + c
            if i == len(centroids): break
            ax_mat = fig.add_subplot(gs[r*2*n_cols+c])
            ax_mat.set_title('texton %d'%i)
            ax_mat.matshow(centroids[i].reshape(n_freq, n_angle), vmin=vmin, vmax=vmax)
            ax_mat.set_xticks([])
            ax_mat.set_yticks([])
            
            ax_cbox = fig.add_subplot(gs[(r*2+1)*n_cols+c])
            cbox = np.ones((1,2,3))
            cbox[:,:,:] = colors[i]
            ax_cbox.imshow(cbox)
            ax_cbox.set_xticks([])
            ax_cbox.set_yticks([])

    plt.tight_layout()

    plt.show()

# <codecell>

visualize_features(final_centroids, dm.n_freq, dm.n_angle, colors=hc_colors)

# <codecell>

textonmap = -1 * np.ones_like(pattern, dtype=np.int)
textonmap = labels.reshape(pattern.shape)
# vis = label2rgb(textonmap, image=cropped_img)
vis = label2rgb(textonmap, colors=hc_colors, alpha=1.)

plt.imshow(pattern, cmap=cm.Greys_r)
plt.show()

plt.imshow(vis)
plt.show()

# <codecell>

from skimage.util import pad
padded_kernels = [None] * dm.n_kernel
for i, kern in enumerate(dm.kernels):
    ksize = kern.shape[0]
    a = (dm.max_kern_size - ksize)/2
    padded_kern = pad(kern, [a, a], mode='constant', constant_values=0)
    padded_kernels[i] = padded_kern

# <codecell>

F = np.vstack([k.flatten() for k in padded_kernels])

from numpy.linalg import lstsq

for f in final_centroids:
    b, r, _, _ = lstsq(F, f)

    c = b.reshape((dm.max_kern_size,dm.max_kern_size))

    plt.imshow(c[dm.max_kern_size/2-30:dm.max_kern_size/2+30,
                 dm.max_kern_size/2-30:dm.max_kern_size/2+30], cmap=plt.cm.Greys_r)
    
    plt.show()

