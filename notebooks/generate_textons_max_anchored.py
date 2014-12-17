# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from preamble import *

# <codecell>

# try:
#     features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

# except Exception as e:

features = dm.load_pipeline_result('features', 'npy')

valid_features = features[:, dm.mask].T
n_valid = len(valid_features)

del features

def rotate_features(fs):
    features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                               for i, ai in enumerate(max_angle_indices)], (fs.shape[0], dm.n_freq * dm.n_angle))

    return features_rotated

from joblib import Parallel, delayed

n_splits = 1000
features_rotated_list = Parallel(n_jobs=16)(delayed(rotate_features)(fs) for fs in np.array_split(valid_features, n_splits))
features_rotated = np.vstack(features_rotated_list)

del valid_features

dm.save_pipeline_result(features_rotated, 'features_rotated', 'npy')

# <codecell>

n_texton = 100
# n_texton = 10

# # regular kmeans very slow
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=n_texton)

try:
    centroids = dm.load_pipeline_result('original_centroids', 'npy')

except:
    
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
    # kmeans.fit(features_rotated_pca)
    kmeans.fit(features_rotated)
    centroids = kmeans.cluster_centers_
    # labels = kmeans.labels_

    dm.save_pipeline_result(centroids, 'original_centroids', 'npy')

# <codecell>

from scipy.cluster.hierarchy import fclusterdata
# cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
cluster_assignments = fclusterdata(centroids, 80., method="complete", criterion="distance")

reduced_centroids = np.array([centroids[cluster_assignments == i].mean(axis=0) for i in set(cluster_assignments)])

n_reduced_texton = len(reduced_centroids)
print n_reduced_texton

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_reduced_texton, batch_size=1000, init=reduced_centroids)
# kmeans.fit(features_rotated_pca)
kmeans.fit(features_rotated)
final_centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# <codecell>

# dm.save_pipeline_result(reduced_centroids, 'textons', 'npy')

# <codecell>

# def visualize_features(centroids, n_freq, n_angle, colors=None):
#     """
#     if colors is not None, colorcodes are plotted below feature matrices
#     """

#     import itertools
#     from matplotlib import gridspec
    
#     n_cols = min(10, len(centroids))
#     n_rows = int(np.ceil(n_texton/n_cols))
        
#     vmin = centroids.min()
#     vmax = centroids.max()

#     fig = plt.figure(figsize=(20,20), facecolor='white')
        
#     if colors is None:
#         gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[1]*n_rows)
#         for r, c in itertools.product(range(n_rows), range(n_cols)):
#             i = r * n_cols + c
#             if i == len(centroids): break
#             ax_mat = fig.add_subplot(gs[r*n_cols+c])
#             ax_mat.set_title('texton %d'%i)
#             ax_mat.matshow(centroids[i].reshape(n_freq, n_angle), vmin=vmin, vmax=vmax)
#             ax_mat.set_xticks([])
#             ax_mat.set_yticks([])
#     else:
#         gs = gridspec.GridSpec(2*n_rows, n_cols, width_ratios=[1]*n_cols, height_ratios=[4,1]*n_rows)
#         for r, c in itertools.product(range(n_rows), range(n_cols)):
#             i = r * n_cols + c
#             if i == len(centroids): break
#             ax_mat = fig.add_subplot(gs[r*2*n_cols+c])
#             ax_mat.set_title('texton %d'%i)
#             ax_mat.matshow(centroids[i].reshape(n_freq, n_angle), vmin=vmin, vmax=vmax)
#             ax_mat.set_xticks([])
#             ax_mat.set_yticks([])
            
#             ax_cbox = fig.add_subplot(gs[(r*2+1)*n_cols+c])
#             cbox = np.ones((1,2,3))
#             cbox[:,:,:] = colors[i]
#             ax_cbox.imshow(cbox)
#             ax_cbox.set_xticks([])
#             ax_cbox.set_yticks([])

#     plt.tight_layout()

#     plt.show()

# <codecell>

# hc_colors = np.loadtxt('hc_colors.txt', delimiter=',')/ 255.
# hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt')/ 255.

hc_colors = np.loadtxt('../visualization/100colors.txt')

# hc_colors = np.random.random((n_texton, 3))
# np.savetxt('../visualization/100colors.txt', hc_colors)

# <codecell>

# visualize_features(centroids, dm.n_freq, dm.n_angle, colors=hc_colors)

# <codecell>

# visualize_features(reduced_centroids, dm.n_freq, dm.n_angle, colors=hc_colors)

