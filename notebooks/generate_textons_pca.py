# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v4'
    REPO_DIR = '/home/yuncong/Brain'

dm = DataManager(DATA_DIR, REPO_DIR)

# import argparse
# import sys

# parser = argparse.ArgumentParser(
# formatter_class=argparse.RawDescriptionHelpFormatter,
# description='Execute feature extraction pipeline',
# epilog="""
# The following command processes image RS141_x5_0001.tif using blueNissl for both gabor parameters and segmentation parameters.
# python %s RS141 x5 1 -g blueNissl -s blueNissl
# """%(os.path.basename(sys.argv[0]), ))

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNissl')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 0
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

# <codecell>

from skimage.filter import gabor_kernel

theta_interval = dm.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = dm.gabor_params['freq_step']
freq_max = 1./dm.gabor_params['min_wavelen']
freq_min = 1./dm.gabor_params['max_wavelen']
bandwidth = dm.gabor_params['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)
angles = np.arange(0, n_angle)*np.deg2rad(theta_interval)

kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies for t in angles]
kernels = map(np.real, kernels)

n_kernel = len(kernels)

print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in kernels])
print 'max kernel matrix size:', max_kern_size

# <codecell>

biases = np.array([k.sum() for k in kernels])
mean_bias = biases.mean()
kernels = [k/k.sum()*mean_bias for k in kernels]

# dm.save_pipeline_result(kernels, 'kernels', 'pkl')

# <codecell>

def crop_borders(data):
    cropped_data = data[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, ...].copy()
    return cropped_data

# crop borders

try:
    cropped_features = dm.load_pipeline_result('cropFeatures', 'npy')
except:
    cropped_features = crop_borders(features)
    dm.save_pipeline_result(cropped_features, 'cropFeatures', 'npy')

try:
    cropped_img = dm.load_pipeline_result('cropImg', 'tif')    
except:
    cropped_img = crop_borders(dm.image)
    dm.save_pipeline_result(cropped_img, 'cropImg', 'tif')

try:
    cropped_mask = dm.load_pipeline_result('cropMask', 'npy')
except:
    cropped_mask = crop_borders(dm.mask)
    dm.save_pipeline_result(cropped_mask, 'cropMask', 'npy')
    dm.save_pipeline_result(cropped_mask, 'cropMask', 'tif')

cropped_height, cropped_width = cropped_img.shape[:2]
print cropped_height, cropped_width

# <codecell>

valid_features = cropped_features[cropped_mask]
n_valid = len(valid_features)

# del cropped_features

# <codecell>

def rotate_features(fs):
    features_tabular = fs.reshape((fs.shape[0], n_freq, n_angle))
    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                               for i, ai in enumerate(max_angle_indices)], (fs.shape[0], n_freq * n_angle))
    
    return features_rotated

from joblib import Parallel, delayed

n_splits = 1000
features_rotated_list = Parallel(n_jobs=16)(delayed(rotate_features)(fs) for fs in np.array_split(valid_features, n_splits))
features_rotated = np.vstack(features_rotated_list)

del valid_features

# <codecell>

dm.save_pipeline_result(features_rotated, 'features_rotated', 'npy')

# <codecell>

features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

# <codecell>

b = time.time()

n_components = 5

from sklearn.decomposition import RandomizedPCA 
pca = RandomizedPCA(n_components=n_components, whiten=True)
# pca = PCA(n_components=n_components, whiten=True)
pca.fit(features_rotated)
print(pca.explained_variance_ratio_)

features_rotated_pca = pca.transform(features_rotated)

print time.time() - b

# <codecell>

dm.save_pipeline_result(features_rotated_pca, 'features_rotated_pca', 'npy')

# <codecell>

import time
b = time.time()

n_texton = 100
# n_texton = 10

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
# kmeans.fit(features_rotated_pca)
kmeans.fit(features_rotated)
centroids = kmeans.cluster_centers_
# labels = kmeans.labels_


from scipy.cluster.hierarchy import fclusterdata
cluster_assignments = fclusterdata(centroids, 1.15, method="complete", criterion="inconsistent")
# cluster_assignments = fclusterdata(centroids, 60., method="complete", criterion="distance")


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
kmeans.fit(features_rotated)
final_centroids = kmeans.cluster_centers_
labels = kmeans.labels_

textonmap = -1 * np.ones_like(cropped_img, dtype=np.int)
textonmap[cropped_mask] = labels


print time.time() - b

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

# hc_colors = np.loadtxt('hc_colors.txt', delimiter=',')/ 255.
# hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt')/ 255.

hc_colors = np.loadtxt('../visualization/100colors.txt')

# hc_colors = np.random.random((n_texton, 3))
# np.savetxt('../visualization/100colors.txt', hc_colors)

# <codecell>

visualize_features(centroids, n_freq, n_angle, colors=hc_colors)

# <codecell>

visualize_features(reduced_centroids, n_freq, n_angle, colors=hc_colors)
# visualize_features(final_centroids, n_freq, n_angle, colors=hc_colors) # same as reduced_centroids

# <codecell>

plt.hist(textonmap.flat, bins=np.arange(n_reduced_texton+1))
plt.xlabel('reduced texton')
plt.xticks(np.arange(n_reduced_texton))
plt.show()

# <codecell>

vis = label2rgb(textonmap, colors=hc_colors, alpha=1.)
display(vis)

# <codecell>

dm.save_pipeline_result(reduced_centroids, 'textons', 'npy')

# <codecell>

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

def visualize_textonmap_layered(textonmap, cropped_img):
    
    for s in range(n_texton):
        print s
        overlayed = overlay_labels(cropped_img, textonmap, [s])
        cv2.imwrite('overlayed_pca_texton%d.png'%s, img_as_ubyte(overlayed)[..., ::-1])
    return 

# <codecell>

# Visualize pca textons

n_cols = 10
n_rows = int(np.ceil(n_texton/n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,5), facecolor='white', sharey=True)
axes = np.atleast_2d(axes)

for i in range(n_rows):
    for j in range(n_cols):
        axes[i, j].set_title('texton %d'%(i*10+j))
        axes[i, j].bar(np.arange(n_components), centroids[i*10+j])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
plt.tight_layout()

# plt.savefig('textons2.png', bbox_inches='tight')
# plt.close(fig)

