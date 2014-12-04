# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 0

from utilities import *

if 'SSH_CONNECTION' in os.environ:
    DATA_DIR = '/home/yuncong/DavidData'
    REPO_DIR = '/home/yuncong/Brain'
else:
    DATA_DIR = '/home/yuncong/BrainLocal/DavidData_4'
    REPO_DIR = '/home/yuncong/BrainSaliencyDetection'

dm = DataManager(DATA_DIR, REPO_DIR)

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    vq_params_id = 'blueNissl'
    

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)
dm.set_vq_params(vq_params_id=args.vq_params_id)

from joblib import Parallel, delayed
from skimage.filter import gabor_kernel

theta_interval = dm.gabor_params['theta_interval']
n_angle = int(180/theta_interval)
freq_step = dm.gabor_params['freq_step']
freq_max = 1./dm.gabor_params['min_wavelen']

dm.gabor_params['max_wavelen'] = 50

freq_min = 1./dm.gabor_params['max_wavelen']
bandwidth = dm.gabor_params['bandwidth']
n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
frequencies = freq_max/freq_step**np.arange(n_freq)
angles = np.arange(0, n_angle)*np.deg2rad(theta_interval)

raw_kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in frequencies for t in angles]
raw_kernels = map(np.real, raw_kernels)

n_kernel = len(raw_kernels)

print 'num. of kernels: %d' % (n_kernel)
print 'frequencies:', frequencies
print 'wavelength (pixels):', 1/frequencies

max_kern_size = np.max([kern.shape[0] for kern in raw_kernels])
print 'max kernel matrix size:', max_kern_size

# compensate the numerical biases of kernels

biases = np.array([k.sum() for k in raw_kernels])

mean_bias = biases.mean()
kernels = [k/k.sum()*mean_bias for k in raw_kernels]

# <codecell>

from skimage.transform import rotate

w = 10
h = 10

im_stripes = np.zeros((h, w))

stripe_spacing = 4
stripe_width = 2

for row in range(0, h, stripe_spacing):
    im_stripes[row:row+stripe_width] = 1

thetas = np.random.randint(0, 6, 8)*30
ims = [rotate(im_stripes, -theta) > 0.5 for theta in thetas]
    
pattern = np.vstack([np.hstack(ims[:3]), np.hstack([ims[3], np.zeros((h, w)), ims[4]]), np.hstack(ims[5:])])

plt.imshow(pattern)

# <codecell>

from joblib import Parallel, delayed
from scipy.signal import fftconvolve

def convolve_per_proc(i):
    return fftconvolve(pattern, kernels[i], 'same').astype(np.half)

filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) 
                        for i in range(n_kernel))

features = np.empty((pattern.shape[0], pattern.shape[1], n_kernel), dtype=np.half)
for i in range(n_kernel):
    features[...,i] = filtered[i]

del filtered

# <codecell>

valid_features = features.reshape((pattern.shape[0] * pattern.shape[1], -1))

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

n_texton = 100
# n_texton = 10

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
# kmeans.fit(features_rotated_pca)
kmeans.fit(features_rotated)
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
kmeans.fit(features_rotated)
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

visualize_features(final_centroids, n_freq, n_angle, colors=hc_colors)

# <codecell>

from utilities import display

# <codecell>

textonmap = -1 * np.ones_like(pattern, dtype=np.int)
textonmap = labels.reshape((30,30))
# vis = label2rgb(textonmap, image=cropped_img)
vis = label2rgb(textonmap, colors=hc_colors, alpha=1.)

plt.imshow(pattern, cmap=cm.Greys_r)
plt.show()

plt.imshow(vis)
plt.show()

