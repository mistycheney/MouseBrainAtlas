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
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'

# <codecell>

dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)

# <codecell>

# @timeit
# def build_gabor_kernels(gabor_params):
#     """
#     Generate the Gabor kernels
#     """

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

<<<<<<< HEAD
del valid_features

# <codecell>

dm.save_pipeline_result(features_rotated, 'features_rotated', 'npy')
=======
# <codecell>

del valid_features
>>>>>>> f8912cf4dc09a4e962a3631c724b7324db943a86

# <codecell>

b = time.time()

n_components = 5

from sklearn.decomposition import RandomizedPCA 
<<<<<<< HEAD
pca = RandomizedPCA(n_components=n_components, whiten=True)
# pca = PCA(n_components=n_components, whiten=True)
=======
pca = RandomizedPCA(n_components=n_components)
>>>>>>> f8912cf4dc09a4e962a3631c724b7324db943a86
pca.fit(features_rotated)
print(pca.explained_variance_ratio_)

features_rotated_pca = pca.transform(features_rotated)

print time.time() - b

# <codecell>

<<<<<<< HEAD
dm.save_pipeline_result(features_rotated_pca, 'features_rotated_pca', 'npy')

# <codecell>

# n_texton = 100
n_texton = 10

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=1000)
kmeans.fit(features_rotated_pca)
# kmeans.fit(features_rotated)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# <codecell>

a = np.random.choice(features_rotated_pca.shape[0], 1000)

plt.scatter(features_rotated_pca[a, 0], features_rotated_pca[a, 1], c='r')

plt.scatter(centroids[:, 0], centroids[:, 1])

# <codecell>

# from scipy.spatial.distance import pdist, squareform

# D = squareform(pdist(centroids))
# Dmin = D[D > 0].min()
# Dmax = D.max()
# plt.matshow(D, vmin=Dmin, vmax=Dmax)

# <codecell>

# hc_colors = np.loadtxt('hc_colors.txt', delimiter=',')/ 255.
# hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt')/ 255.

hc_colors = np.loadtxt('../visualization/100colors.txt')

# hc_colors = np.random.random((n_texton, 3))
# np.savetxt('../visualization/100colors.txt', hc_colors)

# <codecell>

# Visualize textons and color codes (in original space)

n_cols = 10
n_rows = int(np.ceil(n_texton/n_cols))

fig, axes = plt.subplots(2*n_rows, n_cols, figsize=(20,5), facecolor='white')
axes = np.atleast_2d(axes)

vmin = centroids.min()
vmax = centroids.max()

for i in range(n_rows):
    for j in range(n_cols):
        axes[2*i, j].set_title('texton %d'%(i*10+j))
        axes[2*i, j].matshow(centroids[i*10+j].reshape(n_freq, n_angle), vmin=vmin, vmax=vmax)
        axes[2*i, j].set_xticks([])
        axes[2*i, j].set_yticks([])
        
        cbox = np.ones((3,10,3))
        cbox[:,:,:] = hc_colors[i*10+j]
        axes[2*i+1, j].imshow(cbox)
        axes[2*i+1, j].set_xticks([])
        axes[2*i+1, j].set_yticks([])
        
# plt.tight_layout()

plt.subplots_adjust(left=0, right=1., top=1, bottom=0., wspace=0.1, hspace=0)

# plt.savefig('textons2.png', bbox_inches='tight')
# plt.close(fig)

# <codecell>

# Visualize textons (in original space)

n_cols = 10
n_rows = int(np.ceil(n_texton/n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,20), facecolor='white')
axes = np.atleast_2d(axes)

vmin = centroids.min()
vmax = centroids.max()

for i in range(n_rows):
    for j in range(n_cols):
        axes[i, j].set_title('texton %d'%(i*10+j))
        axes[i, j].matshow(centroids[i*10+j].reshape(n_freq, n_angle), vmin=vmin, vmax=vmax)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
plt.tight_layout()

# plt.savefig('textons2.png', bbox_inches='tight')
# plt.close(fig)

# <codecell>

# Visualize color codes (in original space)

n_cols = 10
n_rows = int(np.ceil(n_texton/n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,20), facecolor='white')
axes = np.atleast_2d(axes)

for i in range(n_rows):
    for j in range(n_cols):
        axes[i, j].set_title('texton %d'%(i*10+j))
        cbox = np.ones((10,10,3))
        cbox[:,:,:] = hc_colors[i*10+j]
        axes[i, j].imshow(cbox)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
plt.tight_layout()

# plt.savefig('textons2.png', bbox_inches='tight')
# plt.close(fig)

# <codecell>

# a = np.random.choice(features_rotated.shape[0], 10000)
# plt.scatter(features_rotated[a, 0], features_rotated[a, 1], c='r', s=.1)
# plt.scatter(centroids[:, 0], centroids[:, 1])
=======
b = time.time()

n_components = 5

from sklearn.decomposition import PCA
pca = PCA(n_components=n_components)
pca.fit(features_rotated)
print(pca.explained_variance_ratio_)

features_rotated_pca = pca.transform(features_rotated)

print time.time() - b

# <codecell>

n_texton = 50

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=n_texton, batch_size=100)
kmeans.fit(features_rotated_pca)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# <codecell>

plt.hist(textonmap.flat)
plt.show()

# <codecell>

hc_colors = np.loadtxt('hc_colors.txt', delimiter=',')/ 255.
>>>>>>> f8912cf4dc09a4e962a3631c724b7324db943a86

# <codecell>

textonmap = -1 * np.ones_like(cropped_img, dtype=np.int)
textonmap[cropped_mask] = labels
# vis = label2rgb(textonmap, image=cropped_img)
vis = label2rgb(textonmap, colors=hc_colors, alpha=1.)

# <codecell>

<<<<<<< HEAD
plt.hist(textonmap.flat, bins=np.arange(n_texton+1))
plt.xlabel('texton')
plt.show()

# <codecell>

=======
>>>>>>> f8912cf4dc09a4e962a3631c724b7324db943a86
cv2.imwrite('textonmap2.png', img_as_ubyte(vis)[..., ::-1])
from IPython.display import FileLink
FileLink('textonmap2.png')

# <codecell>

<<<<<<< HEAD
for s in range(n_texton):
    print s
    overlayed = overlay_labels(cropped_img, textonmap, [s])
    cv2.imwrite('overlayed_pca_texton%d.png'%s, img_as_ubyte(overlayed)[..., ::-1])
#     from IPython.display import FileLink
#     FileLink('overlayed.png')

# <codecell>

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

# <codecell>

# Visualize textons

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
=======
plt.matshow(cropped_features[..., 88], cmap=plt.cm.Greys_r)
>>>>>>> f8912cf4dc09a4e962a3631c724b7324db943a86

# <codecell>

# Over-segment the image into superpixels using SLIC (http://ivrg.epfl.ch/research/superpixels)

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_ubyte

img_rgb = gray2rgb(dm.image)

try:
    segmentation = dm.load_pipeline_result('segmentation', 'npy')
    
except Exception as e:
    segmentation = slic(img_rgb, n_segments=int(dm.segm_params['n_superpixels']), 
                        max_iter=10, 
                        compactness=float(dm.segm_params['slic_compactness']), 
                        sigma=float(dm.segm_params['slic_sigma']), 
                        enforce_connectivity=True)
    print 'segmentation computed'
    
    dm.save_pipeline_result(segmentation.astype(np.int16), 'segmentation', 'npy')

# <codecell>

from skimage.segmentation import relabel_sequential

try:
    cropped_segmentation_relabeled = dm.load_pipeline_result('cropSegmentation', 'npy')
except:
    # segmentation starts from 0
    cropped_segmentation = crop_borders(segmentation)
    n_superpixels = len(np.unique(cropped_segmentation))
    cropped_segmentation[~cropped_mask] = -1
    cropped_segmentation_relabeled, fw, inv = relabel_sequential(cropped_segmentation + 1)

    # make background label -1
    cropped_segmentation_relabeled -= 1
    dm.save_pipeline_result(cropped_segmentation_relabeled, 'cropSegmentation', 'npy')

# <codecell>

sp_props = regionprops(cropped_segmentation_relabeled + 1, intensity_image=cropped_img, cache=True)

def obtain_props_worker(i):
    return sp_props[i].centroid, sp_props[i].area, sp_props[i].mean_intensity, sp_props[i].bbox

r = Parallel(n_jobs=16)(delayed(obtain_props_worker)(i) for i in range(len(sp_props)))
sp_centroids = np.array([s[0] for s in r])
sp_areas = np.array([s[1] for s in r])
sp_mean_intensity = np.array([s[2] for s in r])
sp_bbox = np.array([s[3] for s in r])

sp_properties = np.column_stack([sp_centroids, sp_areas, sp_mean_intensity, sp_bbox])

dm.save_pipeline_result(sp_properties, 'cropSpProps', 'npy')

n_superpixels = len(np.unique(cropped_segmentation_relabeled))

img_superpixelized = mark_boundaries(gray2rgb(cropped_img), cropped_segmentation_relabeled)
img_superpixelized_text = img_as_ubyte(img_superpixelized)

# background label (-1) is not displayed
for s in range(n_superpixels - 1):
    img_superpixelized_text = cv2.putText(img_superpixelized_text, str(s), 
                      tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      .5, ((255,0,255)), 1)

dm.save_pipeline_result(img_superpixelized_text, 'cropSegmentation', 'tif')

# <codecell>

# Compute neighbor lists and connectivity matrix

from skimage.morphology import disk
from skimage.filter.rank import gradient
# from scipy.sparse import coo_matrix

try:
    neighbors = dm.load_pipeline_result('neighbors', 'npy')

except:

    edge_map = gradient(cropped_segmentation_relabeled.astype(np.uint8), disk(3))
    neighbors = [set() for i in range(n_superpixels)]

    for y,x in zip(*np.nonzero(edge_map)):
        neighbors[cropped_segmentation_relabeled[y,x]] |= set(cropped_segmentation_relabeled[y-2:y+2,x-2:x+2].ravel())

    for i in range(n_superpixels):
        neighbors[i] -= set([i])

    dm.save_pipeline_result(neighbors, 'neighbors', 'npy')

# <codecell>


