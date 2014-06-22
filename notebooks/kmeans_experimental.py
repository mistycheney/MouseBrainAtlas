# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Load the image and cached texture features.

# <codecell>

import warnings
warnings.filterwarnings('ignore')

%load_ext autoreload
%autoreload 2

import skimage
from skimage.color import color_dict
import cv2
import numpy as np
from utilities import *

# img_name = 'Resized_1_region'
# img_name = 'Resized_1_region2'
# img_name = 'Resized_2_region'
# img_name = 'Resized_6_region'
# img_name = 'Resized_6_region_similar'

# CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
# IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region1/'
# img_name_fmt = 'PMD1305_%d.reduce2.region1'
# img_id = 244
# img_name = img_name_fmt%img_id

# CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
# IMG_DIR = '/home/yuncong/my_csd181/ParthaData/PMD1305_reduce2/region3/'
# img_name_fmt = 'PMD1305_%d.reduce2.region3'
# img_id = 4
# img_name = img_name_fmt%img_id

# img_name = 'Resized_6_region'
# IMG_DIR = '/home/yuncong/my_csd181_scratch/'

# img = cv2.imread(IMG_DIR + 'PMD1305_%d.reduce2.region.tif' % img_id, 0)
img = cv2.imread(IMG_DIR + img_name + '.tif', 0)

# features = np.load('/home/yuncong/my_csd181_scratch/'+img_name+'_features.npy')
# print features.shape

filtered = np.load('/home/yuncong/my_csd181_scratch/'+img_name+'_filtered.npy')

dirmap = np.load('/home/yuncong/my_csd181_scratch/'+img_name + '_dirmap.npy')

# img = cv2.imread('/home/yuncong/my_csd181_scratch/'+img_name+'.png', 0)

height, width, n_feat = filtered.shape
# features = features[200:1400, 200:3000, :]
# img = img[200:1400, 200:3000]


# features = features[200:2200, 200:1400, :]
# img = img[200:2200, 200:1400]

# <codecell>

import skimage
from skimage.color import color_dict

from skimage.segmentation import clear_border, felzenszwalb, find_boundaries, slic, mark_boundaries
from skimage.filter import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.morphology import binary_dilation, binary_erosion, watershed, label
from skimage.color import gray2rgb, label2rgb
from skimage.measure import regionprops
from scipy.spatial.distance import pdist, squareform
from skimage.util import img_as_ubyte
from IPython.display import FileLink, Image, FileLinks

mask = foreground_mask(img)
cv2.imwrite(img_name+'_img.jpg', 
            img_as_ubyte(img*mask))
FileLink(img_name+'_img.jpg')

# <codecell>

# features = features[100:-100, 100:-100, :]
filtered = filtered[100:-100, 100:-100, :]
img = img[100:-100, 100:-100]
mask = mask[100:-100, 100:-100]
dirmap = dirmap[100:-100, 100:-100]

# <codecell>

fig, axes = plt.subplots(ncols=4, nrows=n_feat/4, figsize=(20,100))
for i, ax in zip(range(n_feat), axes.flat):
    ax.matshow(filtered[...,i], cmap=plt.cm.coolwarm_r)
    ax.set_title('feature %d'%i)
    ax.axis('off')

plt.show()

# <codecell>

im = cv2.imread('/oasis/projects/nsf/csd181/iizhaki/Final/Folder_2026182415/KMean.png')
img = im[:,:,0]

# <codecell>

features_masked = features.view(ma.MaskedArray)
features_masked[~mask,:] = np.ma.masked

# <codecell>

from scipy.spatial.distance import cdist

X = np.random.random((30,2))
def kmean(X, num_clusters):
    num_data = X.shape[0]
    centroids = X[np.random.choice(range(num_data), size=num_clusters)]
    for iteration in range(10):
        num_clusters = 3
        D = cdist(X, centroids)
        labels = D.argmin(axis=1)
        centroids

        
        
    

# <codecell>

import h5py
CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
h5f = h5py.File(CACHE_DIR+'data.h5','r')
filtered = h5f['dataset_1'][:]
h5f.close()

# <codecell>

CACHE_DIR = '/home/yuncong/my_csd181_scratch/'
img_name = 'PMD1305_244_reduce0_region0'
features = np.load(CACHE_DIR + img_name + '_features.npy')

# <codecell>

features = np.dstack(filtered)

# <codecell>

b = features.reshape(-1, 72)

# <codecell>

del filtered

# <codecell>

%%time
from sklearn.cluster import MiniBatchKMeans 

num_textons = 20
kmeans_model = MiniBatchKMeans(num_textons, batch_size=100)
kmeans_model.fit(b)

# <codecell>

from sklearn.cluster import MiniBatchKMeans 
from skimage.color import label2rgb
num_textons = 20
kmeans_model = MiniBatchKMeans(num_textons)
kmeans_model.fit(features[mask,:].reshape(-1, n_feat))

# <markdowncell>

# Oversegment the image into superpixels.

# <codecell>

img_masked = img.view(np.ma.MaskedArray)
img_masked[~mask] = np.ma.masked

# <codecell>

img_rgb = gray2rgb(img)
# segmentation = slic(img_rgb, compactness=10, sigma=20)
# segmentation = slic(img_rgb, compactness=5, sigma=20)
segmentation = slic(img_rgb, n_segments=2000, max_iter=10, compactness=5, sigma=10, enforce_connectivity=True)

# <codecell>

from skimage.segmentation import felzenszwalb
img_rgb = gray2rgb(img)

for scale in range(1,10):
    segmentation = felzenszwalb(img_rgb, scale=scale, sigma=5, min_size=100)
#     img_superpixelized = mark_boundaries(img_rgb, segmentation)
    img_superpixelized_patches = label2rgb(segmentation, img_rgb)
#     img_superpixelized_patches = label2rgb(segmentation, None)
    cv2.imwrite(CACHE_DIR + img_name+'.img_superpixelized_patches_scale%d.jpg'%scale, img_as_ubyte(img_superpixelized_patches))

# <codecell>

from skimage.segmentation import felzenszwalb
img_rgb = gray2rgb(img)
segmentation = felzenszwalb(img_rgb, scale=2, sigma=5, min_size=100)

# <codecell>

from skimage.segmentation import quickshift
img_rgb = gray2rgb(img)
segmentation = quickshift(img_rgb)

# <codecell>

n_superpixels = len(np.unique(segmentation))

img_superpixelized = mark_boundaries(img_rgb, segmentation)
superpixel_boundaries = find_boundaries(segmentation)
img_superpixelized_patches = label2rgb(segmentation, img_rgb)

sp_props = regionprops(segmentation+1, intensity_image=img, cache=True)
sp_centroids = np.array([s.centroid for s in sp_props])
sp_areas = np.array([s.area for s in sp_props])
# sp_wcentroids = np.array([s.weighted_centroid for s in sp_props])
sp_centroid_dist = pdist(sp_centroids)
sp_centroid_dist_matrix = squareform(sp_centroid_dist)

sp_mean_intensity = np.array([s.mean_intensity for s in sp_props])

# cv2.imwrite(img_name+'_superpixelized.jpg', img_as_ubyte(img_superpixelized))
# Image(img_name+'_superpixelized.jpg')

# colors = np.random.random((n_superpixels,3))
# a = label2rgb(segmentation, img_rgb, colors=colors)

img_superpixelized_text = img_as_ubyte(img_superpixelized)
for s in range(n_superpixels):
    img_superpixelized_text = cv2.putText(img_superpixelized_text, str(s), 
                                          tuple(np.floor(sp_centroids[s][::-1]).astype(np.int)), 
                                          cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                          .5, ((255,0,255)), 1)
    
img_superpixelized_text = img_superpixelized_text/255.
# cv2.imwrite(img_name+'.img_superpixelized_text.jpg', img_as_ubyte(img_superpixelized_text))
# Image(img_name+'.img_superpixelized_text.jpg')

# <codecell>

img_superpixelized_patches = label2rgb(segmentation, img)
cv2.imwrite(img_name+'.img_superpixelized_patches.jpg', img_as_ubyte(img_superpixelized_patches))
FileLink(img_name+'.img_superpixelized_patches.jpg')

# <codecell>

cv2.imwrite(img_name+'.img_superpixelized_text.jpg', img_as_ubyte(img_superpixelized_text))
FileLink(img_name+'.img_superpixelized_text.jpg')

# <codecell>

cv2.imwrite(img_name+'.img_superpixelized.jpg', img_as_ubyte(img_superpixelized))
FileLink(img_name+'.img_superpixelized.jpg')

# <codecell>

superpixels_bg_count = np.array([(~mask[segmentation==i]).sum() for i in range(n_superpixels)])
bg_superpixels = np.nonzero((superpixels_bg_count/sp_areas) > 0.7)[0]
print len(bg_superpixels), 'background superpixels'

# <markdowncell>

# Cell segmentation

# <codecell>

import skimage
from skimage.color import color_dict

from skimage.segmentation import clear_border
from skimage.filter import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import watershed

from skimage.morphology import label

# t_img = threshold_adaptive(img, 25, offset=.01)
# t_img = gaussian_filter(img, sigma=20) < 220
t_img = gaussian_filter(img, sigma=10) < 220./255.
# b_img = binary_erosion(-t_img, np.ones((3, 3)))
# d_img = binary_dilation(b_img, np.ones((3, 3)))
# clear_border(d_img)

# t_img = remove_small_objects(t_img, min_size=40)

labels, n_labels = label(t_img, neighbors=4, return_num=True)

reg = regionprops(labels+1)

all_areas = np.array([r.area for r in reg])

a = np.concatenate([labels[0,:] ,labels[-1,:] ,labels[:,0] ,labels[:,-1]])
border_labels = np.unique(a)
border_labels_large = np.nonzero(all_areas > 250)[0]
border_labels_remove = [i for i in border_labels_large if i != all_areas.argmax()]
background = np.ones_like(img)
for i in border_labels_remove:
    background[labels==i] = 0

labelmap = skimage.color.label2rgb(labels, image=None, colors=None, alpha=0.3, \
                                   bg_label=0, bg_color=color_dict['white'], image_alpha=1)

plt.imshow(background, cmap=plt.cm.Greys_r)
plt.show()
# cv2.imwrite(img_name+'_cells.jpg', img_as_ubyte(labelmap))
# Image(img_name+'_cells.jpg')

# <codecell>

# a = np.zeros((n_superpixels), dtype=np.int)
# for i,l in zip(segmentation.flat, labels.flat):
#     a[i].append(l)
    
sp_cell_labels = [labels.flat[segmentation.flat == s] for s in range(n_superpixels)]

# <codecell>

sp_cell_counts = [len(np.unique(s))-1 for s in sp_cell_labels]

# <codecell>

sp_cell_counts[91]

# <markdowncell>

# Group the features into 10 clusters using k-means. Each cluster is called a _texton_.

# <codecell>

textonmap = kmeans_model.predict(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])
textonmap_rgb = label2rgb(textonmap, image=None, colors=None, alpha=0.3, image_alpha=1)

# cv2.imwrite(img_name+'_textonmap_with_boundary_text.jpg', 
#             img_as_ubyte(.4*textonmap_rgb + .6*img_superpixelized_text))
# Image(img_name+'_textonmap_with_boundary_text.jpg')

cv2.imwrite(img_name+'_textonmap.jpg', 
            img_as_ubyte(textonmap_rgb))
FileLink(img_name+'_textonmap.jpg')

# <markdowncell>

# dirmap as texton map

# <codecell>

im = cv2.imread('/oasis/projects/nsf/csd181/iizhaki/Final/Folder_2026182415/KMean.png')
textonmap = im[:,:,0]/12

# textonmap = dirmap
num_textons = len(np.unique(textonmap))

textonmap_rgb = label2rgb(np.where(mask, textonmap, -1), image=None, colors=None, alpha=0.3, image_alpha=1)
cv2.imwrite(img_name+'_textonmap.jpg',  img_as_ubyte(textonmap_rgb))
FileLink(img_name+'_textonmap.jpg')

# <codecell>

cv2.imwrite(img_name+'_textonmap_superpixelized.jpg',  img_as_ubyte(.5*textonmap_rgb + .5*gray2rgb(superpixel_boundaries)))
FileLink(img_name+'_textonmap_superpixelized.jpg')

# <codecell>

for i in range(num_textons):
    plt.imshow((textonmap == i)*mask, cmap=plt.cm.Greys_r);
    plt.title("dir %d"%i)
    plt.axis('off')
    plt.show()

# <markdowncell>

# Compute the connectivity matrix of the superpixels.

# <codecell>

from skimage.morphology import disk
from skimage.filter.rank import gradient
from scipy.sparse import coo_matrix

edge_map = gradient(segmentation.astype(np.uint8), disk(3))
neighbors = [set() for i in range(n_superpixels)]
for y,x in zip(*np.nonzero(edge_map)):
    neighbors[segmentation[y,x]] |= set(segmentation[y-2:y+2,x-2:x+2].ravel())

# connectivity_matrix = np.zeros((n_segmentation, n_segmentation), dtype=np.bool)
rows = np.hstack([s*np.ones((len(neighbors[s]),), dtype=np.int) for s in range(n_superpixels)])
cols = np.hstack([list(neighbors[s]) for s in range(n_superpixels)])
data = np.ones((cols.size, ), dtype=np.bool)
connectivity_matrix = coo_matrix((data, (rows, cols)), shape=(n_superpixels,n_superpixels))
# plt.matshow(connectivity_matrix.todense(), cmap=plt.cm.Greys_r)
# plt.show()

from skimage.draw import line

superpixel_connectivity_img = img_superpixelized.copy()
for i in range(n_superpixels):
    for neig in neighbors[i]:
        rr, cc = line(int(sp_centroids[i,0]), int(sp_centroids[i,1]),
                      int(sp_centroids[neig,0]), int(sp_centroids[neig,1]))
        superpixel_connectivity_img[rr, cc] = (0,0,1)
        
# cv2.imwrite(img_name+'_superpixel_connectivity_img.jpg', img_as_ubyte(superpixel_connectivity_img))
# FileLink(img_name+'_superpixel_connectivity_img.jpg')

# <markdowncell>

# Compute the texton histogram of each superpixel. Also compute the histogram distance between each pair of superpixels. The distance measure is the chi-square distance.

# <codecell>

sample_interval = 1
gridy, gridx = np.mgrid[:img.shape[0]:sample_interval, :img.shape[1]:sample_interval]

all_seg = segmentation[gridy.ravel(), gridx.ravel()]
all_texton = textonmap[gridy.ravel(), gridx.ravel()]
sp_texton_hist = np.array([np.bincount(all_texton[all_seg == s], minlength=num_textons) 
                 for s in range(n_superpixels)])

row_sums = sp_texton_hist.sum(axis=1)
sp_texton_hist_normalized = sp_texton_hist.astype(np.float) / row_sums[:, np.newaxis]

def chi2(u,v):
    return np.sum(np.where(u+v!=0, (u-v)**2/(u+v), 0))

eps = 0.001
def kl(u,v):
    return np.sum((u+eps)*np.log((u+eps)/(v+eps)))

def kl_no_eps(u,v):
    return np.sum(u*np.log(u/v))

# D = pdist(sp_texton_hist_normalized, chi2)
# hist_distance_matrix = squareform(D)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(hist_distance_matrix)
# fig.colorbar(cax)
# plt.show()

# <codecell>

hist_distance_matrix[7,6]

# <codecell>

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(111)
ax.plot(np.sort(hist_distance_matrix[20]))
plt.xticks(range(n_superpixels), hist_distance_matrix[20].argsort())
plt.show()

# <markdowncell>

# To find salient superpixels, we calculate a saliency map. 
# 
# **First approach:** Use the distance between histogram of each superpixel vs. histogram of the overall image.

# <markdowncell>

# intensity histogram

# <codecell>

intensitymap = sp_mean_intensity[segmentation].astype(np.int)

overall_intensity_hist, _ = np.histogram(intensitymap[mask].flat, bins=range(257))
overall_intensity_hist_normalized = overall_intensity_hist.astype(np.float) / overall_intensity_hist.sum()

from scipy.ndimage.filters import gaussian_filter1d

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].bar(np.arange(256), overall_intensity_hist_normalized)

overall_intensity_hist_normalized_smoothed = gaussian_filter1d(overall_intensity_hist_normalized, .5)

axes[1].bar(np.arange(256), overall_intensity_hist_normalized_smoothed)
plt.show()

individual_saliency_score = np.exp(-overall_intensity_hist_normalized_smoothed[sp_mean_intensity.astype(np.int)])
saliency_score = individual_saliency_score
saliency_score[bg_superpixels] = 0

# sp_diameters = np.array([s.equivalent_diameter for s in sp_props])
# sp_diameter_mean = sp_diameters.mean()

# saliency_score = np.zeros((n_superpixels,))
# for i, sp_hist in enumerate(sp_label_hist_normalized):
#     saliency_score[i] = individual_saliency_score[i]
#     neighbor_term = 0
#     for j in neighbors[i]:
#         if j!=i:
#             neighbor_term += np.exp(-hist_distance_matrix[i,j]/sp_diameter_mean) * individual_saliency_score[j]
#     saliency_score[i] += neighbor_term/(len(neighbors[i])-1)
        
saliency_map = saliency_score[segmentation]
saliency_map[~mask] = saliency_map[mask].min()
plt.matshow(saliency_map, cmap=cm.Greys_r)
plt.colorbar()
plt.show()

# <codecell>

np.unique(saliency_map*mask)

# <markdowncell>

# texton histogram

# <codecell>

overall_texton_hist = np.bincount(textonmap[mask].flat)
overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()

individual_saliency_score = np.array([chi2(sp_hist, overall_texton_hist_normalized) for sp_hist in sp_texton_hist_normalized])

# sp_diameters = np.array([s.equivalent_diameter for s in sp_props])
# sp_diameter_mean = sp_diameters.mean()

saliency_score = np.zeros((n_superpixels,))
for i, sp_hist in enumerate(sp_texton_hist_normalized):
    if i not in bg_superpixels:
        saliency_score[i] = individual_saliency_score[i]
#         neighbor_term = 0
#         c = 0
#         for j in neighbors[i]:
#             if j!=i and j not in bg_superpixels:
#                 neighbor_term += np.exp(-hist_distance_matrix[i,j]) * individual_saliency_score[j]
#                 c += 1
#         saliency_score[i] += neighbor_term/c
        
saliency_map = saliency_score[segmentation]
plt.matshow(saliency_map, cmap=cm.Greys_r)
plt.colorbar()
plt.show()

# <codecell>

plt.bar(np.arange(num_textons)-0.3, sp_texton_hist_normalized[511], alpha=.5, width=0.3, color='r', label='sp')
plt.bar(np.arange(num_textons), overall_texton_hist_normalized, alpha=.5, width=0.3, color='g', label='overall')
plt.xticks(np.arange(num_textons))
plt.legend()
plt.show()

# <markdowncell>

# **Second approach:**
# The saliency of superpixel $i$ is the sum of its difference with all the other superpixels:
# $$S(i) = \sum_{j\ne i}D(i,j).$$ The difference between two superpixels is defined as:
# $$D(i,j) = \exp \left(- \frac{d_{spatial}(i,j)}{a} \right) \cdot d_{texton}(i,j)$$
# This is the texton histogram distance weighted by the spatial distance. $a$ is the average diameter of a superpixel.

# <codecell>

sp_diameters = np.array([s.equivalent_diameter for s in sp_props])
sp_diameter_mean = sp_diameters.mean()

saliency_score = np.zeros((n_superpixels,))
for i in range(n_superpixels):
#     saliency_score[i] = np.sum([np.log(1 + sp_centroid_dist_matrix[i,j] / sp_diameter_mean) + hist_distance_matrix[i,j] 
#                                 for j in range(n_superpixels) if j != i ])
#     saliency_score[i] = np.sum([np.log(1 + sp_centroid_dist_matrix[i,j] / sp_diameter_mean)
#                                 for j in range(n_superpixels) if j != i ])
#     saliency_score[i] = np.sum([1./np.log(1 + sp_centroid_dist_matrix[i,j] / sp_diameter_mean) * hist_distance_matrix[i,j] 
#                             for j in range(n_superpixels) if j != i ])
    saliency_score[i] = np.sum([np.exp(- sp_centroid_dist_matrix[i,j] / sp_diameter_mean) * hist_distance_matrix[i,j] 
                            for j in range(n_superpixels) if j != i ])
#     saliency_score[i] = np.mean([hist_distance_matrix[i,j] 
#                             for j in range(n_superpixels) if j != i ])
#     saliency_score[i] = np.sum([hist_distance_matrix[i,j] 
#                             for j in neighbors[i] if j != i ])

saliency_map = saliency_score[segmentation]
plt.matshow(saliency_map, cmap=cm.Greys_r)
plt.colorbar()
plt.show()

# <markdowncell>

# Both saliency maps look reasonable. 
# 
# Now we find the most salient superpixel, and use it as a seed to find a group of connected superpixels with similar texton histograms.

# <codecell>

neighbors[24]

# <codecell>

chosen_superpixels = set([])

clusters = []

dist_thresh = .5

n_top_clusters = 10

for t in range(1, n_top_clusters+1):
    for i in saliency_score.argsort()[::-1]:
        if i not in chosen_superpixels and i not in bg_superpixels:
            break
            
    print saliency_score[i]
    
    curr_cluster = np.array([i], dtype=np.int)
    frontier = [i]
    while len(frontier) > 0:
    #     print 'frontier', frontier
        i = frontier.pop(-1)
        for j in neighbors[i]:
#             if j==69:
#                 print i, j, zip(curr_cluster, hist_distance_matrix[curr_cluster,j]),\
#                 hist_distance_matrix[curr_cluster,j].mean(), np.median(hist_distance_matrix[curr_cluster,j]), hist_distance_matrix[i,j]
            if j != i and j not in curr_cluster and j not in chosen_superpixels\
            and hist_distance_matrix[i,j] < dist_thresh\
            and j not in bg_superpixels and i not in bg_superpixels:
#             and hist_distance_matrix[curr_cluster,j].max() < dist_thresh:
            
                curr_cluster = np.append(curr_cluster, j)
                frontier.append(j)
    print curr_cluster

    clusters.append(curr_cluster)
    chosen_superpixels |= set(curr_cluster)
  
# print 'cluster', clusters
# propagate_selection = np.equal.outer(segmentation, curr_cluster).any(axis=2)
# selection = mark_boundaries(img, cluster_map)
# cv2.imwrite(img_name+'_selection.jpg', img_as_ubyte(selection))
# FileLink(img_name+'_selection.jpg')

segmentation_copy = np.zeros_like(segmentation)

for i, c in enumerate(clusters):
    propagate_selection = np.equal.outer(segmentation, c).any(axis=2)
    segmentation_copy[propagate_selection] = i + 1

colors = np.random.random((n_superpixels,3))

# colors_hsv = np.zeros((n_top_clusters,3))
# colors_hsv[:,0] = 0.4
# colors_hsv[:,2] = 1
# colors_hsv[:,1] = np.linspace(1./n_top_clusters,1,n_top_clusters)
# colors = hsv2rgb(colors_hsv.reshape((n_top_clusters,1,3)))
# colors = np.tile(np.linspace(0.3,0.8,n_top_clusters), (3,1)).T
# selection_rgb = label2rgb(segmentation_copy, img_superpixelized_text, 
#                           colors=colors.reshape(-1,3))
selection_rgb = label2rgb(segmentation_copy, img_superpixelized_text, 
                          bg_label=0, bg_color=(1,1,1), 
                          colors=colors)
# selection_rgb = label2rgb(segmentation_copy, img_superpixelized_text, 
#                           bg_label=0, bg_color=(1,1,1), 
#                           colors=None)

cv2.imwrite(img_name+'.selection_rgb.jpg', img_as_ubyte(selection_rgb))
Image(img_name+'.selection_rgb.jpg')

# <codecell>

hist_distance_matrix[73,75]

# <codecell>

%run 'boosting.ipynb'

# <codecell>

from sklearn import preprocessing

X = sp_label_hist_normalized
# X_scaled = preprocessing.scale(sp_label_hist_normalized)
Y = -1*np.ones((n_superpixels, ))
# Y[np.r_[clusters[0],clusters[1]]] = 1
Y[clusters[0]] = 1

satPs, satBs, rules, descriptions, weights = ADT(X, Y, n_iter=5)

# import cPickle as pickle
# rule_pkl = open('rules.pkl', 'wb')
# pickle.dump(rules, rule_pkl)

# f = open('rules.pkl', 'rb')
# rules = pickle.load(f)
# f.close()

F = apply_ADT(X, rules)

Ff = np.where(F > 0, +1, -1)
print 'final training error', np.count_nonzero(Ff != Y)/float(len(Y))

# jt, params_t, alpha_t = adaboost_decision_stump(sp_label_hist_normalized, Y, n_iter=10)
# print jt, params_t, alpha_t

# F = apply_classifiers(sp_label_hist_normalized, jt, params_t, alpha_t)

# for i in zip(Y, X_scaled[:,8], np.sign(F)):
#     print i


# <codecell>

from sklearn.svm import SVC
svc = SVC(kernel='linear', gamma=0.7, C=1.0).fit(X, Y)
print svc.coef_
print 'final training error', np.count_nonzero(svc.predict(X) != Y)/float(len(Y))

# <codecell>

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
clf.fit(X, Y)

print clf.estimator_weights_

print 'final training error', np.count_nonzero(svc.predict(X) != Y)/float(len(Y))
# a = clf.estimators_[0]
# a.predict(X)

# <codecell>

from sklearn.externals.six import StringIO  
from sklearn import tree
import pydot
dot_data = StringIO() 
tree.export_graphviz(clf.estimators_[2], out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_jpg('graph.jpg')
FileLink('graph.jpg')

# <codecell>

from sklearn.cluster import Ward


# ward = Ward(n_clusters=5).fit(hist_distance_matrix)
ward = Ward(n_clusters=15, connectivity=connectivity_matrix).fit(hist_distance_matrix)
sp_labels = ward.labels_

new_labels = sp_labels[segmentation]

new_labels_rgb = label2rgb(new_labels, img)
cv2.imwrite(img_name+'_new_labels_rgb.jpg', img_as_ubyte(new_labels_rgb))
FileLink(img_name+'_new_labels_rgb.jpg')

# new_boundaries = mark_boundaries(img, new_labels)
# cv2.imwrite(img_name+'new_boundaries.jpg', img_as_ubyte(new_boundaries))
# FileLink(img_name+'new_boundaries.jpg')

# new_label_boundaries = mark_boundaries(img_rgb, new_labels)
# cv2.imwrite('new_label_boundaries.jpg', img_as_ubyte(new_label_boundaries))
# FileLink('new_label_boundaries.jpg')

# <codecell>

cv2.imwrite(img_name+'_orig.jpg', img_as_ubyte(img))
FileLink(img_name+'_orig.jpg')

# <codecell>

from scipy import ndimage


import skimage
from skimage.color import color_dict

from skimage.segmentation import clear_border
from skimage.filter import threshold_otsu, threshold_adaptive
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import watershed

from skimage.morphology import label

t_img = threshold_adaptive(img, 25, offset=.01)
b_img = binary_erosion(-t_img, np.ones((3, 3)))
d_img = binary_dilation(b_img, np.ones((3, 3)))
clear_border(d_img)
labels, n_labels = label(d_img, neighbors=4, return_num=True)


labelmap = skimage.color.label2rgb(labels, image=None, colors=None, alpha=0.3, \
                                   bg_label=0, bg_color=color_dict['white'], image_alpha=1)


fig = plt.figure()
ax1= fig.add_subplot(111)
p = ax1.imshow(labelmap, cmap=cm.Greys_r)
ax1.set_title('cell segmentation')
plt.show()

# <codecell>

cell_props = regionprops(labels, img, cache=True)

# <codecell>

n_cells = len(labels)

# <codecell>

cell_prop = cell_props[8]
plt.imshow(cell_prop.convex_image, cmap=plt.cm.Greys_r)

from skimage.measure import find_contours
cont = find_contours(cell_prop.convex_image[:,::-1], 0.5)
print len(cont)

fig = plt.figure()
ax = fig.add_subplot(111)
for c in cont:
    ax.plot(c[:,0], c[:,1])
plt.show()

# <codecell>

# from skimage.io import imsave, use_plugin

# use_plugin('pil')
# use_plugin('jpeg')

from skimage import util

cellseg = util.img_as_ubyte(labelmap)
img_cv = util.img_as_ubyte(img)
cell_overlay = cv2.addWeighted(cellseg, 0.1, cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB), 0.9, 1)

cv2.imwrite('cell_overlay.jpg', cell_overlay)
FileLink('cell_overlay.jpg')

# <codecell>

cv2.imwrite('cellseg.jpg', cellseg)
FileLink('cellseg.jpg')

# <codecell>

from sklearn.decomposition import PCA
k = 4
pca = PCA(k)
X = features.reshape(-1, n_selected)
X_pca = pca.fit_transform(X)
# kmeans_model.fit(X_pca)

# <codecell>

cell_features = np.array([(c.centroid[0], c.centroid[1], 
                          c.area, c.orientation, c.eccentricity) 
                          for c in cell_props])

# <codecell>

# plt.hist(areas[areas>100])
# plt.show()
n_cells = cell_features.shape[0]

# <codecell>

from sklearn.neighbors import NearestNeighbors
# X = cell_features[:,:2]
reduced_cell_features = cell_features[np.random.choice(n_cells, 10000),:]
X = reduced_cell_features[:,:2]
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
# distances, indices = nbrs.kneighbors(X)

# <codecell>

nz = nbrs.kneighbors_graph(X).toarray()

# <codecell>

rows, cols = np.nonzero(nz)
# similarity_matrix = np.zeros((n_cells, n_cells, 5), dtype=np.float)
# distance_matrix = np.((n_cells, n_cells, 5), dtype=np.float)
# area_dist_matrix = np.zeros_like((nz), dtype=np.float)
# centroid_dist_matrix = np.zeros_like((nz), dtype=np.float)
# orient_dist_matrix = np.zeros_like((nz), dtype=np.float)
# eccent_dist_matrix = np.zeros_like((nz), dtype=np.float)

from scipy.sparse import coo_matrix

centroid_dist_data = np.sqrt(((reduced_cell_features[rows,:2] - reduced_cell_features[cols,:2])**2).sum(axis=1))
centroid_sim_data = np.exp(-centroid_dist_data/centroid_dist_data.std())

area_dist_data = reduced_cell_features[rows,2] - reduced_cell_features[cols,2]
area_sim_data = np.exp(-area_dist_data/area_dist_data.std())

orient_dist_data = reduced_cell_features[rows,3] - reduced_cell_features[cols,3]
orient_sim_data = np.exp(-orient_dist_data/orient_dist_data.std())

eccent_dist_data = reduced_cell_features[rows,4] - reduced_cell_features[cols,4]
eccent_sim_data = np.exp(-eccent_dist_data/eccent_dist_data.std())

sim_matrix = coo_matrix((centroid_sim_data + 
                        area_sim_data + 
                        orient_sim_data + 
                        eccent_sim_data, (rows, cols)), shape=(10000,10000))


# <codecell>

from sklearn.cluster import spectral_clustering
labels = spectral_clustering(sim_matrix.todense(), n_clusters=8, n_init=1, eigen_solver='amg')

# <codecell>

sim_matrix.todense().shape

# <codecell>

n_cells

# <codecell>

plt.hist(centroid_sim_data)
plt.show()

# <codecell>

labels.shape

# <codecell>

fig = plt.figure(figsize=(10,10), dpi=100)
ax = fig.add_subplot(111)
l = 4
ax.scatter(reduced_cell_features[labels==l, 0], reduced_cell_features[labels==l, 1][::-1], s=1)
plt.show()

