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

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    slice_ind = 1
    gabor_params_id = 'blueNisslWide'
    segm_params_id = 'blueNissl'
    
dm.set_image(args.stack_name, args.resolution, args.slice_ind)
dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
dm.set_segmentation_params(segm_params_id=args.segm_params_id)

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

biases = np.array([k.sum() for k in kernels])
mean_bias = biases.mean()
kernels = [k/k.sum()*mean_bias for k in kernels]

# dm.save_pipeline_result(kernels, 'kernels', 'pkl')

def crop_borders(data):
    cropped_data = data[max_kern_size/2:-max_kern_size/2, max_kern_size/2:-max_kern_size/2, ...].copy()
    return cropped_data

# crop borders

# try:
#     cropped_features = dm.load_pipeline_result('cropFeatures', 'npy')
# except:
#     cropped_features = crop_borders(features)
#     dm.save_pipeline_result(cropped_features, 'cropFeatures', 'npy')

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

from scipy.ndimage.filters import gaussian_filter

patch = cropped_img[1000:1050, 1000:1050]

plt.imshow(patch, cmap=plt.cm.Greys_r)
plt.show()

# <codecell>

# Gaussian derivatives

center = np.array([25, 25])
pulse = np.zeros((2*center+1))
pulse[center[0], center[1]] = 1.
weights = gaussian_filter(pulse, sigma=5., order=[1, 0])

def crop(x):

    rs, cs = where(x != 0)
    rmin = rs.min()
    rmax = rs.max()
    cmin = cs.min()
    cmax = cs.max()
    
    return x[rmin:rmax+1, cmin:cmax+1].copy()

weights = crop(weights)

plt.matshow(weights, cmap=cm.Greys_r)

from scipy.ndimage.interpolation import rotate

weights_rotated = rotate(weights, 45)
weights_rotated = crop(weights_rotated)

plt.matshow(weights_rotated, cmap=cm.Greys_r)
plt.show()

# <codecell>

Gxx = np.zeros((50,50))
gaussian_filter(patch, sigma=.5, order=[0,1], output=Gxx)
plt.imshow(Gxx, cmap=plt.cm.Greys_r)
plt.show()

# <codecell>

# Local Binary Pattern

from skimage.feature import local_binary_pattern

radius = 3
n_points = 8 * radius
METHOD = 'uniform'

lbp = local_binary_pattern(patch, n_points, radius, METHOD)

# <codecell>

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')
        
def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
plt.gray()

titles = ('edge', 'flat', 'corner')
w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(patch, lbp, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, lbp)
    highlight_bars(bars, labels)
    ax.set_ylim(ymax=np.max(counts[:-1]))
    ax.set_xlim(xmax=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')

# <codecell>


