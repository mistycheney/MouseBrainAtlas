# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob, re, os, sys, subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage.filter import threshold_otsu, gaussian_filter
from skimage.measure import regionprops, label

# <codecell>

w_reduce2, h_reduce2 = [6000, 4500]

region = [dict([]) for i in range(3)]

margin_border_effect = 200

region[0]['stack'] = 'PMD1305'
region[0]['start_frame'] = 240
region[0]['stop_frame'] = 250
# min_row, min_col, height, width
region[0]['bbox'] = [(2600.-margin_border_effect)/h_reduce2, (2040.-margin_border_effect)/w_reduce2, 
                     (800.+2*margin_border_effect)/h_reduce2, (2300.+2*margin_border_effect)/w_reduce2]

region[1]['stack'] = 'PMD1305'
region[1]['start_frame'] = 159
region[1]['stop_frame'] = 176
region[1]['bbox'] = [(1550.-margin_border_effect)/h_reduce2, (2320.-margin_border_effect)/w_reduce2, 
                     (2280.+2*margin_border_effect)/h_reduce2, (1570.+2*margin_border_effect)/w_reduce2]

region[2]['stack'] = 'PMD1305'
region[2]['start_frame'] = 1
region[2]['stop_frame'] = 20
region[2]['bbox'] = [(2100.-margin_border_effect)/h_reduce2, (2100.-margin_border_effect)/w_reduce2, 
                     (1200.+2*margin_border_effect)/h_reduce2, (820.+2*margin_border_effect)/w_reduce2]

# <codecell>

NB_DIR = '/home/yuncong/Brain/'
CACHE_DIR = '/home/yuncong/my_csd181_scratch/'

stack_name = 'PMD1305'
DATA_DIR = '/home/yuncong/ParthaData/'
IMG_DIR = DATA_DIR+'/'+stack_name

# <codecell>

region[2]

# <codecell>

# uncompress tarball, if not having done so

os.chdir(DATA_DIR)
if not os.path.exists(stack_name):
    if os.path.exists(stack_name+'.tar.gz'):
        return_code = subprocess.call('tar xfz %s.tar.gz'%stack_name, shell=True)
    elif os.path.exists(stack_name+'.tar'):
        return_code = subprocess.call('tar xf %s.tar'%stack_name, shell=True)

# <codecell>

# find bounding box for all images

margin = 0 # at lowest level whole frame is 120 by 90

bboxes = []

bbox = dict([])
images = dict([])
os.chdir(IMG_DIR)
for fn in glob.iglob('*.jpg'):
    m = re.match('.*_([0-9]{4}).jpg', fn)
    idx = int(m.groups()[0])
    img = cv2.imread(fn, 0)
    blurred = gaussian_filter(img, 2)
    thresholded = blurred < threshold_otsu(blurred) + 10./255.
    labelmap, n_components = label(thresholded, background=0, return_num=True)    
    component_props = skimage.measure.regionprops(labelmap+1, cache=True)
    major_comp_prop = sorted(component_props, key=lambda x: x.area, reverse=True)[0]
    y1, x1, y2, x2 = major_comp_prop.bbox
    
    # bbox is <top>,<left>,<height>,<width> in range [0,1]
    bbox[idx] = [float(y1)/img.shape[0], float(x1)/img.shape[1], 
                 float(y2-y1)/img.shape[0], float(x2-x1)/img.shape[1]]
    images[idx] = img
    
n_images = np.max(bbox.keys())

# <codecell>

from mpl_toolkits.axes_grid1 import ImageGrid
from IPython.display import Image, FileLink

ncols = 12
nrows = n_images/ncols+1

fig = plt.figure(1, figsize=(50., 50./ncols*nrows))
grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (nrows, ncols), # creates 2x2 grid of axes
                axes_pad=0.1, # pad between axes in inch.
                )

for i in bbox.iterkeys():
    y1, x1, bbh, bbw = bbox[i]
    img = images[i]
    h, w = img.shape
    grid[i].imshow(img[int(y1*h):int((y1+bbh)*h), int(x1*w):int((x1+bbw)*w)], cmap=plt.cm.Greys_r, aspect='auto');
    grid[i].set_title(i)
    grid[i].axis('off')

os.chdir(NB_DIR)
plt.savefig('overview.png', bbox_inches='tight')
plt.close();

FileLink('overview.png')

# import matplotlib.gridspec as gridspec

# fig = plt.figure(figsize=(20,5*(n_images/6+1)))
# gs = gridspec.GridSpec(nrows=n_images/6+1, ncols=6, wspace=0.0)
# ax = [plt.subplot(gs[i]) for i in range(n_images)]
# gs.update(hspace=0)
# #gs.tight_layout(fig, h_pad=0,w_pad=0)

# for i in bbox.iterkeys():
#     y1, x1, y2, x2 = bbox[i]
#     ax[i].imshow(images[i][y1:y2, x1:x2], cmap=plt.cm.Greys_r, aspect='auto');
#     ax[i].set_title(i)
#     ax[i].axis('off')
    
# plt.show()

# <codecell>

# extract the bounding box at specific reduce level

os.chdir(IMG_DIR)
level = 0
output_dir = '../'+stack_name+'_reduce%d_region0'%level

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
for jp2 in glob.iglob('*.jp2'):
    m = re.match('.*_([0-9]{4}).jp2', jp2)
    idx = int(m.groups()[0])
    tif = '%s_%d_reduce%d_region0.tif'%(stack_name, idx, level)
#     tif = re.sub('.*_([0-9]{4}).jp2$', stack_name+r'_\1_reduce%d.tif'%(level), jp2)
#     top,left,height,width = bbox[idx]
    if idx < region[0]['start_frame'] or idx > region[0]['stop_frame']: continue
    top,left,height,width = region[0]['bbox']
    return_code = subprocess.call("kdu_expand -i '%s' -o '%s' -reduce %d -region '{%f,%f},{%f,%f}'"%(jp2, output_dir+'/'+tif, level,
                                                                                  top,left,height,width), shell=True)  

