# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from skimage.filter import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.color import color_dict, gray2rgb, label2rgb
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation, binary_erosion, watershed, remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_ubyte
import cv2
import numpy as np
import os, csv

def foreground_mask(img, min_size=64, thresh=200):
    """
    Find the mask that covers exactly the foreground of the brain slice image.
    This depends heavily on the manually chosen threshold, and thus is very fragile.
    It works reasonably well on bright backgrounds, such as blue nissl images; 
    but without tuning the threshold, it does not work on images with dark background, such as fluorescent images.
    
    Parameters
    ----------
    img : image
        input image
    min_size : float
    thresh : float
    
    Return
    ------
    mask : 
        foreground mask
    """
    
#     t_img = gaussian_filter(img, sigma=3) < 220./255.
    t_img = denoise_bilateral(img) < thresh/255.

    labels, n_labels = label(t_img, neighbors=4, return_num=True)
    
    reg = regionprops(labels+1)
    all_areas = np.array([r.area for r in reg])
    
    a = np.concatenate([labels[0,:] ,labels[-1,:] ,labels[:,0] ,labels[:,-1]])
    border_labels = np.unique(a)
    
    border_labels_large = np.extract(all_areas[border_labels] > 250, border_labels)

    mask = np.ones_like(img, dtype=np.bool)
    for i in border_labels_large:
        if i != all_areas.argmax():
            mask[labels==i] = 0

    mask = remove_small_objects(mask, min_size=min_size, connectivity=1, in_place=False)
            
    return mask

from scipy.ndimage import measurements

def crop_image(img, smooth=20):
    blurred = gaussian_filter(img, smooth)
    thresholded = blurred < threshold_otsu(blurred)
    slc = measurements.find_objects(thresholded)[0]

#     margin = 100
#     xstart = max(slc[0].start - margin, 0)
#     xstop = min(slc[0].stop + margin, img.shape[0])
#     ystart = max(slc[1].start - margin, 0)
#     ystop = min(slc[1].stop + margin, img.shape[1])
#     cutout = img[xstart:xstop, ystart:ystop]
    return slc

# <codecell>

import time
 
def timeit(func=None,loops=1,verbose=False):
    if func != None:
        def inner(*args,**kwargs):
 
            sums = 0.0
            mins = 1.7976931348623157e+308
            maxs = 0.0
            print '==== %s ====' % func.__name__
            for i in range(0,loops):
                t0 = time.time()
                result = func(*args,**kwargs)
                dt = time.time() - t0
                mins = dt if dt < mins else mins
                maxs = dt if dt > maxs else maxs
                sums += dt
                if verbose == True:
                    print '\t%r ran in %2.9f sec on run %s' %(func.__name__,dt,i)
            
            if loops == 1:
                print '%r run time was %2.9f sec' % (func.__name__,sums)
            else:
                print '%r min run time was %2.9f sec' % (func.__name__,mins)
                print '%r max run time was %2.9f sec' % (func.__name__,maxs)
                print '%r avg run time was %2.9f sec in %s runs' % (func.__name__,sums/loops,loops)
            
            return result
 
        return inner
    else:
        def partial_inner(func):
            return timeit(func,loops,verbose)
        return partial_inner

# <codecell>

class ParameterSet(object):
    
    def __init__(self, gabor_params_id=None, vq_params_id=None, segm_params_id=None):

        print os.path.join(PARAMS_DIR, 'gabor_' + gabor_params_id + '.json')
 
        self.gabor_params = json.load(open(os.path.join(PARAMS_DIR, 'gabor_' + gabor_params_id + '.json'), 'r')) if gabor_params_id is not None else None

        
        self.segm_params = json.load(open(os.path.join(PARAMS_DIR, 'segm_' + segm_params_id + '.json', 'r')) if segm_params_id is not None else None
        self.vq_params = None
    
    def set_gabor_params(self, gabor_params_id):
        pass
    
    def set_vq_params(self, vq_params_id):
        pass
        
    def set_segm_params(self, segm_params_id):
        pass
        
        
        

# <codecell>

# class DataManager(object):
#     def __init__(self):
#         generate_local_tree()
        
        
#     def generate_local_tree(self):
        

#     def generate_remote_tree(self):
        
        
        
#     def label_elements(self):
        
        
#     def generate_gui_paths(self):
        
        
        

# <codecell>

import json
from pprint import pprint

DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v3'
REPO_DIR = '/home/yuncong/BrainSaliencyDetection'
PARAMS_DIR = os.path.join(REPO_DIR, 'params')

class Instance(object):
    """
    A class for holding various properties of data instances
    """
#     def __init__(self, stack, resol, slice=None, paramset=None):
    def __init__(self, stack, resol, slice=None):
        """
        Construct a processing instance.
        
        Parameters
        ----------
        stack : str
            stack name, e.g. RS141
        resol : str
            resolution string, e.g. x5
        slice : int
            slice number
        paramset : dict
            parameter set. A dict with keys "gabor_params_id", "vq_params_id", "segm_params_id".
            
        """
        
        self.stack = stack
        self.resol = resol        
        self.image = None
        self.slice = None
#         self.paramset = None

        if slice is not None:
            self.set_slice(slice)

#         if paramset is not None:
#             self.set_paramset(paramset)
            
    def set_slice(self, slice_num):
        
        self.slice = slice_num
        self.image_dir = os.path.join(DATA_DIR, self.stack, self.resol, '%04d' % self.slice)
        self.image_name = '_'.join([self.stack, self.resol, '%04d' % self.slice])
        self.instance_name = self.image_name
        
#         if self.paramset_name is not None:
#             self.results_dir = os.path.join(self.image_dir, self.paramset_name + '_pipelineResults')
#             self.instance_name = '_'.join([self.stack, self.resol, '%04d' % self.slice, self.paramset_name])
        
#     def set_paramset(self, paramset_name):

#         self.paramset_name = paramset_name
        
#         if self.slice is not None:
#             self.results_dir = os.path.join(self.image_dir, paramset_name + '_pipelineResults')
#             self.instance_name = '_'.join([self.stack, self.resol, '%04d' % self.slice, paramset_name])

#         self.paramset = load_paramset(paramset_name)
    
    def load_image(self):
        
        if self.image is None:
            
            image_filename = os.path.join(self.image_dir, self.image_name + '.tif')
            assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')

            im = cv2.imread(image_filename, 0)
            self.image = regulate_image(im)
        
        return self.image

    def load_pipeline_result(self, result_name, ext):
        
        result_filename = os.path.join(self.results_dir, self.instance_name + '_' + result_name + '.' + ext)

        if ext == 'npy':
            assert os.path.exists(result_filename), "Pipeline result '%s' does not exist" % (result_name + '.' + ext)
            res = np.load(result_filename)
        elif ext == 'tif' or ext == 'png':
            res = cv2.imread(filename, 0)
            res = regulate_image(res)
            
        print 'loaded %s' % result_filename

        return res

    def save_pipeline_result(self, data, result_name, ext):

        result_filename = os.path.join(self.results_dir, self.instance_name + '_' + result_name + '.' + ext)

        if ext == 'npy':
            res = np.save(result_filename, data)
        elif ext == 'tif' or ext == 'png':
            res = regulate_image(res)
            res = cv2.imwrite(result_filename, res)
            
        print 'saved %s' % result_filename
    

# <codecell>

# import json
# from pprint import pprint

# DATA_DIR = '/home/yuncong/BrainLocal/DavidData_v3'
# REPO_DIR = '/home/yuncong/BrainSaliencyDetection'
# PARAMS_DIR = os.path.join(REPO_DIR, 'params')

# class Instance(object):
#     """
#     A class for holding various properties of data instances
#     """
#     def __init__(self, stack, resol, slice=None, paramset=None):
#         self.stack = stack
#         self.resol = resol        
#         self.image = None
#         self.slice = None
#         self.paramset_name = None

#         if slice is not None:
#             self.set_slice(slice)

#         if paramset is not None:
#             self.set_paramset(paramset)
            
#     def set_slice(self, slice_num):
        
#         self.slice = slice_num
#         self.image_dir = os.path.join(DATA_DIR, self.stack, self.resol, '%04d' % self.slice)
#         self.image_name = '_'.join([self.stack, self.resol, '%04d' % self.slice])
        
#         if self.paramset_name is not None:
#             self.results_dir = os.path.join(self.image_dir, self.paramset_name + '_pipelineResults')
#             self.instance_name = '_'.join([self.stack, self.resol, '%04d' % self.slice, self.paramset_name])
        
#     def set_paramset(self, paramset_name):

#         self.paramset_name = paramset_name
        
#         if self.slice is not None:
#             self.results_dir = os.path.join(self.image_dir, paramset_name + '_pipelineResults')
#             self.instance_name = '_'.join([self.stack, self.resol, '%04d' % self.slice, paramset_name])

#         self.paramset = load_paramset(paramset_name)
    
#     def load_image(self):
        
#         if self.image is None:
            
#             image_filename = os.path.join(self.image_dir, self.image_name + '.tif')
#             assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')

#             im = cv2.imread(image_filename, 0)
#             self.image = regulate_image(im)
        
#         return self.image

#     def load_pipeline_result(self, result_name, ext):
        
#         result_filename = os.path.join(self.results_dir, self.instance_name + '_' + result_name + '.' + ext)

#         if ext == 'npy':
#             assert os.path.exists(result_filename), "Pipeline result '%s' does not exist" % (result_name + '.' + ext)
#             res = np.load(result_filename)
#         elif ext == 'tif' or ext == 'png':
#             res = cv2.imread(filename, 0)
#             res = regulate_image(res)
            
#         print 'loaded %s' % result_filename

#         return res

#     def save_pipeline_result(self, data, result_name, ext):

#         result_filename = os.path.join(self.results_dir, self.instance_name + '_' + result_name + '.' + ext)

#         if ext == 'npy':
#             res = np.save(result_filename, data)
#         elif ext == 'tif' or ext == 'png':
#             res = regulate_image(res)
#             res = cv2.imwrite(result_filename, res)
            
#         print 'saved %s' % result_filename
    

# <codecell>

def load_paramset(paramset_name):

    params_dir = os.path.realpath(PARAMS_DIR)
    param_file = os.path.join(params_dir, 'param_%s.json' % paramset_name)
    param_default_file = os.path.join(params_dir, 'param_default.json')
    param = json.load(open(param_file, 'r'))
    param_default = json.load(open(param_default_file, 'r'))

    for k, v in param_default.iteritems():
        if not isinstance(param[k], basestring):
            if np.isnan(param[k]):
                param[k] = v
                
    return param
    

def regulate_images(imgs):
    """
    Ensure all images are of type RGB uint8.
    """
    
    return np.array(map(regulate_image, imgs))
        
    
def regulate_image(img):
    """
    Ensure the image is of type RGB uint8.
    """
    
    if not np.issubsctype(img, np.uint8):
        try:
            img = img_as_ubyte(img)
        except:
            img_norm = (img-img.min()).astype(np.float)/(img.max() - img.min())    
            img = img_as_ubyte(img_norm)
            
    if img.ndim == 2:
        img = gray2rgb(img)
        
    return img

# <codecell>

# def visualize_cluster(scores, cluster='all', title='', filename=None):
#     '''
#     Generate black and white image with the cluster of superpixels highlighted
#     '''
    
#     vis = scores[segmentation]
#     if cluster != 'all':
#         cluster_selection = np.equal.outer(segmentation, cluster).any(axis=2)
#         vis[~cluster_selection] = 0
    
#     plt.matshow(vis, cmap=plt.cm.Greys_r);
#     plt.axis('off');
#     plt.title(title)
#     if filename is not None:
#         plt.savefig(os.path.join(result_dir, 'stages', filename + '.png'), bbox_inches='tight')
# #     plt.show()
#     plt.close();
    

def paint_superpixels_on_image(superpixels, segmentation, img):
    '''
    Highlight a cluster of superpixels on the real image
    '''    

    cluster_map = -1*np.ones_like(segmentation)
    for s in superpixels:
        cluster_map[segmentation==s] = 1
    vis = label2rgb(cluster_map, image=img)
    return vis
    
def paint_superpixel_groups_on_image(sp_groups, segmentation, img, colors):
    '''
    Highlight multiple superpixel groups with different colors on the real image
    '''
    
    cluster_map = -1*np.ones_like(segmentation)
    for i, sp_group in enumerate(sp_groups):
        for j in sp_group:
            cluster_map[segmentation==j] = i
    vis = label2rgb(cluster_map, image=img, colors=colors)
    return vis

# <codecell>

def chi2(u,v):
    """
    Compute Chi^2 distance between two distributions.
    
    Empty bins are ignored.
    
    """
    
    r = np.nansum((u-v)**2/(u+v))
    return r

# <codecell>

# from mpl_toolkits.axes_grid1 import ImageGrid

# def image_grid(images, ):
#     ncols = 12
#     nrows = n_images/ncols+1

#     fig = plt.figure(1, figsize=(20., 20./ncols*nrows))
#     grid = ImageGrid(fig, 111, # similar to subplot(111)
#                     nrows_ncols = (nrows, ncols), # creates 2x2 grid of axes
#                     axes_pad=0.1, # pad between axes in inch.
#                     )

#     for i in bbox.iterkeys():
#         y1, x1, y2, x2 = bbox[i]
#         grid[i].imshow(images[i][y1:y2, x1:x2], cmap=plt.cm.Greys_r, aspect='auto');
#         grid[i].set_title(i)
#         grid[i].axis('off')

#     plt.show()

