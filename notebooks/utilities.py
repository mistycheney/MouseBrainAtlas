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
    Find the mask that covers exactly the foreground of the brain slice image
    
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

# from copy_reg import pickle
# from types import MethodType

# def _pickle_method(method):
#     """
#     Pickle methods properly, including class methods.
#     """
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     if isinstance(cls, type):
#         # handle classmethods differently
#         cls = obj
#         obj = None
#     if func_name.startswith('__') and not func_name.endswith('__'):
#         #deal with mangled names
#         cls_name = cls.__name__.lstrip('_')
#         func_name = '_%s%s' % (cls_name, func_name)

#     return _unpickle_method, (func_name, obj, cls)

# def _unpickle_method(func_name, obj, cls):
#     """
#     Unpickle methods properly, including class methods.
#     """
#     if obj is None:
#         return cls.__dict__[func_name].__get__(obj, cls)
#     for cls in cls.__mro__:
#         try:
#             func = cls.__dict__[func_name]
#         except KeyError:
#             pass
#         else:
#             break
#     return func.__get__(obj, cls)

# pickle(MethodType, _pickle_method, _unpickle_method)

# <codecell>

# import copy_reg
# import types

# def _pickle_method(method):
#     func_name = method.im_func.__name__
#     obj = method.im_self
#     cls = method.im_class
#     if func_name.startswith('__') and not func_name.endswith('__'):
#         #deal with mangled names
#         cls_name = cls.__name__.lstrip('_')
#         func_name = '_%s%s' % (cls_name, func_name)
#     return _unpickle_method, (func_name, obj, cls)

# def _unpickle_method(func_name, obj, cls):
#     if obj and func_name in obj.__dict__:
#         cls, obj = obj, None # if func_name is classmethod
#     for cls in cls.__mro__:
#         try:
#             func = cls.__dict__[func_name]
#         except KeyError:
#             pass
#         else:
#             break
#     return func.__get__(obj, cls)

# copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

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

def load_array(suffix, instance_name, results_dir):
    """
    Load the cached result.
    
    Parameters
    ----------
    suffix : str
        name of the result
    instance_name : str
        instance name
    results_dir : str
        the location of results
    
    Returns
    -------
    arr : array
        the loaded array    
    """
    
    arr_file = os.path.join(results_dir, '%s_%s.npy'%(instance_name, suffix))
    arr = np.load(arr_file)
    print 'load %s' % (arr_file)
    return arr

def save_array(arr, suffix, instance_name, results_dir, overwrite=True):
    """
    Save array
    
    Parameters
    ----------
    arr : array
        the array to cache
    suffix : str
        name of the result
    instance_name : str
        instance name
    results_dir : str
        results directory
    overwrite : bool
        if True, overwrite existing file
    """
    
    arr_file = os.path.join(results_dir, '%s_%s.npy'%(instance_name, suffix))
    if not os.path.exists(arr_file) or overwrite:
        np.save(arr_file, arr)
        print '%s saved to %s' % (suffix, arr_file)
    else:
        print '%s already exists' % (arr_file)

# <codecell>



# def load_array(suffix, img_name, param_id, output_dir):
#     """
#     Load the cached result.
    
#     Parameters
#     ----------
#     suffix : str
#         name of the result
#     img_name : str
#         image name
#     param_id : str
#         parameter set name
#     output_dir : str
#         the location of results
    
#     Returns
#     -------
#     arr : array
#         the loaded array    
#     """
    
#     result_name = img_name + '_' + str(param_id)
#     arr_file = os.path.join(output_dir, result_name, '%s_%s.npy'%(result_name, suffix))
#     arr = np.load(arr_file)
#     print 'load %s' % (arr_file)
#     return arr




# def save_array(arr, suffix, img_name, param_id, cache_dir='scratch', overwrite=True):
#     """
#     Save array into a file
    
#     Parameters
#     ----------
#     arr : array
#         the array to cache
#     suffix : str
#         name of the result
#     img_name : str
#         image name
#     param_id : str
#         parameter set name
#     cache_dir : str
#         output folder
#     overwrite : bool
#         if True, overwrite existing file
#     """
    
#     result_name = img_name + '_' + str(param_id)
#     arr_file = os.path.join(cache_dir, result_name, '%s_%s.npy'%(result_name, suffix))
    
#     if not os.path.exists(arr_file) or overwrite:
#         np.save(arr_file, arr)
#         print '%s saved to %s' % (suffix, arr_file)
#     else:
#         print '%s already exists' % (arr_file)
        
def regulate_images(imgs):
    return np.array(map(regulate_img, imgs))
        
def regulate_img(img):
    '''
    Make all images be RGB uint8.
    '''
    
    if not np.issubsctype(img, np.uint8):
        try:
            img = img_as_ubyte(img)
        except:
            img_norm = (img-img.min()).astype(np.float)/(img.max() - img.min())    
            img = img_as_ubyte(img_norm)
            
    if img.ndim == 2:
        img = gray2rgb(img)
        
    return img
        
    
def save_image(img, suffix, instance_name, results_dir, ext='tif', overwrite=True):
    '''
    Save image to file
    
    Parameters
    ----------
    img : uint8 or float array
        image to save
    suffix : str
        name of the result
    instance_name : str
        instance name
    results_dir : str
        results directory
    overwrite : bool
        if True, overwrite existing file
    '''
    
    img = regulate_img(img)
        
    img_fn = os.path.join(results_dir, '%s_%s.%s'%(instance_name, suffix, ext))
#     img_fn = get_img_filename(suffix, img_name, param_id, cache_dir, ext='tif')
    if not os.path.exists(img_fn) or overwrite:
        cv2.imwrite(img_fn, img)
        print '%s saved to %s' % (suffix, img_fn)
    else:
        print '%s already exists' % (img_fn)
        
#     img_fn = os.path.join(results_dir, '%s_%s.png'%(instance_name, suffix))
# #     img_fn = get_img_filename(suffix, img_name, param_id, cache_dir, ext='png')
#     if not os.path.exists(img_fn) or overwrite:
#         cv2.imwrite(img_fn, img)
#         print '%s saved to %s' % (suffix, img_fn)
#     else:
#         print '%s already exists' % (img_fn)

        
def load_image(suffix, instance_name, results_dir):
    '''
    Load image
    
    Parameters
    ----------
    suffix : str
        name of the result
    instance_name : str
        instance name
    results_dir : str
        results directory
        
    Returns
    -------
    img : uint8 RGB
        image
    '''
    
    img_fn = os.path.join(results_dir, '%s_%s.tif'%(instance_name, suffix))
    img = cv2.imread(img_fn)
    reg_img = regulate_img(img)
    return reg_img
    
    
# def save_img(img, suffix, img_name, param_id, 
#              cache_dir='scratch', overwrite=True):
#     '''
#     Save image to file
    
#     Parameters
#     ----------
#     img : uint8 or float array
#         image to save
#     suffix : str
#         name of the result
#     img_name : str
#         image name
#     param_id : str
#         parameter set name
#     cache_dir : str
#         output_folder
#     overwrite : bool
#         if True, overwrite existing file
#     '''
#     img = regulate_img(img)
        
#     img_fn = get_img_filename(suffix, img_name, param_id, cache_dir, ext='tif')
#     if not os.path.exists(img_fn) or overwrite:
#         cv2.imwrite(img_fn, img)
#         print '%s saved to %s' % (suffix, img_fn)
#     else:
#         print '%s already exists' % (img_fn)
        
#     img_fn = get_img_filename(suffix, img_name, param_id, cache_dir, ext='png')
#     if not os.path.exists(img_fn) or overwrite:
#         cv2.imwrite(img_fn, img)
#         print '%s saved to %s' % (suffix, img_fn)
#     else:
#         print '%s already exists' % (img_fn)

# def get_img_filename(suffix, img_name, param_id, cache_dir, ext='tif'):
#     result_name = img_name + '_param_' + str(param_id)
#     img_fn = os.path.join(cache_dir, result_name, '%s_%s.%s'%(result_name, suffix, ext))
#     return img_fn

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

# def load_parameters(params_file):

#     parameters = dict([])

#     with open(params_file, 'r') as f:
#         param_reader = csv.DictReader(f)
#         for param in param_reader:
#             for k in param.iterkeys():
#                 if param[k] != '':
#                     try:
#                         param[k] = int(param[k])
#                     except ValueError:
#                         param[k] = float(param[k])
#             if param['param_id'] == 0:
#                 default_param = param
#             else:
#                 for k, v in param.iteritems():
#                     if v == '':
#                         param[k] = default_param[k]
#             parameters[param['param_id']] = param
            
#     return parameters


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

