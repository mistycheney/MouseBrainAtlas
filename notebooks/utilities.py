# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from skimage.filter import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.color import color_dict, gray2rgb, label2rgb, rgb2gray
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation, binary_erosion, watershed, remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_ubyte
import cv2
import numpy as np
import os, csv


def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=5, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow 
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head 
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head 
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    
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

# import tables

# class DataManager(object):

    
#     def __init__(self):
#         complevel = 5
#         filters = tables.Filters(complevel=complevel, complib='blosc')
#         # h5file = tables.open_file("gabor_files.h5", mode = "w")
#         self.h5file = tables.open_file("results.h5", mode = "a", title = "Pipeline Results", filters=filters)
    
#         self.stack_name = 'RS141'
#         self.resolution = 'x5'
#         self.slice_num = 1
        
#         class Result(tables.IsDescription):
#             name = StringCol()
        
#         self.stack_group = self.h5file.create_group('/', 'RS141', 'RS141')
#         self.resol_group = self.h5file.create_group(self.stack_group, 'x5', 'x5')
#         self.slice_group = self.h5file.create_group(self.resol_group, 'slice001', 'slice001')

#     def save_result(self, data, name):
        
#         self.h5file.create_carray(self.slice_group, name, obj=data)

# <codecell>

# REGENERATE_ALL_RESULTS = True
REGENERATE_ALL_RESULTS = False

import json
import cPickle as pickle

class DataManager(object):

    def __init__(self, data_dir, repo_dir):
        self.data_dir = data_dir
        self.repo_dir = repo_dir
        self.params_dir = os.path.join(repo_dir, 'params')

        self.image_name = None
        
    def set_stack(self, stack, resol):
        self.stack = stack
        self.resol = resol
        self.resol_dir = os.path.join(self.data_dir, self.stack, self.resol)
        
    def set_slice(self, slice_ind):
        assert self.stack is not None and self.resol is not None, 'Stack is not specified'
        self.slice_ind = slice_ind
        self.slice_str = '%04d' % slice_ind
        self.image_dir = os.path.join(self.data_dir, self.stack, self.resol, self.slice_str)
        self.image_name = '_'.join([self.stack, self.resol, self.slice_str])

        self.labelings_dir = os.path.join(self.image_dir, 'labelings')
        
        self.results_dir = os.path.join(self.image_dir, 'pipelineResults')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

    def set_image(self, stack, resol, slice_ind):
        self.set_stack(stack, resol)
        self.set_slice(slice_ind)
        self._load_image()
        
    def _load_image(self):
        
        assert self.image_name is not None, 'Image is not specified'

        image_filename = os.path.join(self.image_dir, self.image_name + '.tif')
        assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')

        self.image = cv2.imread(image_filename, 0)
        self.image_height, self.image_width = self.image.shape[:2]

        mask_filename = os.path.join(self.image_dir, self.image_name + '_mask.png')
        self.mask = cv2.imread(mask_filename, 0) > 0
        
    def set_gabor_params(self, gabor_params_id):
        
        self.gabor_params_id = gabor_params_id
        self.gabor_params = json.load(open(os.path.join(self.params_dir, 'gabor', 'gabor_' + gabor_params_id + '.json'), 'r')) if gabor_params_id is not None else None
        
    def set_segmentation_params(self, segm_params_id):
        
        self.segm_params_id = segm_params_id
        self.segm_params = json.load(open(os.path.join(self.params_dir, 'segm', 'segm_' + segm_params_id + '.json'), 'r')) if segm_params_id is not None else None

    def set_vq_params(self, vq_params_id):
        
        self.vq_params_id = vq_params_id
        self.vq_params = json.load(open(os.path.join(self.params_dir, 'vq', 'vq_' + vq_params_id + '.json'), 'r')) if vq_params_id is not None else None
        
            
    def _get_result_filename(self, result_name, ext, results_dir=None, param_dependencies=None):

        results_dir = self.results_dir
        
        if result_name in ['features', 'kernels', 'features_rotated', 'features_rotated_pca']:
            param_dependencies = ['gabor']

        elif result_name in['segmentation', 'segmentationWithText', 'spProps', 'neighbors']:
            param_dependencies = ['segm']
                        
        elif result_name in ['dirMap', 'dirHist']:
            param_dependencies = ['gabor', 'segm']
            
        elif result_name == 'textons':
            param_dependencies = ['gabor', 'vq']
            
        elif result_name == 'texMap':
            param_dependencies = ['gabor', 'vq']

        elif result_name in ['texHist', 'clusters']:
            param_dependencies = ['gabor', 'segm', 'vq']
            
        # elif result_name == 'tmp':
        #     results_dir = '/tmp'
        #     instance_name = 'test'
        
        # elif result_name == 'models':
        #     results_dir = self.resol_dir
        #     instance_name = '_'.join([self.stack, self.resol,
        #                               'gabor-' + self.gabor_params_id + '-segm-' + self.segm_params_id + \
        #                               '-vq-' + self.vq_params_id])
        
        else:
            raise Exception('result name %s unknown' % result_name)

        # instance_name = self.image_name

        param_strs = []
        if 'gabor' in param_dependencies:
            param_strs.append('gabor-' + self.gabor_params_id)
        if 'segm' in param_dependencies:
            param_strs.append('segm-' + self.segm_params_id)
        if 'vq' in param_dependencies:
            param_strs.append('vq-' + self.vq_params_id)
            # raise Exception("parameter dependency string not recognized")

        result_filename = os.path.join(results_dir, self.image_name + '_' + '-'.join(param_strs) + '_' + result_name + '.' + ext)
        
        return result_filename
            
    def load_pipeline_result(self, result_name, ext, is_rgb=None):
        
        if REGENERATE_ALL_RESULTS:
            raise
        
        result_filename = self._get_result_filename(result_name, ext)
        print result_filename

        if ext == 'npy':
            assert os.path.exists(result_filename), "Pipeline result '%s' does not exist" % (result_name + '.' + ext)
            data = np.load(result_filename)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = cv2.imread(result_filename)
            data = self._regulate_image(data, is_rgb)
        elif ext == 'pkl':
            data = pickle.load(open(result_filename, 'r'))

        print 'loaded %s' % result_filename

        return data
        
    def save_pipeline_result(self, data, result_name, ext, is_rgb=None):
            
        result_filename = self._get_result_filename(result_name, ext)

        if ext == 'npy':
            np.save(result_filename, data)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = self._regulate_image(data, is_rgb)
            if data.ndim == 3:
                cv2.imwrite(result_filename, data[..., ::-1])
            else:
                cv2.imwrite(result_filename, data)
        elif ext == 'pkl':
            pickle.dump(data, open(result_filename, 'w'))
            
        print 'saved %s' % result_filename
        
    def _regulate_image(self, img, is_rgb=None):
        """
        Ensure the image is of type uint8.
        """

        if not np.issubsctype(img, np.uint8):
            try:
                img = img_as_ubyte(img)
            except:
                img_norm = (img-img.min()).astype(np.float)/(img.max() - img.min())    
                img = img_as_ubyte(img_norm)

        if is_rgb is not None:
            if img.ndim == 2 and is_rgb:
                img = gray2rgb(img)
            elif img.ndim == 3 and not is_rgb:
                img = rgb2gray(img)

        return img
    
    
    def load_labeling(self, labeling_name):
        labeling_fn = os.path.join(self.labelings_dir, self.image_name + '_' + labeling_name + '.pkl')
        labeling = pickle.load(open(labeling_fn, 'r'))
        return labeling

# <codecell>

def display(vis, filename='tmp.jpg'):
    
    if vis.dtype != np.uint8:
        if vis.ndim == 3:
            cv2.imwrite(filename, img_as_ubyte(vis)[..., ::-1])
        else:
            cv2.imwrite(filename, img_as_ubyte(vis))
    else:
        if vis.ndim == 3:
            cv2.imwrite(filename, vis[..., ::-1])
        else:
            cv2.imwrite(filename, vis)
    from IPython.display import FileLink
    return FileLink(filename)

# <codecell>

import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background

    return detected_peaks

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

