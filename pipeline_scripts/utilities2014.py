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
from skimage.io import imread, imsave
import numpy as np
import os
import csv
import sys
from operator import itemgetter
import json
import cPickle as pickle

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=5, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    import cv2

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
        input grey image
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

def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

def generate_json(data_dir, res_dir, labeling_dir):
    """
    Return a JSON file that represents the hierarchy of input directory with semantics of our application
    """

    data_hierarchy = get_directory_structure(data_dir).values()[0]

    result_hierarchy = get_directory_structure(res_dir).values()[0]
    labeling_hierarchy = get_directory_structure(labeling_dir).values()[0]

    dataset = {'available_stack_names': [],
        'stacks': []
        }

    for stack_name, stack_content in data_hierarchy.items():
        if stack_content is None or len(stack_content) == 0: continue
        
        stack_info = {'name': stack_name,
                    'available_resolution': stack_content.keys(),
                    'section_num': None
                    }

        sec_infos = []

        # print stack_content
        # print stack_info

        if len(stack_info['available_resolution']) > 0:
            sec_items = stack_content[stack_info['available_resolution'][0]].items()

        for sec_str, sec_content in sec_items:

            if sec_content is None: 
                continue
            
            # stack_info['available_sections'].append(int(sec_ind))
            sec_info = {'index': int(sec_str)}

            for resolution in stack_info['available_resolution']:
                sec_info[resolution + '_imagepath'] = os.path.join(data_dir, stack_name, resolution, sec_str, 
                                                        '_'.join([stack_name, resolution, sec_str])+'.tif')
                sec_info[resolution + '_maskpath'] = os.path.join(data_dir, stack_name, resolution, sec_str, 
                                                        '_'.join([stack_name, resolution, sec_str])+'_mask.png')

            if stack_name in labeling_hierarchy:
                if sec_str in labeling_hierarchy[stack_name]:
                    labeling_list = labeling_hierarchy[stack_name][sec_str]
                    sec_info['labeling_num'] = len(labeling_list)
                    if len(labeling_list) > 0:
                        sec_info['labelings'] = [{'filename':k} for k in labeling_list.keys() if k.endswith('pkl')]
                        for i, l in enumerate(sec_info['labelings']):
                            labeling_path = os.path.join(labeling_dir, stack_name, sec_str, l['filename'])
                            l['filepath'] = labeling_path
                            l['previewpath'] = os.path.join(labeling_dir, stack_name, sec_str, l['filename'][:-4]+'.jpg')
                            labeling_dict = pickle.load(open(labeling_path, 'r'))
                            # print i, labeling_dict['final_polygons']
                            # print itemgetter(0)(labeling_dict['final_polygons'])
                            l['used_labels'] = np.unique(map(itemgetter(0), labeling_dict['final_polygons']))


            # if 'labelings' in sec_content.keys():
            #     sec_info['labelings'] = [k for k in sec_content['labelings'].keys() if k.endswith('pkl')]
            
            if stack_name in result_hierarchy:
                if sec_str in result_hierarchy[stack_name]:
                    results_list = result_hierarchy[stack_name][sec_str]
                    if len(results_list) > 0:
                        sec_info['available_results'] = [ r for r in results_list.keys() if resolution in r]

            # if 'pipelineResults' in sec_content.keys():
            #     sec_info['available_results'] = sec_content['pipelineResults'].keys()
            sec_infos.append(sec_info)

        sec_infos = sorted(sec_infos, key=lambda x: x['index'])

        stack_info['sections'] = sec_infos
        if stack_info['section_num'] is None:
            stack_info['section_num'] = len(sec_infos)

        dataset['stacks'].append(stack_info)
        dataset['available_stack_names'].append(stack_name)

    return dataset


# def build_labeling_index(dataset_json):
#     labeling_database = {}
#     for stack in dataset_json['stacks']:
#         for section in stack['sections']:
#             if 'labelings' in section:
#                 for labeling in section['labelings']:
#                     labeling_database.update({labeling['filename']: labeling['used_labels']})

#     return labeling_database


def build_inverse_labeing_index(dataset_json):
    from collections import defaultdict
    inv_labeling_database = defaultdict(list)

    for stack in dataset_json['stacks']:
        for section in stack['sections']:
            if 'labelings' in section:
                for labeling in section['labelings']:
                    for l in labeling['used_labels']:
                        inv_labeling_database[l].append(labeling)

    return inv_labeling_database


class DataManager(object):

    def __init__(self, data_dir=None, repo_dir=None, 
        result_dir=None, labeling_dir=None,
        generate_hierarchy=False,
        gabor_params_id=None, 
        segm_params_id=None,
        vq_params_id=None,
        stack=None,
        resol=None,
        section=None):

        self.generate_hierarchy = generate_hierarchy

        import os

        if data_dir is None:
            data_dir = os.environ['GORDON_DATA_DIR']
        if repo_dir is None:
            repo_dir=os.environ['GORDON_REPO_DIR']
        if result_dir is None:
            result_dir=os.environ['GORDON_RESULT_DIR']
        if labeling_dir is None:
            labeling_dir=os.environ['GORDON_LABELING_DIR']

        self.data_dir = data_dir
        self.repo_dir = repo_dir
        self.params_dir = os.path.join(repo_dir, 'params')

        self.root_labelings_dir = labeling_dir
        self.labelnames_path = os.path.join(labeling_dir, 'labelnames.txt')
    
        if os.path.isfile(self.labelnames_path):
            with open(self.labelnames_path, 'r') as f:
                self.labelnames = [n.strip() for n in f.readlines()]
                self.labelnames = [n for n in self.labelnames if len(n) > 0]
        else:
            self.labelnames = []

        self.root_results_dir = result_dir

        if self.generate_hierarchy:
            self.local_ds = generate_json(data_dir=data_dir, res_dir=result_dir, labeling_dir=labeling_dir)
            # print self.local_ds

            self.inv_labeing_index = build_inverse_labeing_index(self.local_ds)
            # print self.inv_labeing_index

        self.slice_ind = None
        self.image_name = None

        if gabor_params_id is None:
            self.gabor_params_id = 'blueNisslWide'

        if segm_params_id is None:
            self.segm_params_id = 'blueNisslRegular'

        if vq_params_id is None:
            self.vq_params_id = 'blueNissl'

        if stack is not None:
            self.set_stack(stack)

        if resol is not None:
            self.set_resol(resol)

        if section is not None:
            self.set_slice(section)

    def set_labelnames(self, labelnames):
        self.labelnames = labelnames

        with open(self.labelnames_path, 'w') as f:
            for n in labelnames:
                f.write('%s\n' % n)

    def set_stack(self, stack):
        self.stack = stack
        self.stack_path = os.path.join(self.data_dir, self.stack)

        if self.generate_hierarchy:
            self.stack_info = self.local_ds['stacks'][self.local_ds['available_stack_names'].index(stack)]

        self.slice_ind = None

    def set_resol(self, resol):
        if self.generate_hierarchy:        
            if resol not in self.stack_info['available_resolution']:
                raise Exception('images of resolution %s do not exist' % resol)

        self.resol = resol
        self.resol_dir = os.path.join(self.stack_path, self.resol)

        if self.generate_hierarchy:
            self.sections_info = self.stack_info['sections']

        if self.slice_ind is not None:
            self.set_slice(self.slice_ind)

    def set_slice(self, slice_ind):
        assert self.stack is not None and self.resol is not None, 'Stack is not specified'
        self.slice_ind = slice_ind
        self.slice_str = '%04d' % slice_ind
        self.image_dir = os.path.join(self.data_dir, self.stack, self.resol, self.slice_str)
        self.image_name = '_'.join([self.stack, self.resol, self.slice_str])

        self.image_path = os.path.join(self.image_dir, self.image_name + '.tif')

        # self.labelings_dir = os.path.join(self.image_dir, 'labelings')
        self.labelings_dir = os.path.join(self.root_labelings_dir, self.stack, self.slice_str)
        
#         self.results_dir = os.path.join(self.image_dir, 'pipelineResults')
        self.results_dir = os.path.join(self.root_results_dir, self.stack, self.slice_str)
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        if self.generate_hierarchy:
            self.section_info = self.sections_info[map(itemgetter('index'), self.sections_info).index(self.slice_ind)]

    def set_image(self, stack, resol, slice_ind):
        self.set_stack(stack)
        self.set_resol(resol)
        self.set_slice(slice_ind)
        self._load_image()

    def _get_image_filepath(self, stack=None, resol=None, section=None):
        if stack is None:
            stack = self.stack
        if resol is None:
            resol = self.resol
        if section is None:
            section = self.slice_ind

        image_dir = os.path.join(self.data_dir, stack, resol, '%04d'%section)
        image_name = '_'.join([stack, resol, '%04d'%section])
        image_filename = os.path.join(image_dir, image_name + '.tif')
        return image_filename
        
    def _load_image(self):
        
        assert self.image_name is not None, 'Image is not specified'

        image_filename = self._get_image_filepath()
        assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
        
        self.image = imread(image_filename, as_grey=True)
        self.image_height, self.image_width = self.image.shape[:2]
        
        self.image_rgb = imread(image_filename, as_grey=False)

        mask_filename = os.path.join(self.image_dir, self.image_name + '_mask.png')
        self.mask = imread(mask_filename, as_grey=True) > 0
        
    def set_gabor_params(self, gabor_params_id):
        
        self.gabor_params_id = gabor_params_id
        self.gabor_params = json.load(open(os.path.join(self.params_dir, 'gabor', 'gabor_' + gabor_params_id + '.json'), 'r')) if gabor_params_id is not None else None
        self._generate_kernels(self.gabor_params)
    
    def _generate_kernels(self, gabor_params):
        
        from skimage.filter import gabor_kernel
    
        theta_interval = gabor_params['theta_interval']
        self.n_angle = int(180/theta_interval)
        freq_step = gabor_params['freq_step']
        freq_max = 1./gabor_params['min_wavelen']
        freq_min = 1./gabor_params['max_wavelen']
        bandwidth = gabor_params['bandwidth']
        self.n_freq = int(np.log(freq_max/freq_min)/np.log(freq_step)) + 1
        self.frequencies = freq_max/freq_step**np.arange(self.n_freq)
        self.angles = np.arange(0, self.n_angle)*np.deg2rad(theta_interval)

        kernels = [gabor_kernel(f, theta=t, bandwidth=bandwidth) for f in self.frequencies for t in self.angles]
        kernels = map(np.real, kernels)

        biases = np.array([k.sum() for k in kernels])
        mean_bias = biases.mean()
        self.kernels = [k/k.sum()*mean_bias for k in kernels] # this enforces all kernel sums to be identical, but non-zero

        # kernels = [k - k.sum()/k.size for k in kernels] # this enforces all kernel sum to be zero

        self.n_kernel = len(kernels)
        self.max_kern_size = np.max([kern.shape[0] for kern in self.kernels])

    def print_gabor_info(self):
        print 'num. of kernels: %d' % (self.n_kernel)
        print 'frequencies:', self.frequencies
        print 'wavelength (pixels):', 1/self.frequencies
        print 'max kernel matrix size:', self.max_kern_size
        
    def set_segmentation_params(self, segm_params_id):
        
        self.segm_params_id = segm_params_id
        self.segm_params = json.load(open(os.path.join(self.params_dir, 'segm', 'segm_' + segm_params_id + '.json'), 'r')) if segm_params_id is not None else None

    def set_vq_params(self, vq_params_id):
        
        self.vq_params_id = vq_params_id
        self.vq_params = json.load(open(os.path.join(self.params_dir, 'vq', 'vq_' + vq_params_id + '.json'), 'r')) if vq_params_id is not None else None
        
            
    # def _get_result_filename(self, result_name, ext, results_dir=None, param_dependencies=None):
    def _get_result_filename(self, result_name, ext, param_dependencies=None, section=None):
        
        if section is not None:
            self.set_slice(section)

        if param_dependencies is None:
            param_dependencies = ['gabor', 'segm', 'vq']

        if result_name in ['textons']:
            results_dir = os.path.join(os.environ['GORDON_RESULT_DIR'], self.stack)
            param_dependencies = ['gabor', 'vq']

        else:
            results_dir = self.results_dir
        
            if result_name in ['features', 'kernels', 'features_rotated', 'features_rotated_pca', 'max_angle_indices']:
                param_dependencies = ['gabor']

            elif result_name in['segmentation', 'segmentationWithText', 'segmentationWithoutText',
                                'segmentationTransparent', 'spProps', 'neighbors']:
                param_dependencies = ['segm']
                            
            elif result_name in ['dirMap', 'dirHist', 'spMaxDirInd', 'spMaxDirAngle']:
                param_dependencies = ['gabor', 'segm']
                
                
            elif result_name in ['texMap', 'original_centroids']:
                param_dependencies = ['gabor', 'vq']

            elif result_name in ['texHist', 'clusters', 'groups', 'groupsTop10Vis', 
                            'groupsTop20to30Vis', 'groupsTop10to20Vis', 'texHistPairwiseDist',
                            'votemap', 'votemapOverlaid']:
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
                assert param_dependencies is not None
                # raise Exception('result name %s unknown' % result_name)

        # instance_name = self.image_name

        param_strs = []
        if 'gabor' in param_dependencies:
            param_strs.append('gabor-' + self.gabor_params_id)
        if 'segm' in param_dependencies:
            param_strs.append('segm-' + self.segm_params_id)
        if 'vq' in param_dependencies:
            param_strs.append('vq-' + self.vq_params_id)
            # raise Exception("parameter dependency string not recognized")
        
        if result_name in ['textons']:
            result_filename = os.path.join(results_dir, self.stack + '_' +self.resol + '_' + '-'.join(param_strs) + '_' + result_name + '.' + ext)
        else:
            result_filename = os.path.join(results_dir, self.image_name + '_' + '-'.join(param_strs) + '_' + result_name + '.' + ext)
        
        return result_filename
            

    def check_pipeline_result(self, result_name, ext):
        if REGENERATE_ALL_RESULTS:
            return False

        result_filename = self._get_result_filename(result_name, ext)
        return os.path.exists(result_filename)

    def load_pipeline_result(self, result_name, ext, is_rgb=None, section=None):
        
        if REGENERATE_ALL_RESULTS:
            raise
        
        result_filename = self._get_result_filename(result_name, ext, section=section)

        if ext == 'npy':
            assert os.path.exists(result_filename), "Pipeline result '%s' does not exist" % (result_name + '.' + ext)
            data = np.load(result_filename)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = imread(result_filename, as_grey=False)
            data = self._regulate_image(data, is_rgb)
        elif ext == 'pkl':
            data = pickle.load(open(result_filename, 'r'))

        print 'loaded %s' % result_filename

        return data
        
    def save_pipeline_result(self, data, result_name, ext, 
        param_dependencies=None, is_rgb=None, section=None):
        
        if param_dependencies is None:
            param_dependencies = ['gabor', 'segm', 'vq']

        result_filename = self._get_result_filename(result_name, ext, 
            param_dependencies=param_dependencies, section=section)

        if ext == 'npy':
            np.save(result_filename, data)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = self._regulate_image(data, is_rgb)
            imsave(result_filename, data)
        elif ext == 'pkl':
            pickle.dump(data, open(result_filename, 'w'))
            
        print 'saved %s' % result_filename
        

    def load_labeling(self, stack=None, section=None, labeling_name=None):
        labeling_fn = self._load_labeling_path(stack, section, labeling_name)
        labeling = pickle.load(open(labeling_fn, 'r'))
        return labeling

    def _load_labeling_preview_path(self, stack=None, section=None, labeling_name=None):
        if stack is None:
            stack = self.stack
        if section is None:
            section = self.slice_ind

        if labeling_name.endswith('pkl'): # full filename
            return os.path.join(self.labelings_dir, labeling_name[:-4]+'.jpg')
        else:
            return os.path.join(self.labelings_dir, '_'.join([stack, '%04d'%section, labeling_name]) + '.jpg')
        
    def _load_labeling_path(self, stack=None, section=None, labeling_name=None):
        if stack is None:
            stack = self.stack
        if section is None:
            section = self.slice_ind

        if labeling_name.endswith('pkl'): # full filename
            return os.path.join(self.labelings_dir, labeling_name)
        else:
            return os.path.join(self.labelings_dir, '_'.join([stack, '%04d'%section, labeling_name]) + '.pkl')
        

    def load_labeling_preview(self, stack=None, section=None, labeling_name=None):
        return imread(self._load_labeling_preview_path(stack, section, labeling_name))
        
    def save_labeling(self, labeling, new_labeling_name, labelmap_vis):
        
        try:
            os.makedirs(self.labelings_dir)
        except:
            pass

        new_labeling_fn = self._load_labeling_path(labeling_name=new_labeling_name)
        # os.path.join(self.labelings_dir, self.image_name + '_' + new_labeling_name + '.pkl')
        pickle.dump(labeling, open(new_labeling_fn, 'w'))
        print 'Labeling saved to', new_labeling_fn

        new_preview_fn = self._load_labeling_preview_path(labeling_name=new_labeling_name)

        # os.path.join(self.labelings_dir, self.image_name + '_' + new_labeling_name + '.tif')
        data = self._regulate_image(labelmap_vis, is_rgb=True)
        imsave(new_preview_fn, data)
        print 'Preview saved to', new_preview_fn

        return new_labeling_fn
        
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
    

    def visualize_edges(self, edges, img=None, text=False, color=[0,0,255]):
        '''
        Return a visualization of edgelets
        '''

        if not hasattr(self, 'edge_coords'):
            self.edge_coords = self.load_pipeline_result('edgeCoords', 'pkl')

        if not hasattr(self, 'image'):
            self._load_image()

        if img is None:
            img = self.image
            img_rgb = self.image_rgb
        else:
            img_rgb = gray2rgb(img) if img.ndim == 2 else img

        vis = img_as_ubyte(img_rgb)
        for edge_ind, degde in enumerate(edges):
            q = frozenset(degde)
            if q in self.edge_coords:
                for y, x in self.edge_coords[q]:
                    vis[y, x] = color
                if text:
                    cv2.putText(vis, str(edge_ind), tuple([x, y]), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, 255, 1)
        return vis

    
    def visualize_edge_sets(self, edge_sets, img=None, text=False, colors=None):
        '''
        Return a visualization of multiple sets of edgelets
        '''
        
        if not hasattr(self, 'edge_coords'):
            self.edge_coords = self.load_pipeline_result('edgeCoords', 'pkl')

        if not hasattr(self, 'image'):
            self._load_image()

        if img is None:
            img = self.image
            img_rgb = self.image_rgb
        else:
            img_rgb = gray2rgb(img) if img.ndim == 2 else img
        
        if colors is None:
            colors = np.uint8(np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt') * 255)

        vis = img_as_ubyte(img_rgb)
            
        for edgeSet_ind, edges in enumerate(edge_sets):
            
            pointset = []
            
            for e_ind, degde in enumerate(edges):
                q = frozenset(degde)
                if q in self.edge_coords:
                    for point_ind, (y, x) in enumerate(self.edge_coords[q]):    
                        vis[max(0, y-5):min(self.image_height, y+5), 
                                max(0, x-5):min(self.image_width, x+5)] = colors[edgeSet_ind%len(colors)]
                        pointset.append((y,x))

            if text:
                import cv2
                ymean, xmean = np.mean(pointset, axis=0)
                c = colors[edgeSet_ind%len(colors)].astype(np.int)
                cv2.putText(vis, str(edgeSet_ind), 
                              tuple(np.floor([xmean-100,ymean+100]).astype(np.int)), 
                              cv2.FONT_HERSHEY_DUPLEX,
                              5., ((c[0],c[1],c[2])), 10)
        
        return vis

def display(vis, filename='tmp.jpg'):
    
    if vis.dtype != np.uint8:
        imsave(filename, img_as_ubyte(vis))
    else:
        imsave(filename, vis)
            
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

def kl(a,b):
    m = (a!=0) & (b!=0)
    return np.sum(a[m]*np.log(a[m]/b[m]))

def js(u,v):
    m = .5 * (u + v)
    r = .5 * (kl(u,m) + kl(v,m))
    return r

# <codecell>

def chi2(u,v):
    """
    Compute Chi^2 distance between two distributions.
    
    Empty bins are ignored.
    
    """
    
#     m = (u != 0) & (v != 0)
#     q = (u-v)**2/(u+v)
#     r = np.sum(q[m])
    
    r = np.nansum((u-v)**2/(u+v))
    return r


def alpha_blending(src_rgb, dst_rgb, src_alpha, dst_alpha):
    
    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] +
               dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]
    
    out = np.zeros((src_rgb.shape[0], src_rgb.shape[1], 4))
        
    out[..., :3] = out_rgb
    out[..., 3] = out_alpha
    
    return out

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

