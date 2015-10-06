from skimage.filters import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.color import color_dict, gray2rgb, label2rgb, rgb2gray
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation, binary_erosion, watershed, remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_ubyte, img_as_float
from skimage.io import imread, imsave
from scipy.spatial.distance import cdist
import numpy as np
import os
import csv
import sys
from operator import itemgetter
import json
import cPickle as pickle

from tables import *

from subprocess import check_output, call

import matplotlib.pyplot as plt

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def execute_command(cmd):
    print cmd

    try:
        retcode = call(cmd, shell=True)
        if retcode < 0:
            print >>sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >>sys.stderr, "Child returned", retcode
    except OSError as e:
        print >>sys.stderr, "Execution failed:", e
        raise e

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

class DataManager(object):

    def __init__(self, data_dir=os.environ['GORDON_DATA_DIR'], 
                 repo_dir=os.environ['GORDON_REPO_DIR'], 
                 result_dir=os.environ['GORDON_RESULT_DIR'], 
                 labeling_dir=os.environ['GORDON_LABELING_DIR'],
                 gabor_params_id=None, 
                 segm_params_id=None, 
                 vq_params_id=None,
                 stack=None,
                 resol='lossless',
                 section=None):

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

        self.slice_ind = None
        self.image_name = None

        if gabor_params_id is None:
            self.set_gabor_params('blueNisslWide')
        else:
            self.set_gabor_params(gabor_params_id)

        if segm_params_id is None:
            self.set_segmentation_params('blueNisslRegular')
        else:
            self.set_segmentation_params(segm_params_id)

        if vq_params_id is None:
            self.set_vq_params('blueNissl')
        else:
            self.set_vq_params(vq_params_id)
            
        if stack is not None:
            self.set_stack(stack)

        if resol is not None:
            self.set_resol(resol)

        if section is not None:
            self.set_slice(section)

#     def set_labelnames(self, labelnames):
#         self.labelnames = labelnames

#         with open(self.labelnames_path, 'w') as f:
#             for n in labelnames:
#                 f.write('%s\n' % n)

    def set_stack(self, stack):
        self.stack = stack
#         self.stack_path = os.path.join(self.data_dir, self.stack)
#         self.slice_ind = None
        
    def set_resol(self, resol):
        self.resol = resol
        
    def set_slice(self, slice_ind):
        assert self.stack is not None and self.resol is not None, 'Stack is not specified'
        self.slice_ind = slice_ind
        self.slice_str = '%04d' % slice_ind
        if self.resol == 'lossless':
            self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_cropped')
            self.image_name = '_'.join([self.stack, self.slice_str, self.resol])
            self.image_path = os.path.join(self.image_dir, self.image_name + '_warped.tif')

        self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))

        if self.stack == 'MD593':
            self.mask = np.zeros((self.image_height, self.image_width), np.bool)
            self.mask[1848:1848+4807, 924:924+10186] = True
            
            rs, cs = np.where(self.mask)
            self.ymax = rs.max()
            self.ymin = rs.min()
            self.xmax = cs.max()
            self.xmin = cs.min()
            self.h = self.ymax-self.ymin+1
            self.w = self.xmax-self.xmin+1

        # self.labelings_dir = os.path.join(self.image_dir, 'labelings')
        self.labelings_dir = os.path.join(self.root_labelings_dir, self.stack, self.slice_str)
        
#         self.results_dir = os.path.join(self.image_dir, 'pipelineResults')
        
        self.results_dir = os.path.join(self.root_results_dir, self.stack, self.slice_str)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    # def set_image(self, stack, slice_ind):
    #     self.set_stack(stack)
    #     self.set_slice(slice_ind)
    #     self._load_image()

    def _get_image_filepath(self, stack=None, resol=None, section=None, version='rgb-jpg'):
        if stack is None:
            stack = self.stack
        if resol is None:
            resol = self.resol
        if section is None:
            section = self.slice_ind
            
        slice_str = '%04d' % section

        if version == 'rgb-jpg':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_downscaled')
            image_name = '_'.join([stack, slice_str, resol, 'warped_downscaled'])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif version == 'gray-jpg':
        #     image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale_downscaled')
        #     image_name = '_'.join([stack, slice_str, resol, 'warped'])
        #     image_path = os.path.join(image_dir, image_name + '.jpg')
        elif version == 'gray':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale')
            image_name = '_'.join([stack, slice_str, resol, 'warped_grayscale'])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif version == 'rgb':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped')
            image_name = '_'.join([stack, slice_str, resol, 'warped'])
            image_path = os.path.join(image_dir, image_name + '.tif')
         
        return image_path
    
    def _read_image(self, image_filename):
        if image_filename.endswith('tif') or image_filename.endswith('tiff'):
            from PIL.Image import open
            img = np.array(open(image_filename))/255.
        else:
            img = imread(image_filename)
        return img

    def _load_image(self, versions=['rgb', 'gray', 'rgb-jpg']):
        
        assert self.image_name is not None, 'Image is not specified'

        if 'rgb-jpg' in versions and not hasattr(self, 'image_rgb_jpg'):
            image_filename = self._get_image_filepath(version='rgb-jpg')
            # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
            self.image_rgb_jpg = self._read_image(image_filename)
        
        if 'rgb' in versions and not hasattr(self, 'image_rgb'):
            image_filename = self._get_image_filepath(version='rgb')
            # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
            self.image_rgb = self._read_image(image_filename)

        if 'gray' in versions and not hasattr(self, 'image'):
            image_filename = self._get_image_filepath(version='gray')
            # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
            self.image = self._read_image(image_filename)

        # elif format == 'gray-rgb':
        #     self.image = rgb2gray(self.image_rgb)

        # self.image_rgb = imread(image_filename, as_grey=False)
#         self.image = rgb2gray(self.image_rgb)
        
#         mask_filename = os.path.join(self.image_dir, self.image_name + '_mask.png')
#         self.mask = imread(mask_filename, as_grey=True) > 0
        
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
        # self.segm_params = json.load(open(os.path.join(self.params_dir, 'segm', 'segm_' + segm_params_id + '.json'), 'r')) if segm_params_id is not None else None
        if self.segm_params_id == 'gridsize200':
            self.grid_size = 200
        else:
            self.grid_size = 100

    def set_vq_params(self, vq_params_id):
        
        self.vq_params_id = vq_params_id
        self.vq_params = json.load(open(os.path.join(self.params_dir, 'vq', 'vq_' + vq_params_id + '.json'), 'r')) if vq_params_id is not None else None
        
    
    def _param_str(self, param_dependencies):
        param_strs = []
        if 'gabor' in param_dependencies:
            param_strs.append('gabor-' + self.gabor_params_id)
        if 'segm' in param_dependencies:
            param_strs.append('segm-' + self.segm_params_id)
        if 'vq' in param_dependencies:
            param_strs.append('vq-' + self.vq_params_id)
        return '-'.join(param_strs)
        
    def _refresh_result_info(self):
        with open(self.repo_dir + '/notebooks/results.csv', 'r') as f:
            f.readline()
            self.result_info = {}
            for row in csv.DictReader(f, delimiter=' '):
                self.result_info[row['name']] = row
        
        
    def _get_result_filename(self, result_name):

        if not hasattr(self, 'result_info'):
            with open(self.repo_dir + '/notebooks/results.csv', 'r') as f:
                f.readline()
                self.result_info = {}
                for row in csv.DictReader(f, delimiter=' '):
                    self.result_info[row['name']] = row
                
        info = self.result_info[result_name]
        
        if info['dir'] == '0':
            result_dir = self.results_dir
            prefix = self.image_name
        elif info['dir'] == '1':
            result_dir = os.path.join(os.environ['GORDON_RESULT_DIR'], self.stack)
            prefix = self.stack + '_' + self.resol

        else:
            raise Exception('unrecognized result dir specification')
            
        if info['param_dep'] == '0':
            param_dep = ['gabor']
        elif info['param_dep'] == '1':
            param_dep = ['segm']
        elif info['param_dep'] == '2':
            param_dep = ['gabor', 'vq']
        elif info['param_dep'] == '3':
            param_dep = ['gabor', 'segm', 'vq']
        else:
            raise Exception('unrecognized result param_dep specification')
            
        result_filename = os.path.join(result_dir, '_'.join([prefix, self._param_str(param_dep), 
                                                             result_name + '.' + info['extension']]))

        return result_filename
            
    def check_pipeline_result(self, result_name):
#         if REGENERATE_ALL_RESULTS:
#             return False
        result_filename = self._get_result_filename(result_name)
        return os.path.exists(result_filename)

    def load_pipeline_result(self, result_name, is_rgb=None, section=None):
        
        result_filename = self._get_result_filename(result_name)
        ext = self.result_info[result_name]['extension']
        
        if ext == 'npy':
            assert os.path.exists(result_filename), "%d: Pipeline result '%s' does not exist, trying to find %s" % (self.slice_ind, result_name + '.' + ext, result_filename)
            data = np.load(result_filename)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = imread(result_filename, as_grey=False)
            data = self._regulate_image(data, is_rgb)
        elif ext == 'pkl':
            data = pickle.load(open(result_filename, 'r'))
        elif ext == 'hdf':
            with open_file(result_filename, mode="r") as f:
                data = f.get_node('/data').read()

        # print 'loaded %s' % result_filename

        return data
        
    def save_pipeline_result(self, data, result_name, is_rgb=None, section=None):
        
        result_filename = self._get_result_filename(result_name)
        ext = self.result_info[result_name]['extension']

        if ext == 'npy':
            np.save(result_filename, data)
        elif ext == 'tif' or ext == 'png' or ext == 'jpg':
            data = self._regulate_image(data, is_rgb)
            imsave(result_filename, data)
        elif ext == 'pkl':
            pickle.dump(data, open(result_filename, 'w'))
        elif ext == 'hdf':
            filters = Filters(complevel=9, complib='blosc')
            with open_file(result_filename, mode="w") as f:
                _ = f.create_carray('/', 'data', Atom.from_dtype(data.dtype), filters=filters, obj=data)
            
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

        # new_preview_fn = self._load_labeling_preview_path(labeling_name=new_labeling_name)

        # # os.path.join(self.labelings_dir, self.image_name + '_' + new_labeling_name + '.tif')
        # data = self._regulate_image(labelmap_vis, is_rgb=True)
        # imsave(new_preview_fn, data)
        # print 'Preview saved to', new_preview_fn

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
    
    
    # def visualize_segmentation(self, bg='rgb-jpg', show_sp_index=True):

    #     if bg == 'originalImage':
    #         if not hasattr(self, 'image'):
    #             self._load_image(format='rgb-jpg')
    #         viz = self.image
    #     elif bg == 'transparent':


    #     img_superpixelized = mark_boundaries(viz, self.segmentation)
    #     img_superpixelized = img_as_ubyte(img_superpixelized)
    #     dm.save_pipeline_result(img_superpixelized, 'segmentationWithoutText')

    #     for s in range(n_superpixels):
    #         cv2.putText(img_superpixelized, str(s), 
    #                     tuple(np.floor(sp_centroids[s][::-1]).astype(np.int) - np.array([10,-10])), 
    #                     cv2.FONT_HERSHEY_DUPLEX, .5, ((255,0,255)), 1)



    def visualize_edge_set(self, edges, bg=None, show_edge_index=False, c=None):
        
        import cv2
        
        if not hasattr(self, 'edge_coords'):
            self.edge_coords = self.load_pipeline_result('edgeCoords')
           
        if not hasattr(self, 'edge_midpoints'):
            self.edge_midpoints = self.load_pipeline_result('edgeMidpoints')
            
        if not hasattr(self, 'edge_vectors'):
            self.edge_vectors = self.load_pipeline_result('edgeVectors')

        if bg == 'originalImage':
            if not hasattr(self, 'image'):
                self._load_image(version='rgb_jpg')
            segmentation_viz = self.image
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg
            
        vis = img_as_ubyte(segmentation_viz[self.ymin:self.ymax+1, self.xmin:self.xmax+1])
        
        directed = isinstance(list(edges)[0], tuple)
        
        if c is None:
            c = [255,0,0]
        
        for e_ind, edge in enumerate(edges):
                
            if directed:
                e = frozenset(edge)
                midpoint = self.edge_midpoints[e]
                end = midpoint + 10 * self.edge_vectors[edge]
                cv2.line(vis, tuple((midpoint-(self.xmin, self.ymin)).astype(np.int)), 
                         tuple((end-(self.xmin, self.ymin)).astype(np.int)), 
                         (c[0],c[1],c[2]), 5)
                stroke_pts = self.edge_coords[e]
            else:
                stroke_pts = self.edge_coords[edge]

            for x, y in stroke_pts:
                cv2.circle(vis, (x-self.xmin, y-self.ymin), 5, c, -1)

            if show_edge_index:
                cv2.putText(vis, str(e_ind), 
                            tuple(np.floor(midpoint + [-50, 30] - (self.xmin, self.ymin)).astype(np.int)), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, ((c[0],c[1],c[2])), 3)
        
        return vis
    
    
    def visualize_edge_sets(self, edge_sets, bg='segmentationWithText', show_set_index=0, colors=None, neighbors=None, labels=None):
        '''
        Return a visualization of multiple sets of edgelets
        '''
        
        import cv2
        
        if not hasattr(self, 'edge_coords'):
            self.edge_coords = self.load_pipeline_result('edgeCoords')
           
        if not hasattr(self, 'edge_midpoints'):
            self.edge_midpoints = self.load_pipeline_result('edgeMidpoints')
            
        if not hasattr(self, 'edge_vectors'):
            self.edge_vectors = self.load_pipeline_result('edgeVectors')

        if colors is None:
            colors = np.uint8(np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt') * 255)
        
        if bg == 'originalImage':
            if not hasattr(self, 'image'):
                self._load_image(version='rgb-jpg')
            segmentation_viz = self.image
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg
            
        vis = img_as_ubyte(segmentation_viz[self.ymin:self.ymax+1, self.xmin:self.xmax+1])
            
        # if input are tuples, draw directional sign
        if len(edge_sets) == 0:
            return vis
        else:
            directed = isinstance(list(edge_sets[0])[0], tuple)
            
        for edgeSet_ind, edges in enumerate(edge_sets):
            
#             junction_pts = []
            
            if labels is None:
                s = str(edgeSet_ind)
                c = colors[edgeSet_ind%len(colors)].astype(np.int)
            else:
                s = labels[edgeSet_ind]
                c = colors[int(s)%len(colors)].astype(np.int)
            
            for e_ind, edge in enumerate(edges):
                
                if directed:
                    e = frozenset(edge)
                    midpoint = self.edge_midpoints[e]
                    end = midpoint + 10 * self.edge_vectors[edge]
                    cv2.line(vis, tuple((midpoint-(self.xmin, self.ymin)).astype(np.int)), 
                             tuple((end-(self.xmin, self.ymin)).astype(np.int)), 
                             (c[0],c[1],c[2]), 5)
                    stroke_pts = self.edge_coords[e]
                else:
                    stroke_pts = self.edge_coords[edge]
                
                for x, y in stroke_pts:
                    cv2.circle(vis, (x-self.xmin, y-self.ymin), 5, c, -1)

                    # vis[max(0, y-5):min(self.image_height, y+5), 
                    #     max(0, x-5):min(self.image_width, x+5)] = (c[0],c[1],c[2],1) if vis.shape[2] == 4 else c
                                                
#                 if neighbors is not None:
#                     nbrs = neighbors[degde]
#                     for nbr in nbrs:
#                         pts2 = self.edge_coords[frozenset(nbr)]
#                         am = np.unravel_index(np.argmin(cdist(pts[[0,-1]], pts2[[0,-1]]).flat), (2,2))
# #                         print degde, nbr, am
#                         junction_pt = (pts[-1 if am[0]==1 else 0] + pts2[-1 if am[1]==1 else 0])/2
#                         junction_pts.append(junction_pt)
                 
            if show_set_index:

                if directed:
                    centroid = np.mean([self.edge_midpoints[frozenset(e)] for e in edges], axis=0)
                else:
                    centroid = np.mean([self.edge_midpoints[e] for e in edges], axis=0)

                cv2.putText(vis, s, 
                            tuple(np.floor(centroid + [-100, 100] - (self.xmin, self.ymin)).astype(np.int)), 
                            cv2.FONT_HERSHEY_DUPLEX,
                            3, ((c[0],c[1],c[2])), 3)
            
#             for p in junction_pts:
#                 cv2.circle(vis, tuple(np.floor(p).astype(np.int)), 5, (255,0,0), -1)
        
        return vis

    
    def visualize_kernels(self):
        fig, axes = plt.subplots(self.n_freq, self.n_angle, figsize=(20,20))

        for i, kern in enumerate(self.kernels):
            r, c = np.unravel_index(i, (self.n_freq, self.n_angle))
            axes[r,c].matshow(kern, cmap=plt.cm.gray)
        #     axes[r,c].colorbar()
        plt.show()


    def visualize_cluster(self, cluster, bg='segmentationWithText', seq_text=False, highlight_seed=True,
                         ymin=None, xmin=None, ymax=None, xmax=None):

        if ymin is None:
            ymin=self.ymin
        if xmin is None:
            xmin=self.xmin
        if ymax is None:
            ymax=self.ymax
        if xmax is None:
            xmax=self.xmax
        
        if not hasattr(self, 'sp_coords'):
            self.sp_coords = self.load_pipeline_result('spCoords')
                    
        if bg == 'originalImage':
            segmentation_viz = self.image
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg

        msk = -1*np.ones((self.image_height, self.image_width), np.int8)

        for i, c in enumerate(cluster):
            rs = self.sp_coords[c][:,0]
            cs = self.sp_coords[c][:,1]
            if highlight_seed and i == 0:
                msk[rs, cs] = 1
            else:
                msk[rs, cs] = 0

        viz_msk = label2rgb(msk[ymin:ymax+1, xmin:xmax+1], image=segmentation_viz[ymin:ymax+1, xmin:xmax+1])

        if seq_text:
            viz_msk = img_as_ubyte(viz_msk[...,::-1])

            if not hasattr(self, 'sp_centroids'):
                self.sp_centroids = self.load_pipeline_result('spCentroids')

            import cv2
            for i, sp in enumerate(cluster):
                cv2.putText(viz_msk, str(i), tuple((self.sp_centroids[sp, ::-1] - (xmin, ymin) - (10,-10)).astype(np.int)), cv2.FONT_HERSHEY_DUPLEX, 1., ((0,255,255)), 1)

        return viz_msk

    
    def visualize_edges_and_superpixels(self, edge_sets, clusters, colors=None):
        if colors is None:
            colors = np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt')
            
        vis = self.visualize_multiple_clusters(clusters, colors=colors)
        viz = self.visualize_edge_sets(edge_sets, directed=True, img=vis, colors=colors)
        return viz
        
    
    def visualize_multiple_clusters(self, clusters, bg='segmentationWithText', alpha_blend=True, colors=None,
                                    show_cluster_indices=True,
                                    ymin=None, xmin=None, ymax=None, xmax=None):
        
        if ymin is None:
            ymin = self.ymin
        if xmin is None:
            xmin = self.xmin
        if ymax is None:
            ymax = self.ymax
        if xmax is None:
            xmax = self.xmax

        if not hasattr(self, 'segmentation'):
            self.segmentation = self.load_pipeline_result('segmentation')

        if len(clusters) == 0:
            return segmentation_vis
        
        if colors is None:
            colors = np.loadtxt(os.environ['GORDON_REPO_DIR'] + '/visualization/100colors.txt')
            
        n_superpixels = self.segmentation.max() + 1
        

        if bg == 'originalImage':
            if not hasattr(self, 'image'):
                self._load_image(version='rgb-jpg')
            segmentation_viz = self.image
        elif bg == 'segmentationWithText':
            if not hasattr(self, 'segmentation_vis'):
                self.segmentation_viz = self.load_pipeline_result('segmentationWithText')
            segmentation_viz = self.segmentation_viz
        elif bg == 'segmentationWithoutText':
            if not hasattr(self, 'segmentation_notext_vis'):
                self.segmentation_notext_viz = self.load_pipeline_result('segmentationWithoutText')
            segmentation_viz = self.segmentation_notext_viz
        else:
            segmentation_viz = bg

        mask_alpha = .4
        
        if alpha_blend:
            
            for ci, c in enumerate(clusters):
                m =  np.zeros((n_superpixels,), dtype=np.float)
                m[list(c)] = mask_alpha
                alpha = m[self.segmentation[ymin:ymax+1, xmin:xmax+1]]
                alpha[~self.mask[ymin:ymax+1, xmin:xmax+1]] = 0
                
                mm = np.zeros((n_superpixels,3), dtype=np.float)
                mm[list(c)] = colors[ci]
                blob = mm[self.segmentation[ymin:ymax+1, xmin:xmax+1]]
                
                if ci == 0:
                    vis = alpha_blending(blob, segmentation_viz[ymin:ymax+1, xmin:xmax+1], alpha, 1.)
                else:
                    vis = alpha_blending(blob, vis[..., :-1], alpha, vis[..., -1])

        else:
        
            n_superpixels = self.segmentation.max() + 1

            n = len(clusters)
            m = -1*np.ones((n_superpixels,), dtype=np.int)

            for ci, c in enumerate(clusters):
                m[list(c)] = ci

            a = m[self.segmentation]
            a[~self.mask] = -1
        #     a = -1*np.ones_like(segmentation)
        #     for ci, c in enumerate(clusters):
        #         for i in c:
        #             a[segmentation == i] = ci

            vis = label2rgb(a, image=segmentation_viz)

            vis = img_as_ubyte(vis[...,::-1])

        if show_cluster_indices:
            if not hasattr(self, 'sp_centroids'):
                self.sp_centroids = self.load_pipeline_result('spCentroids')
            for ci, cl in enumerate(clusters):
                cluster_center_yx = sp_centroids[cl].mean(axis=0).astype(np.int)
                vis = cv2.putText(vis, str(ci), tuple(cluster_center_yx[::-1] - np.array([10,-10])), 
                                cv2.FONT_HERSHEY_DUPLEX, 1., ((0,255,255)), 1)

                # for i, sp in enumerate(c):
                #     vis = cv2.putText(vis, str(i), 
                #                       tuple(np.floor(sp_properties[sp, [1,0]] - np.array([10,-10])).astype(np.int)), 
                #                       cv2.FONT_HERSHEY_DUPLEX,
                #                       1., ((0,255,255)), 1)
        
        return vis.copy()


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
    
    u[u==0] = 1e-6
    v[v==0] = 1e-6
    r = np.sum(((u-v)**2).astype(np.float)/(u+v))

    # m = (u != 0) & (v != 0)
    # r = np.sum(((u[m]-v[m])**2).astype(np.float)/(u[m]+v[m]))
    
    # r = np.nansum(((u-v)**2).astype(np.float)/(u+v))
    return r


def chi2s(h1s, h2s):
    '''
    h1s is n x n_texton
    MUST be float type
    '''    
    return np.sum((h1s-h2s)**2/(h1s+h2s+1e-10), axis=1)

# def chi2s(h1s, h2s):
#     '''
#     h1s is n x n_texton
#     '''
#     s = (h1s+h2s).astype(np.float)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         ss = (h1s-h2s)**2/s
#     ss[s==0] = 0
#     return np.sum(ss, axis=1)


def alpha_blending(src_rgb, dst_rgb, src_alpha, dst_alpha):
    
    
    if src_rgb.dtype == np.uint8:
        src_rgb = img_as_float(src_rgb)

    if dst_rgb.dtype == np.uint8:
        dst_rgb = img_as_float(dst_rgb)
        
    if isinstance(src_alpha, float) or  isinstance(src_alpha, int):
        src_alpha = src_alpha * np.ones((src_rgb.shape[0], src_rgb.shape[1]))

    if isinstance(dst_alpha, float) or  isinstance(dst_alpha, int):
        dst_alpha = dst_alpha * np.ones((dst_rgb.shape[0], dst_rgb.shape[1]))

    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] +
               dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]
    
    out = np.zeros((src_rgb.shape[0], src_rgb.shape[1], 4))
        
    out[..., :3] = out_rgb
    out[..., 3] = out_alpha
    
    return out