from skimage.filters import threshold_otsu, threshold_adaptive, gaussian_filter
from skimage.color import color_dict, gray2rgb, label2rgb, rgb2gray
from skimage.segmentation import clear_border
from skimage.morphology import binary_dilation, binary_erosion, watershed, remove_small_objects
from skimage.measure import regionprops, label
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_ubyte, img_as_float
from skimage.io import imread, imsave
from skimage.transform import rescale
from scipy.spatial.distance import cdist, pdist
import numpy as np
import os
import csv
import sys
from operator import itemgetter
import json
import cPickle as pickle
import datetime

import cv2

from tables import open_file, Filters, Atom
import bloscpack as bp

from subprocess import check_output, call

import matplotlib.pyplot as plt

from ipywidgets import FloatProgress
from IPython.display import display

########### Data Directories #############

import subprocess
hostname = subprocess.check_output("hostname", shell=True).strip()

if hostname.endswith('sdsc.edu'):
    print 'Setting environment for Gordon'
    atlasAlignParams_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignParams_atlas'
    atlasAlignOptLogs_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_atlasAlignOptLogs_atlas'
    volume_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes/'
    labelingViz_root = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
    scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremaps_lossless_svm_Sat16ClassFinetuned_v3/'
    scoremapViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremapViz_svm_Sat16ClassFinetuned_v3'
    annotationViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
else:
    print 'Setting environment for Brainstem workstation or Local'
    volume_dir = '/home/yuncong/CSHL_volumes/'
    mesh_rootdir = '/home/yuncong/CSHL_meshes'

############ Class Labels #############

volume_landmark_names_unsided = ['12N', '5N', '6N', '7N', '7n', 'AP', 'Amb', 'LC',
                                 'LRt', 'Pn', 'R', 'RtTg', 'Tz', 'VLL', 'sp5']
linear_landmark_names_unsided = ['outerContour']

labels_unsided = volume_landmark_names_unsided + linear_landmark_names_unsided
labels_unsided_indices = dict((j, i+1) for i, j in enumerate(labels_unsided))  # BackG always 0

labelMap_unsidedToSided = {'12N': ['12N'],
                            '5N': ['5N_L', '5N_R'],
                            '6N': ['6N_L', '6N_R'],
                            '7N': ['7N_L', '7N_R'],
                            '7n': ['7n_L', '7n_R'],
                            'AP': ['AP'],
                            'Amb': ['Amb_L', 'Amb_R'],
                            'LC': ['LC_L', 'LC_R'],
                            'LRt': ['LRt_L', 'LRt_R'],
                            'Pn': ['Pn_L', 'Pn_R'],
                            'R': ['R_L', 'R_R'],
                            'RtTg': ['RtTg'],
                            'Tz': ['Tz_L', 'Tz_R'],
                            'VLL': ['VLL_L', 'VLL_R'],
                            'sp5': ['sp5'],
                           'outerContour': ['outerContour']}

labelMap_sidedToUnsided = {n: nu for nu, ns in labelMap_unsidedToSided.iteritems() for n in ns}

from itertools import chain
labels_sided = list(chain(*(labelMap_unsidedToSided[name_u] for name_u in labels_unsided)))
labels_sided_indices = dict((j, i+1) for i, j in enumerate(labels_sided)) # BackG always 0

############ Physical Dimension #############

section_thickness = 20 # in um
xy_pixel_distance_lossless = 0.46
xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail

#######################################

from enum import Enum

class PolygonType(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    TEXTURE = 'textured'
    TEXTURE_WITH_CONTOUR = 'texture with contour'
    DIRECTION = 'directionality'

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


def save_hdf(data, fn, complevel=9):
    filters = Filters(complevel=complevel, complib='blosc')
    with open_file(fn, mode="w") as f:
        _ = f.create_carray('/', 'data', Atom.from_dtype(data.dtype), filters=filters, obj=data)

def load_hdf(fn):
    with open_file(fn, mode="r") as f:
        data = f.get_node('/data').read()
    return data


def unique_rows(a, return_index=True):
    # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_a = a[idx]
    if return_index:
        return unique_a, idx
    else:
        return unique_a

def unique_rows2(a):
    ind = np.lexsort(a.T)
    return a[np.concatenate(([True],np.any(a[ind[1:]]!=a[ind[:-1]],axis=1)))]


def order_nodes(sps, neighbor_graph, verbose=False):

    from networkx.algorithms import dfs_successors, dfs_postorder_nodes


    subg = neighbor_graph.subgraph(sps)
    d_suc = dfs_successors(subg)

    x = [(a,b) for a,b in d_suc.iteritems() if len(b) == 2]

    if verbose:
        print 'root, two_leaves', x

    if len(x) == 0:
        trav = list(dfs_postorder_nodes(subg))
    else:
        if verbose:
            print 'd_succ'
            for it in d_suc.iteritems():
                print it

        root, two_leaves = x[0]

        left_branch = []
        right_branch = []

        c = two_leaves[0]
        left_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            left_branch.append(c)

        c = two_leaves[1]
        right_branch.append(c)
        while c in d_suc:
            c = d_suc[c][0]
            right_branch.append(c)

        trav = left_branch[::-1] + [root] + right_branch

        if verbose:
            print 'left_branch', left_branch
            print 'right_branch', right_branch

    return trav

def find_score_peaks(scores, min_size = 4, min_distance=10, threshold_rel=.3, threshold_abs=0, peakedness_lim=0,
                    peakedness_radius=1, verbose=False):

    from skimage.feature import peak_local_max

    scores2 = scores.copy()
    scores2[np.isnan(scores)] = np.nanmin(scores)
    scores = scores2

    if len(scores) > min_size:

        scores_shifted = scores[min_size-1:]
        scores_shifted_positive = scores_shifted - scores_shifted.min()

        peaks_shifted = np.atleast_1d(np.squeeze(peak_local_max(scores_shifted_positive,
                                    min_distance=min_distance, threshold_abs=threshold_abs-scores_shifted.min(), exclude_border=False)))

        # print peaks_shifted

        if len(peaks_shifted) == 0:
            high_peaks_sorted = np.array([np.argmax(scores)], np.int)
            high_peaks_peakedness = np.inf

        else:
            peaks_shifted = peaks_shifted[scores_shifted[peaks_shifted] >= np.max(scores_shifted) - threshold_rel]
            peaks_shifted = np.unique(np.r_[peaks_shifted, np.argmax(scores_shifted)])

            if verbose:
                print 'raw peaks', np.atleast_1d(np.squeeze(min_size - 1 + peaks_shifted))

            if len(peaks_shifted) > 0:
                peaks = min_size - 1 + peaks_shifted
            else:
                peaks = np.array([np.argmax(scores[min_size-1:]) + min_size-1], np.int)

            peakedness = np.zeros((len(peaks),))
            for i, p in enumerate(peaks):
                nbrs = np.r_[scores[max(min_size-1, p-peakedness_radius):p], scores[p+1:min(len(scores), p+1+peakedness_radius)]]
                assert len(nbrs) > 0
                peakedness[i] = scores[p]-np.mean(nbrs)

            if verbose:
                print 'peakedness', peakedness
                print 'filtered peaks', np.atleast_1d(np.squeeze(peaks))

            high_peaks = peaks[peakedness > peakedness_lim]
            high_peaks = np.unique(np.r_[high_peaks, min_size - 1 + np.argmax(scores_shifted)])

            high_peaks_order = scores[high_peaks].argsort()[::-1]
            high_peaks_sorted = high_peaks[high_peaks_order]

            high_peaks_peakedness = np.zeros((len(high_peaks),))
            for i, p in enumerate(high_peaks):
                nbrs = np.r_[scores[max(min_size-1, p-peakedness_radius):p], scores[p+1:min(len(scores), p+1+peakedness_radius)]]
                assert len(nbrs) > 0
                high_peaks_peakedness[i] = scores[p]-np.mean(nbrs)

    else:
        high_peaks_sorted = np.array([np.argmax(scores)], np.int)
        high_peaks_peakedness = np.inf

    return high_peaks_sorted, high_peaks_peakedness



all_stacks = ['MD589', 'MD594', 'MD593', 'MD585', 'MD592', 'MD590', 'MD591', 'MD589', 'MD595', 'MD598', 'MD602', 'MD603']

section_number_lookup = {'MD589': 445, 'MD594': 432, 'MD593': 448, 'MD585': 440, 'MD592': 454, \
                        'MD590': 419, 'MD591': 452, 'MD589': 445, 'MD595': 441, 'MD598': 430, 'MD602': 420, 'MD603': 432}
section_range_lookup = {'MD589': (93, 368), 'MD594': (93, 364), 'MD593': (69,350), 'MD585': (78, 347), 'MD592':(91,371), \
                        'MD590':(80,336), 'MD591': (98,387), 'MD589':(93,368), 'MD595': (67,330), 'MD598': (95,354), 'MD602':(96,352), 'MD603':(60,347)}

# xmin, ymin, w, h
brainstem_bbox_lookup = {'MD585': (610,113,445,408), 'MD593': (645,128,571,500), 'MD592': (807,308,626,407), 'MD590': (652,156,601,536), 'MD591': (697,194,550,665), \
                        'MD594': (616,144,451,362), 'MD595': (645,170,735,519), 'MD598': (680,107,695,459), 'MD602': (641,76,761,474), 'MD589':(643,145,419,367), 'MD603':(621,189,528,401)}

# xmin, ymin, w, h
detect_bbox_lookup = {'MD585': (16,144,411,225), 'MD593': (31,120,368,240), 'MD592': (43,129,419,241), 'MD590': (45,124,411,236), 'MD591': (38,117,410,272), \
                        'MD594': (29,120,422,242), 'MD595': (60,143,437,236), 'MD598': (48,118,450,231), 'MD602': (56,117,468,219), 'MD589': (0,137,419,230), 'MD603': (0,165,528,236)}

detect_bbox_range_lookup = {'MD585': (132,292), 'MD593': (127,294), 'MD592': (147,319), 'MD590': (135,280), 'MD591': (150,315), \
                        'MD594': (143,305), 'MD595': (115,279), 'MD598': (150,300), 'MD602': (147,302), 'MD589': (150,316), 'MD603': (130,290)}
# midline_section_lookup = {'MD589': 114, 'MD594': 119}

def find_z_section_map(stack, volume_zmin, downsample_factor = 16):

    section_thickness = 20 # in um
    xy_pixel_distance_lossless = 0.46
    xy_pixel_distance_tb = xy_pixel_distance_lossless * 32 # in um, thumbnail
    # factor = section_thickness/xy_pixel_distance_lossless

    xy_pixel_distance_downsampled = xy_pixel_distance_lossless * downsample_factor
    z_xy_ratio_downsampled = section_thickness / xy_pixel_distance_downsampled

    # build annotation volume
    section_bs_begin, section_bs_end = section_range_lookup[stack]
    print section_bs_begin, section_bs_end

    map_z_to_section = {}
    for s in range(section_bs_begin, section_bs_end+1):
        for z in range(int(z_xy_ratio_downsampled*s) - volume_zmin,
                       int(z_xy_ratio_downsampled*(s+1)) - volume_zmin + 1):
            map_z_to_section[z] = s

    return map_z_to_section


class DataManager(object):

    @staticmethod
    def get_image_filepath(stack, section, version, resol=None):

        data_dir = os.environ['DATA_DIR']

        if resol is None:
            resol = 'lossless'

        slice_str = '%04d' % section

        if version == 'rgb-jpg':
            image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped_downscaled')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled'])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif version == 'gray-jpg':
        #     image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale_downscaled')
        #     image_name = '_'.join([stack, slice_str, resol, 'warped'])
        #     image_path = os.path.join(image_dir, image_name + '.jpg')
        elif version == 'gray':
            image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped_grayscale')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_grayscale'])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif version == 'rgb':
            image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped'])
            image_path = os.path.join(image_dir, image_name + '.tif')

        elif version == 'stereotactic-rgb-jpg':
            image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped_downscaled_stereotactic')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled_stereotactic'])
            image_path = os.path.join(image_dir, image_name + '.jpg')

        return image_path

    def __init__(self, data_dir=os.environ['DATA_DIR'],
                 repo_dir=os.environ['REPO_DIR'],
                 labeling_dir=os.environ['LABELING_DIR'],
                 gabor_params_id=None,
                 segm_params_id='tSLIC200',
                 vq_params_id=None,
                 stack=None,
                 resol='lossless',
                 section=None,
                 load_mask=False):

        self.data_dir = data_dir
        self.repo_dir = repo_dir
        self.params_dir = os.path.join(repo_dir, 'params')

        self.root_labelings_dir = labeling_dir

        # self.labelnames_path = os.path.join(labeling_dir, 'labelnames.txt')

        # if os.path.isfile(self.labelnames_path):
        #     with open(self.labelnames_path, 'r') as f:
        #         self.labelnames = [n.strip() for n in f.readlines()]
        #         self.labelnames = [n for n in self.labelnames if len(n) > 0]
        # else:
        #     self.labelnames = []

        # self.root_results_dir = result_dir

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

        if self.resol == 'lossless':
            if hasattr(self, 'stack') and self.stack is not None:
                self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped')
                self.image_rgb_jpg_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped_downscaled')

        if section is not None:
            self.set_slice(section)
        else:
            try:
                random_image_fn = os.listdir(self.image_dir)[0]
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(self.image_dir, random_image_fn), shell=True).split('x'))
            except:
                d = os.path.join(self.data_dir, 'MD589_lossless_aligned_cropped_downscaled')
                if os.path.exists(d):
                    random_image_fn = os.listdir(d)[0]
                    self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(d, random_image_fn), shell=True).split('x'))

        if load_mask:
            self.thumbmail_mask = imread(self.data_dir+'/%(stack)s_thumbnail_aligned_cropped_mask/%(stack)s_%(slice_str)s_thumbnail_aligned_cropped_mask.png' % {'stack': self.stack, 'slice_str': self.slice_str})
            self.mask = rescale(self.thumbmail_mask.astype(np.bool), 32).astype(np.bool)
            # self.mask[:500, :] = False
            # self.mask[:, :500] = False
            # self.mask[-500:, :] = False
            # self.mask[:, -500:] = False

            xs_valid = np.any(self.mask, axis=0)
            ys_valid = np.any(self.mask, axis=1)
            self.xmin = np.where(xs_valid)[0][0]
            self.xmax = np.where(xs_valid)[0][-1]
            self.ymin = np.where(ys_valid)[0][0]
            self.ymax = np.where(ys_valid)[0][-1]

            self.h = self.ymax-self.ymin+1
            self.w = self.xmax-self.xmin+1


    def load_thumbnail_mask(self):
        self.thumbmail_mask = imread(self.data_dir+'/%(stack)s_thumbnail_aligned_mask_cropped/%(stack)s_%(slice_str)s_thumbnail_aligned_mask_cropped.png' % {'stack': self.stack,
            'slice_str': self.slice_str}).astype(np.bool)
        return self.thumbmail_mask

    def add_labelnames(self, labelnames, filename):
        existing_labelnames = {}
        with open(filename, 'r') as f:
            for ln in f.readlines():
                abbr, fullname = ln.split('\t')
                existing_labelnames[abbr] = fullname.strip()

        with open(filename, 'a') as f:
            for abbr, fullname in labelnames.iteritems():
                if abbr not in existing_labelnames:
                    f.write(abbr+'\t'+fullname+'\n')

    def set_stack(self, stack):
        self.stack = stack
        self.get_image_dimension()
#         self.stack_path = os.path.join(self.data_dir, self.stack)
#         self.slice_ind = None

    def set_resol(self, resol):
        self.resol = resol

    def get_image_dimension(self):

        try:
            if hasattr(self, 'image_path') and os.path.exists(self.image_path):
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))
            else:
                # sys.stderr.write('original TIFF image is not available. Loading downscaled jpg instead...')

                # if section is specified, use that section; otherwise use a random section in the brainstem range
                if hasattr(self, 'slice_ind') and self.slice_ind is not None:
                    sec = self.slice_ind
                else:
                    sec = section_range_lookup[self.stack][0]

                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self._get_image_filepath(section=sec, version='rgb-jpg'), shell=True).split('x'))

        except Exception as e:
            print e
            sys.stderr.write('Cannot find image\n')

        return self.image_height, self.image_width

    def set_slice(self, slice_ind):
        assert self.stack is not None and self.resol is not None, 'Stack is not specified'
        self.slice_ind = slice_ind
        self.slice_str = '%04d' % slice_ind
        if self.resol == 'lossless':
            self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped')
            self.image_name = '_'.join([self.stack, self.slice_str, self.resol])
            self.image_path = os.path.join(self.image_dir, self.image_name + '_aligned_cropped.tif')

        try:
            if os.path.exists(self.image_path):
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))
            else:
                # sys.stderr.write('original TIFF image is not available. Loading downscaled jpg instead...')
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self._get_image_filepath(version='rgb-jpg'), shell=True).split('x'))
        except Exception as e:
            print e
            sys.stderr.write('Cannot find image\n')

        # self.labelings_dir = os.path.join(self.image_dir, 'labelings')

        if hasattr(self, 'result_list'):
            del self.result_list

        self.labelings_dir = os.path.join(self.root_labelings_dir, self.stack, self.slice_str)
        # if not os.path.exists(self.labelings_dir):
        #     os.makedirs(self.labelings_dir)

#         self.results_dir = os.path.join(self.image_dir, 'pipelineResults')

        # self.results_dir = os.path.join(self.root_results_dir, self.stack, self.slice_str)
        # if not os.path.exists(self.results_dir):
        #     os.makedirs(self.results_dir)


    def _get_image_filepath(self, stack=None, resol='lossless', section=None, version='rgb-jpg'):
        if stack is None:
            stack = self.stack
        if resol is None:
            resol = self.resol
        if section is None:
            section = self.slice_ind

        slice_str = '%04d' % section

        if version == 'rgb-jpg':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_downscaled')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled'])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif version == 'gray-jpg':
        #     image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale_downscaled')
        #     image_name = '_'.join([stack, slice_str, resol, 'warped'])
        #     image_path = os.path.join(image_dir, image_name + '.jpg')
        elif version == 'gray':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_grayscale')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_grayscale'])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif version == 'rgb':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped'])
            image_path = os.path.join(image_dir, image_name + '.tif')

        elif version == 'stereotactic-rgb-jpg':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_downscaled_stereotactic')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled_stereotactic'])
            image_path = os.path.join(image_dir, image_name + '.jpg')

        return image_path

    def _read_image(self, image_filename):
        if image_filename.endswith('tif') or image_filename.endswith('tiff'):
            from PIL.Image import open
            img = np.array(open(image_filename))/255.
        else:
            img = imread(image_filename)
        return img

    def _load_image(self, versions=['rgb', 'gray', 'rgb-jpg'], force_reload=True):

        assert self.image_name is not None, 'Image is not specified'

        if 'rgb-jpg' in versions:
            if force_reload or not hasattr(self, 'image_rgb_jpg'):
                image_filename = self._get_image_filepath(version='rgb-jpg')
                # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
                self.image_rgb_jpg = self._read_image(image_filename)

        if 'rgb' in versions:
            if force_reload or not hasattr(self, 'image_rgb'):
                image_filename = self._get_image_filepath(version='rgb')
                # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
                self.image_rgb = self._read_image(image_filename)

        if 'gray' in versions and not hasattr(self, 'image'):
            if force_reload or not hasattr(self, 'gray'):
                image_filename = self._get_image_filepath(version='gray')
                # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
                self.image = self._read_image(image_filename)


    # def get_labeling_path(stack, section, username, timestamp, suffix='consolidated', labeling_list):




    #     if len(labeling_list[username]) == 0:
    #         self.reload_labelings()

    #     if username is None: # search labelings of any user
    #         self.result_list_flatten = [(usr, ts) for usr, timestamps in self.result_list.iteritems() for ts in timestamps ] # [(username, timestamp)..]
    #         if len(self.result_list_flatten) == 0:
    #             # sys.stderr.write('username is empty\n')
    #             return None

    #     if timestamp == 'latest':
    #         if username is not None:

    #             if len(self.result_list[username]) == 0:
    #                 return None

    #             timestamps_sorted = map(itemgetter(1), sorted(map(lambda s: (datetime.datetime.strptime(s, "%m%d%Y%H%M%S"), s), self.result_list[username]), reverse=True))
    #             timestamp = timestamps_sorted[0]
    #         else:
    #             ts_str_usr_sorted = sorted([(datetime.datetime.strptime(ts, "%m%d%Y%H%M%S"), ts, usr) for usr, ts in self.result_list_flatten], reverse=True)
    #             timestamp = ts_str_usr_sorted[0][1]
    #             username = ts_str_usr_sorted[0][2]

    #     return os.path.join(self.labelings_dir, '_'.join([stack, '%04d'%section, username, timestamp]) + '_'+suffix+'.pkl'), username, timestamp


    # def get_labeling_path(self, username, timestamp, stack=None, section=None, suffix='consolidated'):

    #     if stack is None:
    #         stack = self.stack
    #     if section is None:
    #         section = self.slice_ind

    #     if not hasattr(self, 'result_list') or len(self.result_list[username]) == 0:
    #         self.reload_labelings()

    #     if username is None: # search labelings of any user
    #         self.result_list_flatten = [(usr, ts) for usr, timestamps in self.result_list.iteritems() for ts in timestamps ] # [(username, timestamp)..]
    #         if len(self.result_list_flatten) == 0:
    #             # sys.stderr.write('username is empty\n')
    #             return None

    #     if timestamp == 'latest':
    #         if username is not None:

    #             if len(self.result_list[username]) == 0:
    #                 return None

    #             timestamps_sorted = map(itemgetter(1), sorted(map(lambda s: (datetime.datetime.strptime(s, "%m%d%Y%H%M%S"), s), self.result_list[username]), reverse=True))
    #             timestamp = timestamps_sorted[0]
    #         else:
    #             ts_str_usr_sorted = sorted([(datetime.datetime.strptime(ts, "%m%d%Y%H%M%S"), ts, usr) for usr, ts in self.result_list_flatten], reverse=True)
    #             timestamp = ts_str_usr_sorted[0][1]
    #             username = ts_str_usr_sorted[0][2]

    #     return os.path.join(self.labelings_dir, '_'.join([stack, '%04d'%section, username, timestamp]) + '_'+suffix+'.pkl'), username, timestamp


    # def reload_labelings(self):
    #     # if not hasattr(self, 'result_list'):
    #     from collections import defaultdict
    #     import pandas as pd

    #     pd.DataFrame()


    #     # self.result_list = defaultdict(lambda: defaultdict(list))
    #     self.result_list = defaultdict(list)

    #     if os.path.exists(self.labelings_dir):
    #         for fn in os.listdir(self.labelings_dir):
    #             st, se, us, ts, suf = fn[:-4].split('_')
    #             # self.result_list[us][ts].append(suf)
    #             self.result_list[us].append(ts)


    # def load_labeling(self, username, timestamp, suffix, stack=None, section=None):

    #     ret = self.get_labeling_path(username=username, timestamp=timestamp, stack=stack, section=section)
    #     if ret is not None:
    #         path, usr, timestmp
    #         return (usr, ts, suffix, pickle.load( open(path, 'r')))

        # if stack is None:
        #     stack = self.stack
        # if section is None:
        #     section = self.slice_ind

        # if not hasattr(self, 'result_list') or len(self.result_list[username]) == 0:
        #     self.reload_labelings()

        # if username is not None:
        #     if len(self.result_list[username]) == 0:
        #         sys.stderr.write('username %s does not have any annotations for current section %d \n' % (username, self.slice_ind))
        #         return None

        # if suffix == 'all':
        #     results = []
        #     for suf in self.result_list[username][timestamp]:
        #         ret = self.load_review_result_path(username=username, timestamp=timestamp, suffix=suf, stack=stack, section=section)
        #         if ret is None:
        #             return None
        #         else:
        #             path, usr, ts = ret
        #             results.append((username, timestamp, suf, pickle.load(open(path, 'r'))))
        #     return results
        # else:
        #     ret = self.load_review_result_path(username=username, timestamp=timestamp, suffix=suffix, stack=stack, section=section)
        #     if ret is None:
        #         return None
        #     else:
        #         path, usr, ts = ret
        #         return (usr, ts, suffix, pickle.load( open(path, 'r')))


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


def fit_ellipse_to_points(pts):

    pts = np.array(list(pts) if isinstance(pts, set) else pts)

    c0 = pts.mean(axis=0)

    coords0 = pts - c0

    U,S,V = np.linalg.svd(np.dot(coords0.T, coords0)/coords0.shape[0])
    v1 = U[:,0]
    v2 = U[:,1]
    s1 = np.sqrt(S[0])
    s2 = np.sqrt(S[1])

    return v1, v2, s1, s2, c0


def scores_to_vote(scores):
    vals = np.unique(scores)
    d = dict(zip(vals, np.linspace(0, 1, len(vals))))
    votes = np.array([d[s] for s in scores])
    votes = votes/votes.sum()
    return votes


def display_image(vis, filename='tmp.jpg'):

    if vis.dtype != np.uint8:
        imsave(filename, img_as_ubyte(vis))
    else:
        imsave(filename, vis)

    from IPython.display import FileLink
    return FileLink(filename)

def display_images_in_grids(vizs, nc, titles=None, export_fn=None):

    n = len(vizs)
    nr = int(np.ceil(n/float(nc)))
    aspect_ratio = vizs[0].shape[1]/float(vizs[0].shape[0]) # width / height

    fig, axes = plt.subplots(nr, nc, figsize=(nc*5*aspect_ratio, nr*5))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i >= n:
            axes[i].axis('off');
        else:
            axes[i].imshow(vizs[i]);
            if titles is not None:
                axes[i].set_title(titles[i], fontsize=20);
            axes[i].set_xticks([]);
            axes[i].set_yticks([]);

    fig.tight_layout();

    if export_fn is not None:
        plt.savefig(export_fn);

    plt.show();

# <codecell>

# import numpy as np
# from scipy.ndimage.filters import maximum_filter
# from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

# def detect_peaks(image):
#     """
#     Takes an image and detect the peaks usingthe local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """

#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,2)

#     #apply the local maximum filter; all pixel of maximal value
#     #in their neighborhood are set to 1
#     local_max = maximum_filter(image, footprint=neighborhood)==image
#     #local_max is a mask that contains the peaks we are
#     #looking for, but also the background.
#     #In order to isolate the peaks we must remove the background from the mask.

#     #we create the mask of the background
#     background = (image==0)

#     #a little technicality: we must erode the background in order to
#     #successfully subtract it form local_max, otherwise a line will
#     #appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

#     #we obtain the final mask, containing only peaks,
#     #by removing the background from the local_max mask
#     detected_peaks = local_max - eroded_background

#     return detected_peaks

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
    """
    h1s is n x n_texton
    MUST be float type
    """
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


def bbox_2d(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, cmax, rmin, rmax

def bbox_3d(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return cmin, cmax, rmin, rmax, zmin, zmax

# def sample(points, num_samples):
#     n = len(points)
#     sampled_indices = np.random.choice(range(n), min(num_samples, n), replace=False)
#     return points[sampled_indices]
