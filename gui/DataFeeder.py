#! /usr/bin/env python

import sys
import os
import datetime
from random import random
import subprocess
import time
import json
from pprint import pprint
import cPickle as pickle
from itertools import groupby
from operator import itemgetter
from collections import defaultdict, OrderedDict, deque

import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from qt_utilities import *

gray_color_table = [qRgb(i, i, i) for i in range(256)]

class ImageDataFeeder(object):

    def __init__(self, name, stack, sections=None, version='aligned_cropped', use_data_manager=True, downscale=None, labeled_filenames=None):
        self.name = name
        self.stack = stack

        if use_data_manager:
            # index in stack
            assert sections is not None
            self.sections = sections
            self.min_section = min(self.sections)
            self.max_section = max(self.sections)
            # self.first_section, self.last_section = metadata_cache['section_limits'][stack]
            # self.all_sections = range(self.first_section, self.last_section+1)
        else:
            # macro index
            self.sections = sections
            # self.all_sections = sections

        self.n = len(self.sections)

        self.supported_downsample_factors = [1, 32]
        self.image_cache = {}

        self.version = version

        if downscale is not None:
            self.set_downsample_factor(downscale)

        if labeled_filenames is not None:
            self.set_images(labeled_filenames=labeled_filenames)
        elif use_data_manager:
            self.load_images()

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.compute_dimension()

    def set_image(self, sec, qimage=None, numpy_image=None, fp=None, downsample=None):

        if downsample is None:
            downsample = self.downsample

        if downsample not in self.image_cache:
            self.image_cache[downsample] = {}

        # self.sections.append(sec)
        # self.all_sections.append(sec)

        if qimage is None:
            if fp is not None:
                qimage = QImage(fp)
            elif numpy_image is not None:
                qimage = numpy_to_qimage(numpy_image)
            else:
                raise Exception('Either filepath or numpy_image must be provided.')

        self.image_cache[downsample][sec] = qimage
        self.compute_dimension()

    def set_images(self, labels=None, filenames=None, labeled_filenames=None, downsample=None, load_with_cv2=False):

        if labeled_filenames is not None:
            assert isinstance(labeled_filenames, dict)
            labels = labeled_filenames.keys()
            filenames = labeled_filenames.values()

        for lbl, fn in zip(labels, filenames):
            if load_with_cv2: # For loading output tif images from elastix, directly QImage() causes "foo: Can not read scanlines from a tiled image."
                # print fn
                img = cv2.imread(fn)
                if img is None:
                    continue

                qimage = numpy_to_qimage(img)

                # h, w = img.shape[:2]
                # if img.ndim == 3:
                #     qimage = QImage(img.flatten(), w, h, 3*w, QImage.Format_RGB888)
                # else:
                #     qimage = QImage(img.flatten(), w, h, w, QImage.Format_Indexed8)
                #     qimage.setColorTable(gray_color_table)
            else:
                qimage = QImage(fn)
            self.set_image(qimage=qimage, sec=lbl, downsample=downsample)


    def load_images(self, downsample=None, selected_sections=None):
        """
        If use_data_manager, use this function to load images.
        """

        if downsample is None:
            downsample = self.downsample

        if selected_sections is None:
            selected_sections = self.sections

        if self.version == 'aligned_cropped':
            if downsample == 1:
                self.image_cache[downsample] = {sec: QImage(DataManager.get_image_filepath(stack=self.stack, section=sec, version='rgb-jpg', data_dir=data_dir))
                                                for sec in selected_sections}
            elif downsample == 32:
                # self.image_cache[downsample] = {sec: QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnail_aligned_cropped.tif' \
                #                                 % dict(stack=self.stack, sec=sec))
                #                                 for sec in selected_sections}
                self.image_cache[downsample] = {sec: QImage(DataManager.get_image_filepath(stack=self.stack, section=sec, version='cropped', resol='thumbnail'))
                                                for sec in selected_sections}
            else:
                raise Exception('Not implemented.')
        elif self.version == 'aligned':
            if downsample == 32:
                self.image_cache[downsample] = {sec: QImage(DataManager.get_image_filepath(stack=self.stack, section=sec, version='aligned_tif', resol='thumbnail'))
                                                for sec in selected_sections}
                # anchor_fn = DataManager.load_anchor_filename(stack=self.stack)
                # self.image_cache[downsample] = {sec: QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned/%(stack)s_%(sec)04d_thumbnail_aligned.tif' \
                #                                 % dict(stack=self.stack, sec=sec))
                #                                 for sec in selected_sections}
            else:
                raise Exception('Not implemented.')
        elif self.version == 'original':
            if downsample == 32:
                self.image_cache[downsample] = {sec: QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_renamed/%(stack)s_%(sec)04d_thumbnail.tif' \
                                                % dict(stack=self.stack, sec=sec))
                                                for sec in selected_sections}
            else:
                raise Exception('Not implemented.')
        else:
            raise Exception('Not implemented.')

        self.compute_dimension()


    def compute_dimension(self):

        if hasattr(self, 'downsample') and self.downsample in self.image_cache and hasattr(self, 'orientation'):
            arbitrary_img = self.image_cache[self.downsample].values()[0]
            if self.orientation == 'sagittal':
                self.x_dim = arbitrary_img.width()
                self.y_dim = arbitrary_img.height()
            elif self.orientation == 'coronal':
                self.z_dim = arbitrary_img.width()
                self.y_dim = arbitrary_img.height()
            elif self.orientation == 'horizontal':
                self.x_dim = arbitrary_img.width()
                self.z_dim = arbitrary_img.height()

    def set_downsample_factor(self, downsample):
        if downsample not in self.supported_downsample_factors:
            sys.stderr.write('Downsample factor %d is not supported.' % downsample)
            return
        else:
            self.downsample = downsample

        self.compute_dimension()

    def retrive_i(self, i=None, sec=None, downsample=None):
        """
        Retrieve the i'th image in self.sections.
        """

        if downsample is None:
            downsample = self.downsample

        if downsample not in self.supported_downsample_factors:
            sys.stderr.write('Downsample factor %d is not supported.' % downsample)
            return

        if sec is None:
            sec = self.sections[i]

        if downsample not in self.image_cache:
            self.image_cache[downsample] = {}

        if sec not in self.image_cache[downsample]:
            raise Exception('Image is not loaded: %d' % sec)

        return self.image_cache[downsample][sec]


class VolumeResectionDataFeeder(object):

    def __init__(self, name, stack):

        self.name = name
        self.volume_cache = {}
        self.supported_downsample_factors = [4,8,32]

        self.stack = stack

    def set_downsample_factor(self, downsample):
        if downsample not in self.supported_downsample_factors:
            sys.stderr.write('Downsample factor %d is not supported.\n' % downsample)
            return
        else:
            self.downsample = downsample

        if self.downsample in self.volume_cache:
            self.volume = self.volume_cache[self.downsample]
            self.y_dim, self.x_dim, self.z_dim = self.volume.shape

            if self.orientation == 'sagittal':
                # self.min_i = 0
                # self.max_i = self.z_dim - 1
                self.n = self.z_dim
            elif self.orientation == 'coronal':
                # self.min_i = 0
                # self.max_i = self.x_dim - 1
                self.n = self.x_dim
            elif self.orientation == 'horizontal':
                # self.min_i = 0
                # self.max_i = self.y_dim - 1
                self.n = self.y_dim
        else:
            try:
                self.volume_cache[self.downsample] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':self.downsample})
            except:
                sys.stderr.write('Cannot read volume file.\n')

    def set_volume_cache(self, volume_cache):
        self.volume_cache = volume_cache

    def add_volume(self, volume, downsample):
        if downsample not in self.volume_cache:
            self.volume_cache[downsample] = volume

    def set_orientation(self, orientation):
        self.orientation = orientation

    def retrive_i(self, i, orientation=None, downsample=None):
        if orientation is None:
            orientation = self.orientation

        if downsample is None:
            downsample = self.downsample

        if orientation == 'horizontal':
            y = i
            horizontal_data = self.volume[y, :, ::-1].T.flatten()
            horizontal_image = QImage(horizontal_data, self.x_dim, self.z_dim, self.x_dim, QImage.Format_Indexed8)
            horizontal_image.setColorTable(gray_color_table)
            return horizontal_image

        elif orientation == 'coronal':
            x = i
            coronal_data = self.volume[:, x, ::-1].flatten()
            coronal_image = QImage(coronal_data, self.z_dim, self.y_dim, self.z_dim, QImage.Format_Indexed8)
            coronal_image.setColorTable(gray_color_table)
            return coronal_image

        elif orientation == 'sagittal':
            z = i
            sagittal_data = self.volume[:, :, z].flatten()
            sagittal_image = QImage(sagittal_data, self.x_dim, self.y_dim, self.x_dim, QImage.Format_Indexed8)
            sagittal_image.setColorTable(gray_color_table)
            return sagittal_image
