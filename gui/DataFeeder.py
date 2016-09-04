#! /usr/bin/env python

import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

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

import numpy as np

from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
if use_pyside:
    #print 'Using PySide'
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    #print 'Using PyQt4'
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import DataManager
from metadata import *

from collections import defaultdict, OrderedDict, deque

gray_color_table = [qRgb(i, i, i) for i in range(256)]

class ImageDataFeeder(object):

    def __init__(self, name, stack, sections, version='aligned_cropped', use_data_manager=True):
        self.name = name
        self.stack = stack

        if use_data_manager:
            # index in stack
            self.sections = sections
            self.min_section = min(self.sections)
            self.max_section = max(self.sections)

            self.first_section, self.last_section = section_range_lookup[stack]
            self.all_sections = range(self.first_section, self.last_section+1)
        else:
            # macro index
            self.sections = sections
            self.all_sections = sections

        # self.n = len(self.sections)
        self.n = len(self.all_sections)

        self.supported_downsample_factors = [1, 32]
        self.image_cache = {}

        self.version = version

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.compute_dimension()

    def set_image(self, qimage, sec, downsample=None):

        if downsample is None:
            downsample = self.downsample

        if downsample not in self.image_cache:
            self.image_cache[downsample] = {}

        self.image_cache[downsample][sec] = qimage
        self.compute_dimension()

    def load_images(self, downsample=None, selected_sections=None):
        if downsample is None:
            downsample = self.downsample

        if selected_sections is None:
            selected_sections = self.sections

        if self.version == 'aligned_cropped':
            if downsample == 1:
                self.image_cache[downsample] = {sec: QImage(DataManager.get_image_filepath(stack=self.stack, section=sec, version='rgb-jpg', data_dir=data_dir))
                                                for sec in selected_sections}
            elif downsample == 32:
                self.image_cache[downsample] = {sec: QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnail_aligned_cropped.tif' \
                                                % dict(stack=self.stack, sec=sec))
                                                for sec in selected_sections}
            else:
                raise Exception('Not implemented.')
        elif self.version == 'aligned':
            if downsample == 32:
                self.image_cache[downsample] = {sec: QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned/%(stack)s_%(sec)04d_thumbnail_aligned.tif' \
                                                % dict(stack=self.stack, sec=sec))
                                                for sec in selected_sections}
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

        if downsample is None:
            downsample = self.downsample

        if downsample not in self.supported_downsample_factors:
            sys.stderr.write('Downsample factor %d is not supported.' % downsample)
            return

        if sec is None:
            # sec = self.sections[i]
            sec = self.all_sections[i]

        # if downsample not in self.image_cache:
        #     t = time.time()
        #     self.load_images(downsample)
        #     sys.stderr.write('Load images: %.2f\n' % (time.time() - t))

        # if sec not in self.image_cache[downsample]:
        #     if downsample == 1:
        #         self.image_cache[downsample][sec] = QImage(DataManager.get_image_filepath(stack=self.stack, section=sec, version='rgb-jpg', data_dir=data_dir))
        #     elif downsample == 32:
        #         # self.image_cache[downsample][i] = QImage(DataManager.get_image_filepath(stack=self.stack, section=i, resol='thumbnail', version='rgb', data_dir=data_dir))
        #         self.image_cache[downsample][sec] = QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnnail_aligned_cropped.tif'
        #                                         % dict(stack=self.stack, sec=sec))
        #

        if downsample not in self.image_cache:
            self.image_cache[downsample] = {}

        if sec not in self.image_cache[downsample]:
            # sys.stderr.write('Image is not loaded.\n')
            raise Exception('Image is not loaded.')
            # raise Exception('Image is not loaded.')

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
