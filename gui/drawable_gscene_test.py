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

from ui_BrainLabelingGui_v15 import Ui_BrainLabelingGui
# from ui_RectificationTool import Ui_RectificationGUI
# from rectification_tool import *

from matplotlib.colors import ListedColormap, NoNorm, ColorConverter

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import LinearRing as ShapelyLineRing

from skimage.color import label2rgb

from gui_utilities import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import DataManager
from metadata import *

from collections import defaultdict, OrderedDict, deque

from operator import attrgetter

import requests

from joblib import Parallel, delayed

# from LabelingPainter import LabelingPainter
from custom_widgets import *
from SignalEmittingItems import *


from drawable_gscene import *

gray_color_table = [qRgb(i, i, i) for i in range(256)]

#######################################################################


# class DataFeeder(object):
#     def __init__(self, name):
#         super(self.__class__, self).__init__()
#         self.name = name

        # def set_downsample_factor(self, downsample):
        #     self.downsample = downsample

class ImageDataFeeder(object):

    def __init__(self, name, stack, sections):
        self.name = name
        self.stack = stack

        self.sections = sections
        self.min_section = min(self.sections)
        self.max_section = max(self.sections)

        self.n = len(self.sections)

        self.supported_downsample_factors = [1, 32]
        self.image_cache = {}

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.compute_dimension()

    def load_images(self, downsample=None):
        if downsample is None:
            downsample = self.downsample

        if downsample == 1:
            self.image_cache[downsample] = {sec: QImage(DataManager.get_image_filepath(stack=self.stack, section=sec, version='rgb-jpg', data_dir=data_dir))
                            for sec in self.sections}
        elif downsample == 32:
            # self.image_cache[downsample] = {sec: QImage(DataManager.get_image_filepath(stack=self.stack, section=i, resol='thumbnail', version='rgb', data_dir=data_dir))
            #                 for sec in self.sections}
            self.image_cache[downsample] = {sec: QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnnail_aligned_cropped.tif'
                                            % dict(stack=self.stack, sec=sec))
                                            for sec in self.sections}

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

    def retrive_i(self, i, downsample=None):

        if downsample is None:
            downsample = self.downsample

        if downsample not in self.supported_downsample_factors:
            sys.stderr.write('Downsample factor %d is not supported.' % downsample)
            return

        if downsample not in self.image_cache:
            self.load_images(downsample)

        sec = self.sections[i]

        if sec not in self.image_cache[downsample]:
            if downsample == 1:
                self.image_cache[downsample][sec] = QImage(DataManager.get_image_filepath(stack=self.stack, section=sec, version='rgb-jpg', data_dir=data_dir))
            elif downsample == 32:
                # self.image_cache[downsample][i] = QImage(DataManager.get_image_filepath(stack=self.stack, section=i, resol='thumbnail', version='rgb', data_dir=data_dir))
                self.image_cache[downsample][sec] = QImage('/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnnail_aligned_cropped.tif'
                                                % dict(stack=self.stack, sec=sec))

        return self.image_cache[downsample][sec]


class VolumeResectionDataFeeder(object):

    def __init__(self, name):

        self.name = name
        self.volume_cache = {}
        self.supported_downsample_factors = [4,8,32]

    def set_downsample_factor(self, downsample):
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

    def set_volume_cache(self, volume_cache):
        self.volume_cache = volume_cache

    # def set_volume(self, volume):
    #     self.volume = volume
    #     self.y_dim, self.x_dim, self.z_dim = volume.shape

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


class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
# class BrainLabelingGUI(QMainWindow, Ui_RectificationGUI):

    def __init__(self, parent=None, stack=None):
        """
        Initialization of BrainLabelingGUI.
        """
        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.stack = stack
        self.setupUi(self)

        from collections import defaultdict
        self.structure_volumes = {}

        self.sections = range(180, 190)

        # first_sec, last_sec = section_range_lookup[self.stack]
        # self.sections = range(first, last+1)

        self.volume_cache = {32: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':32}),
                            8: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':8})}

        self.downsample_factor = 32

        self.volume = self.volume_cache[self.downsample_factor]
        self.y_dim, self.x_dim, self.z_dim = self.volume.shape

        self.coronal_gscene = DrawableGraphicsScene(id='coronal', gui=self, gview=self.coronal_gview)
        self.coronal_gview.setScene(self.coronal_gscene)

        self.horizontal_gscene = DrawableGraphicsScene(id='horizontal', gui=self, gview=self.horizontal_gview)
        self.horizontal_gview.setScene(self.horizontal_gscene)

        self.sagittal_gscene = DrawableGraphicsScene(id='sagittal', gui=self, gview=self.sagittal_gview)
        self.sagittal_gview.setScene(self.sagittal_gscene)

        self.gscenes = {'coronal': self.coronal_gscene, 'sagittal': self.sagittal_gscene, 'horizontal': self.horizontal_gscene}

        for gscene in self.gscenes.itervalues():
            gscene.drawings_updated.connect(self.drawings_updated)
            gscene.crossline_updated.connect(self.crossline_updated)
            gscene.set_structure_volumes(self.structure_volumes)
            # gscene.set_drawings(self.drawings)

        from functools import partial
        self.gscenes['sagittal'].set_conversion_func_section_to_z(partial(DataManager.convert_section_to_z, stack=self.stack))
        self.gscenes['sagittal'].set_conversion_func_z_to_section(partial(DataManager.convert_z_to_section, stack=self.stack))

        ##################
        self.slider_downsample.valueChanged.connect(self.downsample_factor_changed)

        ###################
        self.contextMenu_set = True

        self.recent_labels = []

        self.structure_names = load_structure_names(os.environ['REPO_DIR']+'/gui/structure_names.txt')
        self.new_labelnames = load_structure_names(os.environ['REPO_DIR']+'/gui/newStructureNames.txt')
        self.structure_names = OrderedDict(sorted(self.new_labelnames.items()) + sorted(self.structure_names.items()))

        self.installEventFilter(self)

        self.keyPressEvent = self.key_pressed

        # self.gscenes['coronal'].set_data({'volume': self.volume})
        # self.gscenes['sagittal'].set_data({'volume': self.volume})
        # self.gscenes['horizontal'].set_data({'volume': self.volume})

        # from functools import partial

        image_feeder = ImageDataFeeder('image feeder', stack=self.stack, sections=self.sections)
        image_feeder.set_orientation('sagittal')
        image_feeder.set_downsample_factor(1)
        self.gscenes['sagittal'].set_data_feeder(image_feeder)

        volume_resection_feeder = VolumeResectionDataFeeder('volume resection feeder')

        coronal_volume_resection_feeder = VolumeResectionDataFeeder('coronal resection feeder')
        coronal_volume_resection_feeder.set_volume_cache(self.volume_cache)
        coronal_volume_resection_feeder.set_orientation('coronal')
        coronal_volume_resection_feeder.set_downsample_factor(self.downsample_factor)
        self.gscenes['coronal'].set_data_feeder(coronal_volume_resection_feeder)

        horizontal_volume_resection_feeder = VolumeResectionDataFeeder('horizontal resection feeder')
        horizontal_volume_resection_feeder.set_volume_cache(self.volume_cache)
        horizontal_volume_resection_feeder.set_orientation('horizontal')
        horizontal_volume_resection_feeder.set_downsample_factor(self.downsample_factor)
        self.gscenes['horizontal'].set_data_feeder(horizontal_volume_resection_feeder)

        # self.gscenes['coronal'].set_downsample_factor(self.downsample_factor)
        # self.gscenes['sagittal'].set_downsample_factor(1)
        # self.gscenes['horizontal'].set_downsample_factor(self.downsample_factor)

        self.gscenes['coronal'].set_active_i(50)
        self.gscenes['sagittal'].set_active_section(181)
        self.gscenes['horizontal'].set_active_i(150)


    @pyqtSlot(int)
    def downsample_factor_changed(self, val):

        # self.cross_x_lossless = self.cross_x * self.downsample_factor
        # self.cross_y_lossless = self.cross_y * self.downsample_factor
        # self.cross_z_lossless = self.cross_z * self.downsample_factor

        print val

        if val == 0:
            self.downsample_factor = 32
        elif val == 1:
            self.downsample_factor = 8
        elif val == 2:
            self.downsample_factor = 4
            if 4 not in self.volume_cache:
                self.volume_cache[4] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':4})
                self.statusBar().showMessage('Volume loaded.')

        self.gscenes['coronal'].set_downsample_factor(self.downsample_factor)
        self.gscenes['sagittal'].set_downsample_factor(self.downsample_factor)
        self.gscenes['horizontal'].set_downsample_factor(self.downsample_factor)

        self.gscenes['coronal'].set_data_feeder(self.resection_volume_feeder, volume=self.volume_cache[self.downsample_factor], orientation='coronal')
        self.gscenes['sagittal'].set_data_feeder(self.resection_volume_feeder, volume=self.volume_cache[self.downsample_factor], orientation='sagittal')
        self.gscenes['horizontal'].set_data_feeder(self.resection_volume_feeder, volume=self.volume_cache[self.downsample_factor], orientation='horizontal')

        self.gscenes['coronal'].update_image()
        self.gscenes['sagittal'].update_image()
        self.gscenes['horizontal'].update_image()

        # self.update_cross(self.cross_x_lossless / self.downsample_factor, self.cross_y_lossless / self.downsample_factor, self.cross_z_lossless / self.downsample_factor)
        # self.update_gscenes('x')
        # self.update_gscenes('y')
        # self.update_gscenes('z')

    @pyqtSlot(object)
    def crossline_updated(self, cross_x_lossless, cross_y_lossless, cross_z_lossless):
        for gscene in self.gscenes.itervalues():
            gscene.update_cross(cross_x_lossless, cross_y_lossless, cross_z_lossless)


    @pyqtSlot(object)
    def drawings_updated(self, polygon):

        print 'Drawings updated.'

        name_u = polygon.label

        downsample = polygon.gscene.data_feeder.downsample

        matched_polygons = [p for i, polygons in polygon.gscene.drawings.iteritems() for p in polygons if p.label == name_u]
        if len(matched_polygons) >= 2:
            contour_points_grouped_by_pos = {p.position: [(c.scenePos().x()*downsample, c.scenePos().y()*downsample)
                                                            for c in p.vertex_circles] for p in matched_polygons}

            if polygon.gscene.data_feeder.orientation == 'sagittal':
                volume, bbox = interpolate_contours_to_volume(contour_points_grouped_by_pos, 'z')
            elif polygon.gscene.data_feeder.orientation == 'coronal':
                volume, bbox = interpolate_contours_to_volume(contour_points_grouped_by_pos, 'x')
            elif polygon.gscene.data_feeder.orientation == 'horizontal':
                volume, bbox = interpolate_contours_to_volume(contour_points_grouped_by_pos, 'y')

            # Here bbox must be of the original dimension.

            self.structure_volumes[name_u] = (volume, bbox)

            self.gscenes['coronal'].update_drawings_from_structure_volume(name_u)
            self.gscenes['horizontal'].update_drawings_from_structure_volume(name_u)


    def key_pressed(self, event):
        # if event.type() == Qt.keyPressEvent:
        key = event.key()
        if key == Qt.Key_1:
            self.gscenes['sagittal'].show_previous()
        elif key == Qt.Key_2:
            self.gscenes['sagittal'].show_next()
        elif key == Qt.Key_3:
            self.gscenes['coronal'].show_previous()
        elif key == Qt.Key_4:
            self.gscenes['coronal'].show_next()
        elif key == Qt.Key_5:
            self.gscenes['horizontal'].show_previous()
        elif key == Qt.Key_6:
            self.gscenes['horizontal'].show_next()

        elif key == Qt.Key_Space:
            for gscene in self.gscenes.itervalues():
                if gscene.mode == 'crossline':
                    gscene.set_mode('idle')
                else:
                    gscene.set_mode('crossline')


    def eventFilter(self, obj, event):
        # print obj.metaObject().className(), event.type()
        return False




def load_structure_names(fn):
    names = {}
    with open(fn, 'r') as f:
        for ln in f.readlines():
            abbr, fullname = ln.split('\t')
            names[abbr] = fullname.strip()
    return names


if __name__ == "__main__":

    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Launch brain labeling GUI.')

    parser.add_argument("stack_name", type=str, help="stack name")
    parser.add_argument("-n", "--num_neighbors", type=int, help="number of neighbor sections to preload, default %(default)d", default=1)
    args = parser.parse_args()

    from sys import argv, exit
    appl = QApplication(argv)

    stack = args.stack_name
    NUM_NEIGHBORS_PRELOAD = args.num_neighbors
    m = BrainLabelingGUI(stack=stack)

    m.showMaximized()
    m.raise_()
    exit(appl.exec_())
