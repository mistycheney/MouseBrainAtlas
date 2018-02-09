#! /usr/bin/env python

import sys
import os

import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from qt_utilities import *

gray_color_table = [qRgb(i, i, i) for i in range(256)]

ACTIVE_SET_SIZE = 10

class SignalEmitter(QObject):
    update_active_set = pyqtSignal(object, object)
    image_loaded = pyqtSignal(int)
    def __init__(self):
        super(SignalEmitter, self).__init__()

class ReadImagesThread(QThread):
    # def __init__(self, stack, sections, img_version, downsample=1, prep_id=2):
    def __init__(self, stack, sections, img_version, resolution=None, prep_id=2):
        """
        This always loads images in raw resolution and then rescale them  to match `resolution`.

        Args:
            resolution (str): desired resolution to show in scene.
        """

        QThread.__init__(self)
        self.stack = stack
        self.sections = sections
        self.img_version = img_version
        # self.downsample = downsample
        self.resolution = resolution
        self.prep_id = prep_id

    # def __del__(self):
    #     self.wait()

    @pyqtSlot(object, object)
    def update_active_set(self, sections_to_load, sections_to_drop):

        sys.stderr.write("ReadImageThread: update_active_set, load %s, drop %s\n" % (sections_to_load, sections_to_drop))

        for sec_to_remove in sections_to_drop:
            sys.stderr.write("Remove section %d from cache.\n" % sec_to_remove)
            self.emit(SIGNAL('drop_image(int)'), sec_to_remove)

        for sec in sections_to_load:
            try:
                fp = DataManager.get_image_filepath_v2(stack=self.stack, section=sec, prep_id=self.prep_id, resol='lossless', version=self.img_version)
            except Exception as e:
                sys.stderr.write('Section %d is invalid: %s\n' % (sec, str(e)))
                continue
            if not os.path.exists(fp):
                sys.stderr.write('Image %s does not exist.\n' % fp)
                continue
            qimage = QImage(fp)

            desired_pixel_size_um = convert_resolution_string_to_voxel_size(resolution=self.resolution, stack=self.stack)
            raw_pixel_size_um = convert_resolution_string_to_voxel_size(resolution='lossless', stack=self.stack)
            scaling = raw_pixel_size_um / float(desired_pixel_size_um)

            if scaling != 1:
                # Downsample the image for CryoJane data, which is too large and exceeds QPixmap size limit.
                # sys.stderr.write('Raw qimage bytes = %d\n' % qimage.byteCount())
                raw_width, raw_height = (qimage.width(), qimage.height())
                new_width, new_height = (raw_width * scaling, raw_height * scaling)
                qimage = qimage.scaled(new_width, new_height)
                sys.stderr.write("Scale image by %.2f from size (w=%d,h=%d) to (w=%d,h=%d)\n" % (scaling, raw_width, raw_height, new_width, new_height))
                # sys.stderr.write('New qimage bytes = %d\n' % qimage.byteCount())

            self.emit(SIGNAL('image_loaded(QImage, int)'), qimage, sec)

    def run(self):
        print 'Worker thread:', self.currentThread()
        self.exec_()

class ImageDataFeeder_v2(object):

    def __init__(self, name, stack, prep_id=None, sections=None, use_data_manager=True, resolution=None, labeled_filenames=None, version=None):
        """
        Args:
            resolution (str):
        """

        self.name = name
        self.stack = stack
        self.prep_id = prep_id
        self.version = version

        if use_data_manager:
            assert sections is not None
            self.sections = sections
            self.min_section = min(self.sections)
            self.max_section = max(self.sections)
        else:
            self.sections = sections

        self.n = len(self.sections)

        # self.supported_downsample_factors = [1, 32]
        self.image_cache = {} # {downscale: {sec: qimage}}

        # if downscale is not None:
        #     self.set_downsample_factor(downscale)

        if resolution is not None:
            self.set_resolution(resolution)

        self.se = SignalEmitter()
        self.image_loader_thread = ReadImagesThread(stack=self.stack, sections=sections,
                img_version=self.version,
                # downsample=self.downsample,
                resolution=self.resolution,
                prep_id=prep_id)

        self.image_loader_thread.connect(self.se, SIGNAL("update_active_set(PyQt_PyObject, PyQt_PyObject)"), self.image_loader_thread.update_active_set)
        self.se.connect(self.image_loader_thread, SIGNAL("image_loaded(QImage, int)"), self.image_loaded)
        self.se.connect(self.image_loader_thread, SIGNAL("drop_image(int)"), self.drop_image)
        self.image_loader_thread.start()

        # if labeled_filenames is not None:
        #     self.set_images(labeled_filenames=labeled_filenames)
        # elif use_data_manager:
        #     self.load_images()

    @pyqtSlot(int)
    def drop_image(self, sec):
        downs = self.image_cache.keys()
        for down in downs:
            if sec in self.image_cache[down]:
                del self.image_cache[down][sec]

    @pyqtSlot(object, int)
    def image_loaded(self, qimage, sec):
        """
        Callback for when an image is loaded.

        Args:
            qimage (QImage): the image
            sec (int): section
        """

        self.set_image(sec=sec, qimage=qimage)
        self.se.image_loaded.emit(sec)

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.compute_dimension()

    def set_image(self, sec=None, i=None, qimage=None, numpy_image=None, fp=None, downsample=None, resolution=None):

        # if downsample is None:
        #     downsample = self.downsample

        if resolution is None:
            resolution = self.resolution

        # if downsample not in self.image_cache:
        #     self.image_cache[downsample] = {}

        if resolution not in self.image_cache:
            self.image_cache[resolution] = {}

        if sec is None:
            sec = self.sections[i]

        if qimage is None:
            if fp is not None:
                qimage = QImage(fp)
            elif numpy_image is not None:
                qimage = numpy_to_qimage(numpy_image)
            else:
                raise Exception('Either filepath or numpy_image must be provided.')

        # self.image_cache[downsample][sec] = qimage
        self.image_cache[resolution][sec] = qimage

        self.compute_dimension()

    def set_images(self, labels=None, filenames=None, labeled_filenames=None, resolution=None, load_with_cv2=False):

        if labeled_filenames is not None:
            assert isinstance(labeled_filenames, dict)
            labels = labeled_filenames.keys()
            filenames = labeled_filenames.values()

        for lbl, fn in zip(labels, filenames):
            if load_with_cv2: # For loading output tif images from elastix, directly QImage() causes "foo: Can not read scanlines from a tiled image."
                img = cv2.imread(fn)
                if img is None:
                    continue
                qimage = numpy_to_qimage(img)
            else:
                qimage = QImage(fn)

            # self.set_image(qimage=qimage, sec=lbl, downsample=downsample)
            self.set_image(qimage=qimage, sec=lbl, resolution=resolution)

    # def load_images(self, downsample=None, resolution=None, selected_sections=None):
    #     """
    #     If use_data_manager, use this function to load images.
    #     """
    #
    #     if downsample is None:
    #         downsample = self.downsample
    #
    #     if selected_sections is None:
    #         selected_sections = self.sections
    #
    #     self.image_cache[downsample] = {}
    #     for sec in selected_sections:
    #         # try:
    #
    #         if downsample == 32:
    #             qimage = QImage(DataManager.get_image_filepath_v2(stack=self.stack, section=sec, prep_id=self.prep_id, resol='thumbnail', version=self.version))
    #         elif downsample == 1:
    #             qimage = QImage(DataManager.get_image_filepath_v2(stack=self.stack, section=sec, prep_id=self.prep_id, resol='lossless', version=self.version))
    #         else:
    #             raise
    #
    #         self.image_cache[downsample][sec] = qimage
    #
    #     self.compute_dimension()

    def compute_dimension(self):

        # if hasattr(self, 'downsample') and self.downsample in self.image_cache and hasattr(self, 'orientation'):
        #     arbitrary_img = self.image_cache[self.downsample].values()[0]
        if hasattr(self, 'resolution') and self.resolution in self.image_cache and hasattr(self, 'orientation'):
            arbitrary_img = self.image_cache[self.resolution].values()[0]
            if self.orientation == 'sagittal':
                self.x_dim = arbitrary_img.width()
                self.y_dim = arbitrary_img.height()
            elif self.orientation == 'coronal':
                self.z_dim = arbitrary_img.width()
                self.y_dim = arbitrary_img.height()
            elif self.orientation == 'horizontal':
                self.x_dim = arbitrary_img.width()
                self.z_dim = arbitrary_img.height()

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.compute_dimension()
    #
    # def set_downsample_factor(self, downsample):
    #     self.downsample = downsample
    #     self.compute_dimension()

    def retrieve_i(self, i=None, sec=None, resolution=None):
        """
        Retrieve the i'th image in self.sections.
        """

        if resolution is None:
            resolution = self.resolution

        # if downsample is None:
        #     downsample = self.downsample

        # if downsample not in self.supported_downsample_factors:
        #     sys.stderr.write('Downsample factor %d is not supported.' % downsample)
        #     return

        if sec is None:
            sec = self.sections[i]

        if resolution not in self.image_cache:
            self.image_cache[resolution] = {}

        # if downsample not in self.image_cache:
        #     self.image_cache[downsample] = {}

        # if sec not in self.image_cache[downsample]:
        if sec not in self.image_cache[resolution]:
            # raise Exception('Image is not loaded: section %d' % sec)
            sys.stderr.write('Image data for section %d is not loaded. Loading now.. \n' % sec)

            # loaded_sections = set(self.image_cache[downsample].keys())
            loaded_sections = set(self.image_cache[resolution].keys())
            active_set = set(range(max(min(self.sections), sec-ACTIVE_SET_SIZE/2), min(max(self.sections), sec+ACTIVE_SET_SIZE/2+1)))
            # active_set = set(range(max(min(self.sections), sec-1), min(max(self.sections), sec+2)))
            print active_set, loaded_sections

            # if len(self.image_cache[downsample]) > 10:
            if len(self.image_cache[resolution]) > 10:
                # a = np.argsort([np.abs(s - sec) for s in loaded_sections])[:-5]
                # sections_to_remove = [loaded_sections[i] for i in a]
                sections_to_remove = loaded_sections - active_set
            else:
                sections_to_remove = []

            sections_to_load = active_set - loaded_sections

            self.se.update_active_set.emit(list(sections_to_load), list(sections_to_remove))

            for t in range(100):
                time.sleep(.1)
                # if sec in self.image_cache[downsample]:
                if sec in self.image_cache[resolution]:
                    break
        # return self.image_cache[downsample][sec]
        return self.image_cache[resolution][sec]

class VolumeResectionDataFeeder(object):

    def __init__(self, name, stack):

        self.name = name
        self.volume_cache = {}

        self.stack = stack

    def set_resolution(self, resolution):
        self.resolution = resolution

        if resolution in self.volume_cache:
            self.volume = self.volume_cache[resolution]
            self.y_dim, self.x_dim, self.z_dim = self.volume.shape
            if self.orientation == 'sagittal':
                self.n = self.z_dim
            elif self.orientation == 'coronal':
                self.n = self.x_dim
            elif self.orientation == 'horizontal':
                self.n = self.y_dim
        else:
            raise
            # try:
                # self.volume_cache[resolution] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':self.downsample})
            # except:
            #     sys.stderr.write('Cannot read volume file.\n')

    # def set_downsample_factor(self, downsample):
    #     # if downsample not in self.supported_downsample_factors:
    #     #     sys.stderr.write('Downsample factor %d is not supported.\n' % downsample)
    #     #     return
    #     # else:
    #     self.downsample = downsample
    #
    #     if self.downsample in self.volume_cache:
    #         self.volume = self.volume_cache[self.downsample]
    #         self.y_dim, self.x_dim, self.z_dim = self.volume.shape
    #
    #         if self.orientation == 'sagittal':
    #             self.n = self.z_dim
    #         elif self.orientation == 'coronal':
    #             self.n = self.x_dim
    #         elif self.orientation == 'horizontal':
    #             self.n = self.y_dim
    #     else:
    #         raise
            # try:
            #     self.volume_cache[self.downsample] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':self.downsample})
            # except:
            #     sys.stderr.write('Cannot read volume file.\n')

    def set_volume_cache(self, volume_cache):
        self.volume_cache = volume_cache

    # def add_volume(self, volume, downsample):
    #     if downsample not in self.volume_cache:
    #         self.volume_cache[downsample] = volume

    def add_volume(self, volume, resolution):
        if resolution not in self.volume_cache:
            self.volume_cache[resolution] = volume

    def set_orientation(self, orientation):
        self.orientation = orientation

    def retrieve_i(self, i, orientation=None, resolution=None):
        if orientation is None:
            orientation = self.orientation

        if resolution is None:
            resolution = self.resolution

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


    # def retrieve_i(self, i, orientation=None, downsample=None):
    #     if orientation is None:
    #         orientation = self.orientation
    #
    #     if downsample is None:
    #         downsample = self.downsample
    #
    #     if orientation == 'horizontal':
    #         y = i
    #         horizontal_data = self.volume[y, :, ::-1].T.flatten()
    #         horizontal_image = QImage(horizontal_data, self.x_dim, self.z_dim, self.x_dim, QImage.Format_Indexed8)
    #         horizontal_image.setColorTable(gray_color_table)
    #         return horizontal_image
    #
    #     elif orientation == 'coronal':
    #         x = i
    #         coronal_data = self.volume[:, x, ::-1].flatten()
    #         coronal_image = QImage(coronal_data, self.z_dim, self.y_dim, self.z_dim, QImage.Format_Indexed8)
    #         coronal_image.setColorTable(gray_color_table)
    #         return coronal_image
    #
    #     elif orientation == 'sagittal':
    #         z = i
    #         sagittal_data = self.volume[:, :, z].flatten()
    #         sagittal_image = QImage(sagittal_data, self.x_dim, self.y_dim, self.x_dim, QImage.Format_Indexed8)
    #         sagittal_image.setColorTable(gray_color_table)
    #         return sagittal_image
