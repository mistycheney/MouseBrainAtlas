#! /usr/bin/env python

import sys
import os
from datetime import datetime
import time
import json
from collections import defaultdict, OrderedDict

import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import LinearRing as ShapelyLineRing
from skimage.color import label2rgb
from pandas import DataFrame

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import DataManager
from metadata import *
from annotation_utilities import *
from gui_utilities import *
from registration_utilities import transform_volume_v4

from ui.ui_BrainLabelingGui_v15 import Ui_BrainLabelingGui

from widgets.custom_widgets import *
from widgets.SignalEmittingItems import *
from widgets.DrawableZoomableBrowsableGraphicsScene_ForLabeling_v2 import DrawableZoomableBrowsableGraphicsScene_ForLabeling

from DataFeeder import ImageDataFeeder_v2, VolumeResectionDataFeeder

######################################################################

MARKER_COLOR_CHAR = 'w'

#######################################################################

class ReadRGBComponentImagesThread(QThread):
    def __init__(self, stack, sections):
        QThread.__init__(self)
        self.stack = stack
        self.sections = sections

    def __del__(self):
        self.wait()

    def run(self):
        for sec in self.gscenes['main_sagittal'].active_section:
            if sec in self.gscenes['main_sagittal'].per_channel_pixmap_cached:
                continue
            try:
                fp = DataManager.get_image_filepath_v2(stack=self.stack, section=sec, prep_id=2, resol='lossless', version='contrastStretchedBlue')
            except Exception as e:
                sys.stderr.write('Section %d is invalid: %s\n' % (sec, str(e)))
                continue
            if not os.path.exists(fp):
                sys.stderr.write('Image %s does not exist.\n' % fp)
                continue
            qimage = QImage(fp)
            self.emit(SIGNAL('component_image_loaded(QImage, int)'), qimage, sec)


class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
# class BrainLabelingGUI(QMainWindow, Ui_RectificationGUI):

    # def __init__(self, parent=None, stack=None, first_sec=None, last_sec=None, downsample=None, img_version=None, prep_id=None):
    # def __init__(self, parent=None, stack=None, first_sec=None, last_sec=None, resolution=None, img_version=None, prep_id=None):
    def __init__(self, parent=None, stack=None, resolution=None, img_version=None, prep_id=None):
        """
        Initialization of BrainLabelingGUI.

        Args:
            resolution (str): desired resolution to show in scene.
        """
        # t0 = time.time()

        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.stack = stack
        # self.sagittal_downsample = downsample
        self.resolution = resolution

        if prep_id == 0:
            self.prep_id = None
        else:
            self.prep_id = prep_id

        self.setupUi(self)

        self.button_save.clicked.connect(self.save_contours)
        self.button_saveMarkers.clicked.connect(self.save_markers)
        self.button_saveStructures.clicked.connect(self.save_structures)
        self.button_saveHanddrawnStructures.clicked.connect(self.save_handdrawn_structures)
        # self.button_saveProbStructures.clicked.connect(self.save_structures)
        self.button_load.clicked.connect(self.load_contours)
        self.button_loadMarkers.clicked.connect(self.load_markers)
        self.button_loadStructures.clicked.connect(self.load_structures)
        self.button_loadHanddrawnStructures.clicked.connect(self.load_handdrawn_structures)
        # self.button_loadProbStructures.clicked.connect(self.load_structures)
        # self.button_loadWarpedAtlas.clicked.connect(self.load_warped_atlas_volume)
        self.button_loadWarpedAtlas.clicked.connect(self.load_warped_structure)
        # self.button_loadUnwarpedAtlas.clicked.connect(self.load_unwarped_atlas_volume)
        self.button_loadUnwarpedAtlas.clicked.connect(self.load_unwarped_structure)
        self.button_inferSide.clicked.connect(self.infer_side)
        self.button_clearSide.clicked.connect(self.clear_side)
        self.button_displayOptions.clicked.connect(self.select_display_options)
        self.button_displayStructures.clicked.connect(self.select_display_structures)
        self.button_navigateToStructure.clicked.connect(self.navigate_to_structure)
        self.button_reconstruct.clicked.connect(self.reconstruct_structure_callback)

        self.lineEdit_username.returnPressed.connect(self.username_changed)

        self.structure_volumes = {'handdrawn': defaultdict(dict), 'aligned_atlas': {}} # {set_name: {(name_unsided, side): structure_info_dict}}
        for name_s in all_known_structures_sided:
            # name_u, side = parse_label(name_s, singular_as_s=True)[:2]
            # self.structure_volumes['aligned_atlas'][(name_u, side)] = {'volume': None, 'origin': None, 'edits': []}
            self.structure_volumes['aligned_atlas'][name_s] = {'volume': None, 'origin': None, 'edits': []}
            self.structure_volumes['handdrawn'][name_s] = {'volume': None, 'origin': None, 'edits': []}

        # self.structure_volume_resolution_um = 16.
        self.structure_volume_resolution_um = 8.

        # loaded_intensity_volume_resol = 'down32'
        loaded_intensity_volume_resol = '10.0um'
        loaded_intensity_volume_resol_um = convert_resolution_string_to_voxel_size(resolution=loaded_intensity_volume_resol, stack=self.stack)
        # target_intensity_volume_resol = 'down32'
        target_intensity_volume_resol = '20.0um'
        # target_intensity_volume_resol = '20.0um'
        target_intensity_volume_resol_um = convert_resolution_string_to_voxel_size(resolution=target_intensity_volume_resol, stack=self.stack)

        self.volume_cache = {}
        try:
            intensity_volume_spec = dict(name=self.stack, resolution=loaded_intensity_volume_resol, prep_id='wholebrainWithMargin', vol_type='intensity')
            thumbnail_volume_dataResol, thumbnail_volume_origin_wrt_wholebrain_dataResol = DataManager.load_original_volume_v2(intensity_volume_spec, return_origin_instead_of_bbox=True)
            # thumbnail_volume_dataResol, thumbnail_volume_origin_wrt_wholebrain_dataResol = DataManager.load_intensity_volume_v3(self.stack, downscale=32, prep_id=4, return_origin_instead_of_bbox=True)
            self.volume_cache[target_intensity_volume_resol] = rescale_by_resampling(thumbnail_volume_dataResol, loaded_intensity_volume_resol_um / target_intensity_volume_resol_um)
            thumbnail_volume_origin_wrt_wholebrain_um = thumbnail_volume_origin_wrt_wholebrain_dataResol * loaded_intensity_volume_resol_um
            print 'Intensity volume', self.volume_cache[target_intensity_volume_resol].shape
            self.THUMBNAIL_VOLUME_LOADED = True
        except:
            sys.stderr.write('Intensity volume of resolution %s does not exist.\n' % loaded_intensity_volume_resol)
            self.THUMBNAIL_VOLUME_LOADED = False

        self.splitter.setSizes([500, 500, 500])
        self.splitter_2.setSizes([500, 500])

        self.tb_sagittal_gscene = DrawableZoomableBrowsableGraphicsScene_ForLabeling(id='tb_sagittal', gui=self, gview=self.tb_sagittal_gview)
        self.tb_sagittal_gview.setScene(self.tb_sagittal_gscene)

        self.tb_coronal_gscene = DrawableZoomableBrowsableGraphicsScene_ForLabeling(id='tb_coronal', gui=self, gview=self.tb_coronal_gview)
        self.tb_coronal_gview.setScene(self.tb_coronal_gscene)

        self.tb_horizontal_gscene = DrawableZoomableBrowsableGraphicsScene_ForLabeling(id='tb_horizontal', gui=self, gview=self.tb_horizontal_gview)
        self.tb_horizontal_gview.setScene(self.tb_horizontal_gscene)

        self.main_sagittal_gscene = DrawableZoomableBrowsableGraphicsScene_ForLabeling(id='main_sagittal', gui=self, gview=self.main_sagittal_gview)
        self.main_sagittal_gview.setScene(self.main_sagittal_gscene)

        if self.THUMBNAIL_VOLUME_LOADED:
            self.gscenes = { 'main_sagittal': self.main_sagittal_gscene,
                            'tb_coronal': self.tb_coronal_gscene,
                            'tb_horizontal': self.tb_horizontal_gscene,
                            'tb_sagittal': self.tb_sagittal_gscene}
        else:
            self.gscenes = { 'main_sagittal': self.main_sagittal_gscene}

        for gscene in self.gscenes.itervalues():
            gscene.drawings_updated.connect(self.drawings_updated)
            gscene.crossline_updated.connect(self.crossline_updated)
            gscene.active_image_updated.connect(self.active_image_updated)
            gscene.structure_volume_updated.connect(self.handle_structure_update)
            # gscene.prob_structure_volume_updated.connect(self.update_prob_structure_volume)
            gscene.global_transform_updated.connect(self.handle_global_transform_update)
            # gscene.set_structure_volumes(self.structure_volumes)
            gscene.set_structure_volumes(self.structure_volumes)
            # gscene.set_drawings(self.drawings)
            gscene.set_structure_volumes_resolution(um=self.structure_volume_resolution_um)

            # Set coordinate converter for every graphics scene.
            converter = CoordinatesConverter()
            gscene.set_coordinates_converter(converter)
            converter.register_new_resolution(resol_name='volume', resol_um=self.structure_volume_resolution_um)

        self.contextMenu_set = True

        self.recent_labels = []

        self.structure_names = load_structure_names(os.environ['REPO_DIR']+'/gui/structure_names.txt')
        self.new_labelnames = load_structure_names(os.environ['REPO_DIR']+'/gui/newStructureNames.txt')
        self.structure_names = OrderedDict(sorted(self.new_labelnames.items()) + sorted(self.structure_names.items()))

        self.installEventFilter(self)

        first_sec0, last_sec0 = DataManager.load_section_limits_v2(self.stack, prep_id=self.prep_id)
        self.sections = range(first_sec0, last_sec0 + 1)

        image_feeder = ImageDataFeeder_v2('image feeder', stack=self.stack, sections=self.sections,
        prep_id=self.prep_id,
        resolution=self.resolution,
        version=img_version,
        auto_load=True,
        use_thread=True)
        image_feeder.set_orientation('sagittal')

        self.gscenes['main_sagittal'].set_data_feeder(image_feeder)
        self.connect(self.gscenes['main_sagittal'], SIGNAL("image_loaded(int)"), self.image_loaded)
        self.gscenes['main_sagittal'].set_active_section(first_sec0)

        cropbox_origin_xy_wrt_wholebrain_tbResol = DataManager.load_cropbox_v2(stack=self.stack, prep_id=self.prep_id)[[0,2]]

        self.gscenes['main_sagittal'].converter.derive_three_view_frames(base_frame_name='data',
        origin_wrt_wholebrain_um=np.r_[cropbox_origin_xy_wrt_wholebrain_tbResol, 0] * convert_resolution_string_to_um(resolution='thumbnail', stack=self.stack))

        if self.THUMBNAIL_VOLUME_LOADED:

            volume_resection_feeder = VolumeResectionDataFeeder('volume resection feeder', self.stack)

            if hasattr(self, 'volume_cache') and self.volume_cache is not None:

                for plane in ['sagittal', 'coronal', 'horizontal']:

                    coronal_volume_resection_feeder = VolumeResectionDataFeeder(plane + ' thumbnail volume resection feeder', self.stack)
                    coronal_volume_resection_feeder.set_volume_cache(self.volume_cache)
                    coronal_volume_resection_feeder.set_orientation(plane)
                    coronal_volume_resection_feeder.set_resolution(self.volume_cache.keys()[0])
                    self.gscenes['tb_' + plane].set_data_feeder(coronal_volume_resection_feeder)
                    self.gscenes['tb_' + plane].set_active_i(0)
                    self.gscenes['tb_' + plane].converter.derive_three_view_frames(base_frame_name='data',
                    origin_wrt_wholebrain_um=thumbnail_volume_origin_wrt_wholebrain_um,
                    )

                # coronal_volume_resection_feeder = VolumeResectionDataFeeder('coronal resection feeder', self.stack)
                # coronal_volume_resection_feeder.set_volume_cache(self.volume_cache)
                # coronal_volume_resection_feeder.set_orientation('coronal')
                # coronal_volume_resection_feeder.set_resolution(self.volume_cache.keys()[0])
                # self.gscenes['tb_coronal'].set_data_feeder(coronal_volume_resection_feeder)
                # self.gscenes['tb_coronal'].set_active_i(50)
                # self.gscenes['tb_coronal'].converter.derive_three_view_frames(base_frame_name='data',
                # origin_wrt_wholebrain_um=thumbnail_volume_origin_wrt_wholebrain_um,
                # )
                #
                # horizontal_volume_resection_feeder = VolumeResectionDataFeeder('horizontal resection feeder', self.stack)
                # horizontal_volume_resection_feeder.set_volume_cache(self.volume_cache)
                # horizontal_volume_resection_feeder.set_orientation('horizontal')
                # horizontal_volume_resection_feeder.set_resolution(self.volume_cache.keys()[0])
                # self.gscenes['tb_horizontal'].set_data_feeder(horizontal_volume_resection_feeder)
                # self.gscenes['tb_horizontal'].set_active_i(150)
                # self.gscenes['tb_horizontal'].converter.derive_three_view_frames(base_frame_name='data',
                # origin_wrt_wholebrain_um=thumbnail_volume_origin_wrt_wholebrain_um,
                # )
                #
                # sagittal_volume_resection_feeder = VolumeResectionDataFeeder('sagittal resection feeder', self.stack)
                # sagittal_volume_resection_feeder.set_volume_cache(self.volume_cache)
                # sagittal_volume_resection_feeder.set_orientation('sagittal')
                # sagittal_volume_resection_feeder.set_resolution(self.volume_cache.keys()[0])
                # self.gscenes['tb_sagittal'].set_data_feeder(sagittal_volume_resection_feeder)
                # self.gscenes['tb_sagittal'].set_active_i(150)
                # self.gscenes['tb_sagittal'].converter.derive_three_view_frames(base_frame_name='data',
                # origin_wrt_wholebrain_um=thumbnail_volume_origin_wrt_wholebrain_um,
                # )

        # for gid, gs in self.gscenes.iteritems():
        #     print gid
        #     for frame_name, frame in gs.converter.frames.iteritems():
        #         print frame_name, frame
        #     print

        for gscene in self.gscenes.itervalues():
            # If image resolution > 10 um
            if convert_resolution_string_to_um(resolution=gscene.data_feeder.resolution, stack=self.stack) > 10.:
                # thumbnail graphics scenes
                linewidth = 1
                vertex_radius = .2
            else:
                # raw graphics scenes
                linewidth = 10
                vertex_radius = 15

            gscene.set_default_line_width(linewidth)
            gscene.set_default_line_color('b')
            gscene.set_default_vertex_radius(vertex_radius)
            gscene.set_default_vertex_color('r')

        # Syncing main scene with crossline localization requires constantly loading
        # raw images which can be slow.
        self.DISABLE_UPDATE_MAIN_SCENE = False

    def reconstruct_structure_callback(self):
        pass
        # selected_structures, structures_to_add, structures_to_remove = self.select_structures(set_name='handdrawn')
        # for name_s in structures_to_add:
        #     self.handle_structure_update(set_name='handdrawn', name_s=name_s, use_confirmed_only=True, recompute_from_contours=True)

    def navigate_to_structure(self):

        # Can also use get_landmark_range_limits_v2()

        loaded_structures = defaultdict(set)
        for index, elements in self.gscenes['main_sagittal'].drawings.iteritems():
            assert index is not None
            section = self.gscenes['main_sagittal'].data_feeder.sections[index]
            print section, [elem.properties['label'] for elem in elements]
            for elem in elements:
                name_s = compose_label(elem.properties['label'], elem.properties['side'])
                loaded_structures[name_s].add(section)

        print loaded_structures

        dial = ListSelection("Select structures to load", "List of structures", sorted(loaded_structures.keys()), [], self)
        if dial.exec_() == QDialog.Accepted:
            selected_structures = map(str, dial.itemsSelected())
        else:
            return

        assert len(selected_structures) == 1, 'Only one structure can be selected'
        selected_structure = selected_structures[0]

        section_to_load = sorted(list(loaded_structures[selected_structure]))[0]
        print 'section_to_load =', section_to_load
        self.gscenes['main_sagittal'].set_active_section(section_to_load)

    @pyqtSlot(object)
    def handle_global_transform_update(self, tf):
        """
        Handle update of the global transform.

        Args:
            tf (12-vector)
        """
        print "Global transform updated."
        print tf.reshape((3,4))
        self.global_transform_from_wholebrain_to_wholebrain_volResol = tf

    @pyqtSlot(int)
    def image_loaded(self, sec):
        gscene_id = self.sender().id
        gscene = self.gscenes[gscene_id]

        if gscene.active_section == sec:
            gscene.update_image()

        self.statusBar().showMessage('Image %d loaded.\n' % sec)
        print 'Image', sec, 'received.'

    @pyqtSlot(object, int)
    def component_image_loaded(self, qimage_blue, sec):
        """
        Callback for when R/G/B images are loaded.

        Args:
            qimage (QImage): the image
            sec (int): section
        """

        self.gscenes['main_sagittal'].per_channel_pixmap_cached[sec] = qimage_blue
        self.statusBar().showMessage('R/G/B images %d loaded.\n' % sec)
        print 'R/G/B images', sec, 'received.'

    @pyqtSlot()
    def username_changed(self):
        self.username = str(self.sender().text())
        print 'username changed to', self.username

    def get_username(self):
        if not hasattr(self, 'username') or self.username is None:
            username, okay = QInputDialog.getText(self, "Username", "Please enter your username:", QLineEdit.Normal, 'anon')
            if not okay: return
            self.username = str(username)
            self.lineEdit_username.setText(self.username)

        return self.username

    def structure_tree_changed(self, item, column):

        tree_widget = self.sender()
        complete_name = str(item.text(column))
        abbr = re.findall('^.*?(\((.*)\))?$', complete_name)[0][1]

        check_state = item.checkState(column)
        if check_state == Qt.Unchecked:

            for gscene in self.gscenes.values():
                for section_index, polygons in gscene.drawings.iteritems():
                    for polygon in polygons:
                        if polygon.properties['label'] == abbr:
                            polygon.setVisible(False)

        elif check_state == Qt.PartiallyChecked:
            pass
        elif check_state == Qt.Checked:
            for gscene in self.gscenes.values():
                for section_index, polygons in gscene.drawings.iteritems():
                    for polygon in polygons:
                        if polygon.properties['label'] == abbr:
                            polygon.setVisible(True)
        else:
            raise Exception('Unknown check state.')

        # selected_items = tree_widget.selectedItems()
        # print [str(it.text(0)) for it in selected_items]


    @pyqtSlot()
    def select_display_structures(self):
        loaded_structure_abbrs = set([convert_to_unsided_label(name_s) for name_s in self.gscenes['main_sagittal'].get_label_section_lookup().keys()])

        structure_tree_dict = json.load(open('structure_tree.json'))
        # structure_tree_dict = {name: d for name, d in structure_tree_dict_all.iteritems() if d['abbr'] in loaded_structure_names}
        # structure_tree_names = {'brainstem': {}}

        def structure_entry_to_str(node):
            if 'abbr' in node and len(node['abbr']) > 0:
                key = node['fullname'] + ' (' + node['abbr'] + ')'
            else:
                key = node['fullname']
            return key

        def get_children_names(name):
            node = structure_tree_dict[name]
            key = structure_entry_to_str(node)
            return (key, dict([get_children_names(child_name) for child_name in node['children']]))

        structure_name_tree = dict([get_children_names('brainstem')])

        # extract_names(structure_tree_names['brainstem'], structure_tree_dict['brainstem'], structure_tree_dict)
        # structure_tree_names = {'midbrain': ['IC', 'SC'], 'hindbrain': {'pons': ['7N', '5N'], 'medulla': ['7n', 'SCP']}}

        display_structures_widget = QDialog(self)

        tree_widget = QTreeWidget(display_structures_widget)
        tree_widget.setHeaderLabels(['Structures'])
        fill_tree_widget(tree_widget, structure_name_tree, loaded_structure_abbrs)
        # tree_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tree_widget.setMinimumHeight(1000)
        tree_widget.setMinimumWidth(500)

        # http://stackoverflow.com/questions/27521391/signal-a-qtreewidgetitem-toggled-checkbox
        tree_widget.itemChanged.connect(self.structure_tree_changed)

        dialog_layout = QVBoxLayout(display_structures_widget)
        dialog_layout.addWidget(tree_widget)
        display_structures_widget.setLayout(dialog_layout)
        # display_structures_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        display_structures_widget.setWindowTitle("Select structures to show")
        # display_structures_widget.exec_()
        display_structures_widget.show()

    @pyqtSlot()
    def select_display_options(self):

        if not hasattr(self, 'show_polygons'):
            self.show_polygons = True
            self.show_vertices = True
            self.show_labels = True
            self.hide_interpolated = False

        display_option_widget = QDialog(self)

        layout = QVBoxLayout()

        checkbox_showPolygons = QCheckBox("Polygon")
        checkbox_showPolygons.setChecked(self.show_polygons)
        checkbox_showPolygons.stateChanged.connect(self.checkbox_showPolygons_callback)
        layout.addWidget(checkbox_showPolygons)

        checkbox_showVertices = QCheckBox("Vertices")
        checkbox_showVertices.setChecked(self.show_vertices)
        checkbox_showVertices.stateChanged.connect(self.checkbox_showVertices_callback)
        layout.addWidget(checkbox_showVertices)

        checkbox_showLabels = QCheckBox("Labels")
        checkbox_showLabels.setChecked(self.show_labels)
        checkbox_showLabels.stateChanged.connect(self.checkbox_showLabels_callback)
        layout.addWidget(checkbox_showLabels)

        checkbox_hideInterpolated = QCheckBox("Hide interpolated")
        checkbox_hideInterpolated.setChecked(self.hide_interpolated)
        checkbox_hideInterpolated.stateChanged.connect(self.checkbox_hideInterpolated_callback)
        layout.addWidget(checkbox_hideInterpolated)

        display_option_widget.setLayout(layout)
        display_option_widget.setWindowTitle("Select display options")
        display_option_widget.exec_()

    @pyqtSlot(int)
    def checkbox_showLabels_callback(self, checked):
        self.show_labels = checked
        for gscene in self.gscenes.itervalues():
            for section_index, polygons in gscene.drawings.iteritems():
                for polygon in polygons:
                    polygon.properties['label_textItem'].setVisible(checked)

    @pyqtSlot(int)
    def checkbox_showVertices_callback(self, checked):
        self.show_vertices = checked
        for gscene in self.gscenes.itervalues():
            for section_index, polygons in gscene.drawings.iteritems():
                for polygon in polygons:
                    for v in polygon.vertex_circles:
                        v.setVisible(checked)

    @pyqtSlot(int)
    def checkbox_showPolygons_callback(self, checked):
        self.show_polygons = checked
        for gscene in self.gscenes.itervalues():
            for section_index, polygons in gscene.drawings.iteritems():
                for polygon in polygons:
                    polygon.setVisible(checked)

    @pyqtSlot(int)
    def checkbox_hideInterpolated_callback(self, checked):
        self.hide_interpolated = checked
        for gscene in self.gscenes.itervalues():
            for section_index, polygons in gscene.drawings.iteritems():
                for polygon in polygons:
                    if polygon.type != 'confirmed':
                        polygon.setVisible(not bool(checked))

    @pyqtSlot()
    def infer_side(self):
        self.gscenes['main_sagittal'].infer_side()

        # try:
        #     self.gscenes['tb_sagittal'].infer_side()
        # except:
        #     pass
        #
        # try:
        #     self.gscenes['tb_coronal'].infer_side()
        # except:
        #     pass
        #
        # try:
        #     self.gscenes['tb_horizontal'].infer_side()
        # except:
        #     pass


    @pyqtSlot()
    def clear_side(self):
        self.gscenes['main_sagittal'].clear_side()
        #
        # try:
        #     self.gscenes['tb_sagittal'].clear_side()
        # except:
        #     pass
        #
        # try:
        #     self.gscenes['tb_coronal'].clear_side()
        # except:
        #     pass
        #
        # try:
        #     self.gscenes['tb_horizontal'].clear_side()
        # except:
        #     pass


    @pyqtSlot()
    def save_markers(self):
        """
        Save markers.
        """

        timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

        sagittal_markers_curr_session = self.gscenes['main_sagittal'].convert_drawings_to_entries(timestamp=timestamp, username=self.username, classes=['neuron'])
        sagittal_markers_original = convert_annotation_v3_aligned_cropped_to_original(DataFrame(sagittal_markers_curr_session).T, stack=self.stack,
        prep_id=self.prep_id)
        if self.prep_id == 3: # thalamus only
            sagittal_markers_fp = DataManager.get_annotation_thalamus_filepath(stack=self.stack, by_human=True, suffix='neurons', timestamp=timestamp)
        else:
            sagittal_markers_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='neurons', timestamp=timestamp)
        save_hdf_v2(sagittal_markers_original, sagittal_markers_fp)
        upload_to_s3(sagittal_markers_fp)
        print 'Sagittal markers saved to %s.' % sagittal_markers_fp
        self.statusBar().showMessage('Sagittal markers saved to %s.' % sagittal_markers_fp)

    # @pyqtSlot()
    # def save_structures(self):
    #     """
    #     Save 3D structure volumes.
    #     """
    #
    #     # timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
    #     import uuid
    #
    #     entries = {}
    #     for (name, side), v in self.structure_volumes.iteritems():
    #         entry = {}
    #         entry['volume'] = bp.pack_ndarray_str(v['volume'])
    #         entry['origin'] = v['origin']
    #         entry['name'] = name
    #         entry['side'] = side
    #         if 'edits' not in v or v['edits'] is None or len(v['edits']) == 0:
    #             entry['edits'] = []
    #         else:
    #             entry['edits'] = v['edits']
    #
    #         if hasattr(v, 'structure_id') and v.properties['structure_id'] is not None:
    #             structure_id = v.properties['structure_id']
    #         else:
    #             structure_id = str(uuid.uuid4().fields[-1])
    #
    #         entries[structure_id] = entry
    #
    #     structure_df = DataFrame(entries).T
    #     if self.prep_id == 3: # thalamus
    #         structure_df_fp = DataManager.get_annotation_thalamus_filepath(stack=self.stack, by_human=True, suffix='structures', timestamp='now')
    #     else:
    #         structure_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='structures', timestamp='now')
    #     save_hdf_v2(structure_df, structure_df_fp)
    #     upload_to_s3(structure_df_fp)
    #     print '3D structures saved to %s.' % structure_df_fp

    @pyqtSlot()
    def save_structures(self):
        """
        Save structure volumes in a file.
        """

        # timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        import uuid

        entries = {}
        for name_s, structure_info in self.structure_volumes['aligned_atlas'].iteritems():

            name_u, side = parse_label(name_s, singular_as_s=True)[:2]

            entry = {}
            vol = structure_info['volume']
            sys.stderr.write("Saved structure %s\n" % name_s)

            if vol is None:
                entry['volume'] = None
                entry['origin'] = None
            else:
                entry['volume'] = bp.pack_ndarray_str(vol)
                entry['origin'] = structure_info['origin']
            entry['name'] = name_u
            entry['side'] = side
            entry['resolution'] = '%.1fum' % self.structure_volume_resolution_um
            if 'edits' not in structure_info or structure_info['edits'] is None or len(structure_info['edits']) == 0:
                entry['edits'] = []
            else:
                entry['edits'] = structure_info['edits']

            if hasattr(structure_info, 'structure_id') and structure_info.properties['structure_id'] is not None:
                structure_id = structure_info.properties['structure_id']
            else:
                structure_id = str(uuid.uuid4().fields[-1])

            entries[structure_id] = entry

        structure_df = DataFrame(entries).T
        if self.prep_id == 3: # thalamus
            structure_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='structures', timestamp='now', annotation_rootdir=ANNOTATION_THALAMUS_ROOTDIR)
        else:
            structure_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='structures', timestamp='now')
        save_hdf_v2(structure_df, structure_df_fp)
        # upload_to_s3(structure_df_fp)
        print 'Probabilistic structures saved to %s.' % structure_df_fp

    @pyqtSlot()
    def save_handdrawn_structures(self):
        """
        Save handdrawn structure volumes in a file.
        """

        # timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        import uuid

        entries = {}
        for name_s, structure_info in self.structure_volumes['handdrawn'].iteritems():

            name_u, side = parse_label(name_s, singular_as_s=True)[:2]

            entry = {}
            vol = structure_info['volume']
            sys.stderr.write("Saved structure %s\n" % name_s)

            if vol is None:
                entry['volume'] = None
                entry['origin'] = None
            else:
                entry['volume'] = bp.pack_ndarray_str(vol)
                entry['origin'] = structure_info['origin']
            entry['name'] = name_u
            entry['side'] = side
            entry['resolution'] = '%.1fum' % self.structure_volume_resolution_um
            if 'edits' not in structure_info or structure_info['edits'] is None or len(structure_info['edits']) == 0:
                entry['edits'] = []
            else:
                entry['edits'] = structure_info['edits']

            if hasattr(structure_info, 'structure_id') and structure_info.properties['structure_id'] is not None:
                structure_id = structure_info.properties['structure_id']
            else:
                structure_id = str(uuid.uuid4().fields[-1])

            entries[structure_id] = entry

        structure_df = DataFrame(entries).T
        if self.prep_id == 3: # thalamus
            structure_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='structuresHanddrawn', timestamp='now', annotation_rootdir=ANNOTATION_THALAMUS_ROOTDIR)
        else:
            structure_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='structuresHanddrawn', timestamp='now')
        save_hdf_v2(structure_df, structure_df_fp)
        # upload_to_s3(structure_df_fp)
        print 'Handdrawn structures saved to %s.' % structure_df_fp


    @pyqtSlot()
    def save_contours(self):
        """
        Save 2-D boundaries (main sagittal scene).
        """

        timestamp = get_timestamp_now()

        sagittal_contour_entries_curr_session = self.gscenes['main_sagittal'].convert_drawings_to_entries(timestamp=timestamp, username=self.username)

        print '\nSaved the following contours:'
        for cid, contour in sagittal_contour_entries_curr_session.iteritems():
            print contour['name'], contour['side'], 'section =', contour['section'], 'number of vertices =', len(contour['vertices'])
        print '\n'

        assert self.prep_id == 2 or self.prep_id == 3

        # if self.prep_id == 1:
        #     in_wrt = 'alignedPadded'
        # elif self.prep_id == 0 or self.prep_id is None:
        #     in_wrt = 'original'
        # elif self.prep_id == 2:
        #     in_wrt = 'alignedBrainstemCrop'
        # else:
        #     raise

        # sagittal_contours_df_1um = convert_annotations(contour_df=DataFrame(sagittal_contour_entries_curr_session).T,
        # stack=stack, in_wrt=in_wrt, in_resol=self.resolution, out_wrt=out_wrt, out_resol='1um')

        sagittal_contours_df_original = convert_annotation_v3_aligned_cropped_to_original_v2(DataFrame(sagittal_contour_entries_curr_session).T,
        stack=self.stack, resolution=self.gscenes['main_sagittal'].data_feeder.resolution, prep_id=self.prep_id)

        if self.prep_id == 3: # thalamus
            sagittal_contours_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='contours', timestamp=timestamp, annotation_rootdir=ANNOTATION_THALAMUS_ROOTDIR)
        else:
            sagittal_contours_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='contours', timestamp=timestamp)

        save_hdf_v2(sagittal_contours_df_original, sagittal_contours_df_fp)
        # upload_to_s3(sagittal_contours_df_fp)
        self.statusBar().showMessage('Sagittal boundaries saved to %s.' % sagittal_contours_df_fp)
        print 'Sagittal boundaries saved to %s.' % sagittal_contours_df_fp

        # Save coronal
        # coronal_contour_entries_curr_session = self.gscenes['tb_coronal'].convert_drawings_to_entries(timestamp=timestamp, username=self.username)
        # # print coronal_contour_entries_curr_session
        # if len(coronal_contour_entries_curr_session) > 0:
        #     # coronal_contours_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=False, stack_m=stack_m,
        #     #                                                        classifier_setting_m=classifier_setting_m,
        #     #                                                       classifier_setting_f=classifier_setting_f,
        #     #                                                       warp_setting=warp_setting, suffix='contours_coronal')
        #     coronal_contours_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='contours_coronal', timestamp=timestamp)
        #     save_hdf_v2(coronal_contour_entries_curr_session, coronal_contours_df_fp)
        #     upload_to_s3(coronal_contours_df_fp)
        #     self.statusBar().showMessage('Coronal boundaries saved to %s.' % coronal_contours_df_fp)
        #     print 'Coronal boundaries saved to %s.' % coronal_contours_df_fp

        # Save horizontal
        # horizontal_contour_entries_curr_session = self.gscenes['tb_horizontal'].convert_drawings_to_entries(timestamp=timestamp, username=self.username)
        # if len(horizontal_contour_entries_curr_session) > 0:
        #     # horizontal_contours_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=False, stack_m=stack_m,
        #     #                                                        classifier_setting_m=classifier_setting_m,
        #     #                                                       classifier_setting_f=classifier_setting_f,
        #     #                                                       warp_setting=warp_setting, suffix='contours_horizontal')
        #     horizontal_contours_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=True, suffix='contours_horizontal', timestamp=timestamp)
        #     save_hdf_v2(horizontal_contour_entries_curr_session, horizontal_contours_df_fp)
        #     upload_to_s3(horizontal_contours_df_fp)
        #     self.statusBar().showMessage('Horizontal boundaries saved to %s.' % horizontal_contours_df_fp)
        #     print 'Horizontal boundaries saved to %s.' % horizontal_contours_df_fp

    @pyqtSlot()
    def load_markers(self):
        """
        """
        if self.prep_id == 3:
            markers_df_fp = str(QFileDialog.getOpenFileName(self, "Choose marker annotation file", os.path.join(ANNOTATION_THALAMUS_ROOTDIR, self.stack)))
        else:
            markers_df_fp = str(QFileDialog.getOpenFileName(self, "Choose marker annotation file", os.path.join(ANNOTATION_ROOTDIR, self.stack)))

        if markers_df_fp == '':
            return
        # download_from_s3(markers_df_fp)
        markers_df = load_hdf_v2(markers_df_fp)

        markers_df_cropped = convert_annotation_v3_original_to_aligned_cropped(markers_df, stack=self.stack, prep_id=self.prep_id)
        raise
        markers_df_cropped_sagittal = markers_df_cropped[(markers_df_cropped['orientation'] == 'main_sagittal') & (markers_df_cropped['downsample'] == self.gscenes['main_sagittal'].data_feeder.downsample)]

        # for i, marker_entry in markers_df_cropped_sagittal.iterrows():
        #     if 'label' not in marker_entry:
        #         print marker_entry

        self.gscenes['main_sagittal'].load_drawings(markers_df_cropped_sagittal, append=False, vertex_color=MARKER_COLOR_CHAR)


    def get_edit_transform(self, name_s):
        structure_info = self.structure_volumes['aligned_atlas'][name_s]
        print name_s, [ed['type'] for ed in structure_info['edits']]

        global_tf_from_wholebrain_to_wholebrain_volResol = np.eye(4)
        local_tf_from_wholebrain_to_wholebrain_volResol = np.eye(4)
        composed_tf_from_wholebrain_to_wholebrain_volResol = np.eye(4)
        for edit in structure_info['edits']:
            edit_tf_from_wholebrain_to_wholebrain_volRes = consolidate(edit['transform'])
            if edit['type'].startswith('global'): #  global_rotate3d or global_shift3d
                global_tf_from_wholebrain_to_wholebrain_volResol = \
                np.dot(edit_tf_from_wholebrain_to_wholebrain_volRes, global_tf_from_wholebrain_to_wholebrain_volResol)
            elif edit['type'].startswith('prob'):
                local_tf_from_wholebrain_to_wholebrain_volResol = \
                np.dot(edit_tf_from_wholebrain_to_wholebrain_volRes, local_tf_from_wholebrain_to_wholebrain_volResol)

            composed_tf_from_wholebrain_to_wholebrain_volResol = \
            np.dot(edit_tf_from_wholebrain_to_wholebrain_volRes, composed_tf_from_wholebrain_to_wholebrain_volResol)

            # print global_tf_from_wholebrain_to_wholebrain_volResol.reshape((4,4))[:3]
            # print local_tf_from_wholebrain_to_wholebrain_volResol.reshape((4,4))[:3]
            # print composed_tf_from_wholebrain_to_wholebrain_volResol.reshape((4,4))[:3]
        return composed_tf_from_wholebrain_to_wholebrain_volResol.reshape((4,4))[:3]

    # def get_global_transform(self, name_s):
    #     global_tf_from_wholebrain_to_wholebrain_volResol = np.eye(4)
    #     struct_info = self.structure_volumes['aligned_atlas'][name_s]
    #     for edit in struct_info['edits']:
    #         if edit['type'].startswith('global'): #  global_rotate3d or global_shift3d
    #             edit_tf_from_wholebrain_to_wholebrain_volRes = consolidate(edit['transform'])
    #             global_tf_from_wholebrain_to_wholebrain_volResol = \
    #             np.dot(edit_tf_from_wholebrain_to_wholebrain_volRes, global_tf_from_wholebrain_to_wholebrain_volResol)
    #             print name_s, edit['type'], struct_info['origin']
    #             print global_tf_from_wholebrain_to_wholebrain_volResol.reshape((4,4))
    #     return global_tf_from_wholebrain_to_wholebrain_volResol[:3]

    def load_atlas_volume(self, warped=True, structures=all_known_structures_sided):

        atlas_spec = dict(name='atlasV6',
                       vol_type='score',
                       detector_id=None,
                       prep_id=None,
                       structure=None,
                       resolution='10.0um')

        if not warped:
            atlas_volumes = DataManager.load_original_volume_all_known_structures_v3(stack_spec=atlas_spec,
            in_bbox_wrt='canonicalAtlasSpace',
            return_label_mappings=False,
            name_or_index_as_key='name',
            structures=structures,
            common_shape=False,
            return_origin_instead_of_bbox=True
            )

            for name_s, (v_10um, origin_wrt_fixedWholebrain_10um) in atlas_volumes.iteritems():
                # from skimage.transform import rescale
                #### FIX this !!! CHAT brain can be problematic !!!
                # atlas_volumes = {name_s: (rescale(v_10um, self.sagittal_downsample), b_wrt_fixedWholebrain_10um[[0,2,4]] * self.sagittal_downsample)
                #                         for name_s, (v_10um, b_wrt_fixedWholebrain_10um) in atlas_volumes.iteritems()}
                # atlas_bbox_wrt_wholebrain_volResol = atlas_bbox_wrt_wholebrain_volResol * self.sagittal_downsample
                # name_u, side = parse_label(name_s, singular_as_s=True)[:2]

                volume_volResol = rescale_by_resampling(v_10um, 10./self.structure_volume_resolution_um)
                origin_wrt_fixedWholebrain_volResol = origin_wrt_fixedWholebrain_10um * 10./self.structure_volume_resolution_um

                edit_transform_from_wholebrain_to_wholebrain_volResol = self.get_edit_transform(name_s)

                volume_volResol, origin_wrt_fixedWholebrain_volResol = \
                transform_volume_v4(volume=(volume_volResol, origin_wrt_fixedWholebrain_volResol),
                transform=edit_transform_from_wholebrain_to_wholebrain_volResol,
                return_origin_instead_of_bbox=True)

                self.structure_volumes['aligned_atlas'][name_s]['volume'] = volume_volResol
                self.structure_volumes['aligned_atlas'][name_s]['origin'] = origin_wrt_fixedWholebrain_volResol
                print 'Load', name_s, "volume", volume_volResol.shape, "; origin (wrt fixedwholebrain) =", origin_wrt_fixedWholebrain_volResol

                self.handle_structure_update(set_name='aligned_atlas', name_s=name_s, use_confirmed_only=False, recompute_from_contours=False)

        else:

            if stack in ['CHATM2', 'CHATM3']:
                detector_id_f = 799

                atlas_volumes = {}
                for structure in structures:

                    stack_m_spec = dict(name='atlasV6',
                                       vol_type='score',
                                       structure=structure,
                                        resolution='10.0um'
                                       )

                    stack_f_spec = dict(name=self.stack,
                                       vol_type='score',
                                       detector_id=799,
                                       structure=convert_to_original_name(structure),
                                        resolution='10.0um'
                                       )

                    local_alignment_spec = dict(stack_m=stack_m_spec,
                                          stack_f=stack_f_spec,
                                          warp_setting=27)

                    atlas_volumes[structure] = DataManager.load_transformed_volume_v2(alignment_spec=local_alignment_spec, return_origin_instead_of_bbox=True)

                for name_s, (v, o) in atlas_volumes.iteritems():

                    print o

                    self.structure_volumes['aligned_atlas'][name_s]['volume'] = rescale_by_resampling(v, convert_resolution_string_to_voxel_size(resolution='10.0um', stack=self.stack) / self.structure_volume_resolution_um)
                    self.structure_volumes['aligned_atlas'][name_s]['origin'] = o * convert_resolution_string_to_voxel_size(resolution='10.0um', stack=self.stack) / self.structure_volume_resolution_um
                    print 'Load', name_s, self.structure_volumes['aligned_atlas'][name_s]['origin']

                    self.handle_structure_update(set_name='aligned_atlas', name_s=name_s, use_confirmed_only=False, recompute_from_contours=False)

                    # # Update drawings on all gscenes based on `structure_volumes` that was just assigned.
                    # for gscene in self.gscenes.values():
                    #     gscene.update_drawings_from_structure_volume(name_s=name_s, levels=[0.5], set_name='aligned_atlas')

            else:

                if stack in ['MD661', 'MD662']:
                    detector_id_f = 1
                elif stack in ['MD653', 'MD652', 'MD642', 'MD657', 'MD658']:
                    detector_id_f = 13
                else:
                    detector_id_f = 15

                atlas_volumes = DataManager.load_transformed_volume_all_known_structures_v3(stack_m='atlasV5', stack_f=self.stack,
                warp_setting=17, prep_id_f=2, detector_id_f=detector_id_f,
                return_label_mappings=False,
                name_or_index_as_key='name',
                structures=structures
                ) # down32 resolution

                # atlas_volumes = DataManager.load_transformed_volume_all_known_structures(stack_m='atlasV5', stack_f=self.stack,
                # warp_setting=20, prep_id_f=2, detector_id_f=detector_id_f,
                # return_label_mappings=False,
                # name_or_index_as_key='name',
                # structures=structures
                # ) # down32 resolution

                atlas_origin_wrt_wholebrain_tbResol = DataManager.load_cropbox(stack=self.stack, convert_section_to_z=True, prep_id=self.prep_id, return_origin_instead_of_bbox=True)

                for name_s, v_down32 in atlas_volumes.iteritems():

                    self.structure_volumes['aligned_atlas'][name_s]['volume'] = rescale_by_resampling(v_down32, convert_resolution_string_to_voxel_size(resolution='down32', stack=self.stack) / self.structure_volume_resolution_um)
                    self.structure_volumes['aligned_atlas'][name_s]['origin'] = atlas_origin_wrt_wholebrain_tbResol * convert_resolution_string_to_voxel_size(resolution='down32', stack=self.stack) / self.structure_volume_resolution_um
                    print 'Load', name_s, self.structure_volumes['aligned_atlas'][name_s]['origin']

                    # Update drawings on all gscenes based on `structure_volumes` that was just assigned.
                    for gscene in self.gscenes.values():
                        gscene.update_drawings_from_structure_volume(name_s=name_s, levels=[0.5], set_name='aligned_atlas')

    @pyqtSlot()
    def select_structures(self, set_name='aligned_atlas'):
        """
        Select structures from a list.
        In this list, structures that are already created are pre-checked.

        Returns:
            (selected_structures, structures_to_add, structures_to_remove)
        """

        possible_structures_to_load = sorted(all_known_structures_sided)

   #      selected_structure, ok = QInputDialog.getItem(self, "Select one structure",
   # "list of structures", possible_structures_to_load, 0, False)
        # if ok and selected_structure:
        #     self.load_atlas_volume(warped=False, structures=[str(selected_structure)])

        structures_loaded = set([name_s
        for name_s, struct_info in self.structure_volumes[set_name].iteritems()
        if struct_info['volume'] is not None])

        dial = ListSelection("Select structures to load", "List of structures", possible_structures_to_load, structures_loaded, self)
        if dial.exec_() == QDialog.Accepted:
            selected_structures = set(map(str, dial.itemsSelected()))
        else:
            return

        print selected_structures
        new_structures_to_load = selected_structures - structures_loaded
        print 'structures_to_load', new_structures_to_load

        structures_to_remove = structures_loaded - selected_structures
        print 'structures_to_remove', structures_to_remove

        return selected_structures, new_structures_to_load, structures_to_remove

    @pyqtSlot()
    def load_unwarped_structure(self):
        """
        Load particular structures from warped atlas.
        """

        selected_structures, new_structures_to_load, structures_to_remove = self.select_structures()
        self.load_atlas_volume(warped=False, structures=new_structures_to_load)

    @pyqtSlot()
    def load_warped_structure(self):
        """
        Load particular structures from warped atlas.
        """
        selected_structures, new_structures_to_load, structures_to_remove = self.select_structures()
        self.load_atlas_volume(warped=True, structures=new_structures_to_load)

    @pyqtSlot()
    def load_structures(self, set_name='aligned_atlas'):
        """
        Load 3-D structure annotations from file.

        Args:
            set_name (str): default to 'aligned_atlas'
        """

        structures_df_fp = str(QFileDialog.getOpenFileName(self, "Choose the structure annotation file",
        os.path.join((ANNOTATION_THALAMUS_ROOTDIR if self.prep_id == 3 else ANNOTATION_ROOTDIR), self.stack)))

        if structures_df_fp == '':
            return

        structure_df = load_hdf_v2(structures_df_fp)

        selected_structures, new_structures_to_load, structures_to_remove = self.select_structures()

        for sid, struct_info in structure_df.iterrows():
            name_s = compose_label(struct_info['name'], side=struct_info['side'])
            print name_s

            if struct_info['volume'] is None:
                volume_volResol = None
                origin_wrt_wholebrain_volResol = None
            else:
                volume = bp.unpack_ndarray_str(struct_info['volume'])
                origin_wrt_wholebrain_storedVolResol = struct_info['origin']

                scaling = convert_resolution_string_to_voxel_size(stack=self.stack, resolution=struct_info['resolution']) / self.structure_volume_resolution_um
                volume_volResol = rescale_by_resampling(volume, scaling)
                origin_wrt_wholebrain_volResol = origin_wrt_wholebrain_storedVolResol * scaling

            self.structure_volumes[set_name][name_s] = {
            'volume': volume_volResol,
            'origin': origin_wrt_wholebrain_volResol,
            'edits': struct_info['edits']
            }


            if name_s in selected_structures or name_s in new_structures_to_load:
                self.handle_structure_update(set_name=set_name, name_s=name_s, use_confirmed_only=False, recompute_from_contours=False)
                #
                # for gscene_id, gscene in self.gscenes.iteritems():
                #     gscene.converter.derive_three_view_frames(base_frame_name=name_s,
                #     origin_wrt_wholebrain_um=self.structure_volumes['aligned_atlas'][name_s]['origin'] * self.structure_volume_resolution_um,
                #     zdim_um=self.structure_volumes['aligned_atlas'][name_s]['volume'].shape[2] * self.structure_volume_resolution_um)
                #
                # for gscene in self.gscenes.itervalues():
                #     gscene.update_drawings_from_structure_volume(set_name='aligned_atlas', name_s=name_s, levels=[0.5])
                #


    @pyqtSlot()
    def load_handdrawn_structures(self):
        """
        Load 3-D structure annotations from file. Same as load_structures, except for set name.
        """
        self.load_structures(set_name='handdrawn')

    #
    #     structures_df_fp = str(QFileDialog.getOpenFileName(self, "Choose the structure annotation file",
    #     os.path.join((ANNOTATION_THALAMUS_ROOTDIR if self.prep_id == 3 else ANNOTATION_ROOTDIR), self.stack)))
    #
    #     if structures_df_fp == '':
    #         return
    #
    #     structure_df = load_hdf_v2(structures_df_fp)
    #
    #     for sid, struct_info in structure_df.iterrows():
    #         name_s = compose_label(struct_info['name'], side=struct_info['side'])
    #         print name_s
    #
    #         if struct_info['volume'] is None:
    #             volume_volResol = None
    #             origin_wrt_wholebrain_volResol = None
    #         else:
    #             volume = bp.unpack_ndarray_str(struct_info['volume'])
    #             origin_wrt_wholebrain_storedVolResol = struct_info['origin']
    #
    #             scaling = convert_resolution_string_to_voxel_size(stack=self.stack, resolution=struct_info['resolution']) / self.structure_volume_resolution_um
    #             volume_volResol = rescale_by_resampling(volume, scaling)
    #             origin_wrt_wholebrain_volResol = origin_wrt_wholebrain_storedVolResol * scaling
    #
    #         self.structure_volumes['handdrawn'][name_s] = {
    #         'volume': volume_volResol,
    #         'origin': origin_wrt_wholebrain_volResol,
    #         'edits': struct_info['edits']
    #         }
    #
    #         self.handle_structure_update(set_name='handdrawn', name_s=name_s, use_confirmed_only=False, recompute_from_contours=False)
    #
    #         # for gscene_id, gscene in self.gscenes.iteritems():
    #         #     gscene.converter.derive_three_view_frames(base_frame_name=name_s,
    #         #     origin_wrt_wholebrain_um=self.structure_volumes['aligned_atlas'][name_s]['origin'] * self.structure_volume_resolution_um,
    #         #     zdim_um=self.structure_volumes['aligned_atlas'][name_s]['volume'].shape[2] * self.structure_volume_resolution_um)
    #         #
    #         # for gscene in self.gscenes.itervalues():
    #         #     gscene.update_drawings_from_structure_volume(set_name='aligned_atlas', name_s=name_s, levels=[0.5])

    @pyqtSlot()
    def load_contours(self):
        """
        Load contours. (sagittal only)
        The contour file stores a table: rows are contour IDs, columns are polygon properties.
        """

        if self.prep_id == 3:
            annotation_rootdir = ANNOTATION_THALAMUS_ROOTDIR
        else:
            annotation_rootdir = ANNOTATION_ROOTDIR

        sagittal_contours_df_fp = str(QFileDialog.getOpenFileName(self, "Choose sagittal contour annotation file", os.path.join(annotation_rootdir, self.stack)))
        if sagittal_contours_df_fp == '':
            return

        sagittal_contours_df = load_hdf_v2(sagittal_contours_df_fp)
        sagittal_contours_df_cropped = convert_annotation_v3_original_to_aligned_cropped_v2(sagittal_contours_df, stack=self.stack,\
                                        out_resolution=self.gscenes['main_sagittal'].data_feeder.resolution,
                                        prep_id=self.prep_id)
        sagittal_contours_df_cropped_sagittal = sagittal_contours_df_cropped[(sagittal_contours_df_cropped['orientation'] == 'sagittal')]
        self.gscenes['main_sagittal'].load_drawings(sagittal_contours_df_cropped_sagittal, append=False, vertex_color='b')

    @pyqtSlot()
    def active_image_updated(self):

        if self.gscenes['main_sagittal'].active_section is not None:

            if self.THUMBNAIL_VOLUME_LOADED:

                self.setWindowTitle('BrainLabelingGUI, stack %(stack)s, fn %(fn)s, section %(sec)d, z=%(z).2f, x=%(x).2f, y=%(y).2f voxel units' % \
                dict(stack=self.stack,
                sec=self.gscenes['main_sagittal'].active_section
                if self.gscenes['main_sagittal'].active_section is not None else -1,
                fn=metadata_cache['sections_to_filenames'][self.stack][self.gscenes['main_sagittal'].active_section] \
                if self.gscenes['main_sagittal'].active_section is not None else '',
                z=self.gscenes['tb_sagittal'].active_i if self.gscenes['tb_sagittal'].active_i is not None else 0,
                x=self.gscenes['tb_coronal'].active_i if self.gscenes['tb_coronal'].active_i is not None else 0,
                y=self.gscenes['tb_horizontal'].active_i if self.gscenes['tb_horizontal'].active_i is not None else 0))

            else:
                self.setWindowTitle('BrainLabelingGUI, stack %(stack)s, fn %(fn)s, section %(sec)d' % \
                dict(stack=self.stack,
                sec=self.gscenes['main_sagittal'].active_section
                if self.gscenes['main_sagittal'].active_section is not None else -1,
                fn=metadata_cache['sections_to_filenames'][self.stack][self.gscenes['main_sagittal'].active_section] \
                if self.gscenes['main_sagittal'].active_section is not None else ''))

    @pyqtSlot(object)
    def crossline_updated(self, cross):
        """
        Args:
            cross (3-vector): intersection of the cross wrt wholebrain in raw resolution.
        """

        print 'Update all crosses to', cross, 'from', self.sender().id

        for gscene_id, gscene in self.gscenes.iteritems():

            if self.DISABLE_UPDATE_MAIN_SCENE:
                if gscene_id == 'main_sagittal':
                    continue
            # if gscene_id == 'tb_sagittal':
            # if gscene_id == source_gscene_id: # Skip updating the crossline if the update is triggered from this gscene
            #     continue


            # if gscene.mode == 'crossline': # WHY need this ?
            gscene.update_cross(cross)

    @pyqtSlot(object)
    def drawings_updated(self, polygon):
        print 'Drawings updated.'
        # self.save()

    def reconstruct_structure_from_contours(self, name_s, use_confirmed_only, gscene_id):
        """
        Reconstruct the 3-D structure from 2-D contours.

        Can put this into graphic scene class?

        Args:
            name_s (str): the sided name of the structure to reconstruct.
            use_confirmed_only (bool): if true, reconstruct using only the confirmed contours; if false, use all contours including interpolated ones.
            gscene_id (str): reconstruct based on contours on which graphics scene.

        Returns:
            (volume, origin): in the pre-set volume resolution.
        """

        name_u, side = parse_label(name_s, singular_as_s=True)[:2]

        print 'Re-computing volume of %s from contours.' % name_s
        gscene = self.gscenes[gscene_id]
        contours_xyz_wrt_wholebrain_volResol = []
        for i, polygons in gscene.drawings.iteritems():
            for p in polygons:
                if p.properties['label'] == name_u and p.properties['side'] == side and \
                ((p.properties['type'] == 'confirmed') if use_confirmed_only else True):
                    contour_uvi = [(c.scenePos().x(), c.scenePos().y(), i) for c in p.vertex_circles]
                    # print i, p.properties['type'], contour_uvi
                    contour_wrt_wholebrain_volResol = gscene.converter.convert_frame_and_resolution(contour_uvi,
                    in_wrt=('data', gscene.data_feeder.orientation), in_resolution='image_image_index',
                    out_wrt='wholebrain', out_resolution='volume')
                    contours_xyz_wrt_wholebrain_volResol.append(contour_wrt_wholebrain_volResol)

        if len(contours_xyz_wrt_wholebrain_volResol) < 2:
            raise Exception('%s: Cannot reconstruct structure %s because there are fewer than two confirmed polygons.\n' % (gscene_id, name_s))

        structure_volume_volResol, structure_volume_origin_wrt_wholebrain_volResol = \
        interpolate_contours_to_volume(contours_xyz=contours_xyz_wrt_wholebrain_volResol,
        interpolation_direction='z', return_origin_instead_of_bbox=True)

        return structure_volume_volResol, structure_volume_origin_wrt_wholebrain_volResol

    @pyqtSlot(str, str, bool, bool)
    def handle_structure_update(self, set_name, name_s, use_confirmed_only, recompute_from_contours):
        """
        Handler for the signal structure_updated.
        """

        # Arguments passed in are Qt Strings. This guarantees they are python str.
        # name_u = str(name_u)
        # side = str(side)
        name_s = str(name_s)
        set_name = str(set_name)

        if set_name == 'handdrawn':
            if name_s not in self.structure_volumes[set_name] or recompute_from_contours:
                try:
                    self.structure_volumes[set_name][name_s]['volume'] , \
                    self.structure_volumes[set_name][name_s]['origin'] = \
                    self.reconstruct_structure_from_contours(name_s, use_confirmed_only=use_confirmed_only,  gscene_id=self.sender().id)
                except Exception as e:
                    sys.stderr.write('%s\n' % e)
                    return

        # print set_name, name_s, self.structure_volumes[set_name][name_s]['origin']

        if self.structure_volumes[set_name][name_s]['volume'] is not None \
        and self.structure_volumes[set_name][name_s]['origin'] is not None:

            for gscene_id, gscene in self.gscenes.iteritems():
                gscene.converter.derive_three_view_frames(base_frame_name=name_s,
                origin_wrt_wholebrain_um=self.structure_volumes[set_name][name_s]['origin'] * self.structure_volume_resolution_um,
                zdim_um=self.structure_volumes[set_name][name_s]['volume'].shape[2] * self.structure_volume_resolution_um)

            # if affected_gscenes is None:
            #     affected_gscenes = self.gscenes.keys()

            for gscene_id, gscene in self.gscenes.iteritems():
                # if gscene_id == 'main_sagittal':
                #     gscene.update_drawings_from_structure_volume(name_s=name_s, levels=[.5], set_name=set_name)
                if set_name == 'handdrawn':
                    gscene.update_drawings_from_structure_volume(name_s=name_s, levels=[.5], set_name=set_name)
                elif set_name == 'aligned_atlas':
                    gscene.update_drawings_from_structure_volume(name_s=name_s, levels=[.1, .25, .5, .75, .99], set_name=set_name)
                else:
                    raise

        print '3D structure %s of set %s updated.' % (name_s, set_name)
        self.statusBar().showMessage('3D structure of set %s updated.' % set_name)

    def eventFilter(self, obj, event):
        # print obj.metaObject().className(), event.type()

        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_1:
                self.gscenes['main_sagittal'].show_previous()
            elif key == Qt.Key_2:
                self.gscenes['main_sagittal'].show_next()
            elif key == Qt.Key_3:
                self.gscenes['tb_coronal'].show_previous()
            elif key == Qt.Key_4:
                self.gscenes['tb_coronal'].show_next()
            elif key == Qt.Key_5:
                self.gscenes['tb_horizontal'].show_previous()
            elif key == Qt.Key_6:
                self.gscenes['tb_horizontal'].show_next()
            elif key == Qt.Key_7:
                self.gscenes['tb_sagittal'].show_previous()
            elif key == Qt.Key_8:
                self.gscenes['tb_sagittal'].show_next()

            elif key == Qt.Key_Space:
                if not event.isAutoRepeat():
                    for gscene in self.gscenes.itervalues():
                        gscene.set_mode('crossline')

            elif key == Qt.Key_F:

                ##################### Save structure ######################

                # username = self.get_username()
                timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

                # {(name, side): (vol, bbox, edits, id)}
                new_structure_df = self.structure_df_loaded.copy()

                for (name, side), structure_entry in self.structure_volumes.iteritems():
                    struct_id = structure_entry['structure_id']
                    new_structure_df.loc[struct_id]['volume'] = structure_entry['volume']
                    new_structure_df.loc[struct_id]['bbox'] = structure_entry['bbox']
                    new_structure_df.loc[struct_id]['edits'] = structure_entry['edits']

                if self.prep_id == 3: # thalamus
                    new_structure_df_fp = DataManager.get_annotation_thalamus_filepath(stack=self.stack, by_human=False, stack_m='atlasV3',
                                                                       classifier_setting_m=37,
                                                                      classifier_setting_f=37,
                                                                      warp_setting=8, suffix='structures', timestamp=timestamp)
                else:
                    new_structure_df_fp = DataManager.get_annotation_filepath(stack=self.stack, by_human=False, stack_m='atlasV3',
                                                                       classifier_setting_m=37,
                                                                      classifier_setting_f=37,
                                                                      warp_setting=8, suffix='structures', timestamp=timestamp)
                save_hdf_v2(new_structure_df, new_structure_df_fp)
                self.statusBar().showMessage('3D structure labelings are saved to %s.\n' % new_structure_df_fp)

                # ##################### Save contours #####################

                # self.save()

            # elif key == Qt.Key_A:
            #     print "Reconstructing selected structure volumes..."
            #
            #     curr_structure_label = self.gscenes['main_sagittal'].active_polygon.properties['label']
            #     curr_structure_side = self.gscenes['main_sagittal'].active_polygon.properties['side']
            #     self.update_structure_volume(name_u=curr_structure_label, side=curr_structure_side,
            #     use_confirmed_only=False, recompute_from_contours=False, from_gscene_id='main_sagittal')

            # elif key == Qt.Key_P:
            #     print "Reconstructing all structure volumes..."
            #
            #     structures_curr_section = [(p.properties['label'], p.properties['side'])
            #                                 for p in self.gscenes['main_sagittal'].drawings[self.gscenes['main_sagittal'].active_i]]
            #     for curr_structure_label, curr_structure_side in structures_curr_section:
            #         self.update_structure_volume(name_u=curr_structure_label, side=curr_structure_side, use_confirmed_only=False, recompute_from_contours=False)

            elif key == Qt.Key_U:
                pass

                # For all structures on the current section
                # structures_curr_section = [(p.properties['label'], p.properties['side'])
                #                             for p in self.gscenes['main_sagittal'].drawings[self.gscenes['main_sagittal'].active_i]]
                # for curr_structure_label, curr_structure_side in structures_curr_section:

                # curr_structure_label = self.gscenes['main_sagittal'].active_polygon.properties['label']
                # curr_structure_side = self.gscenes['main_sagittal'].active_polygon.properties['side']
                #
                # name_side_tuple = (curr_structure_label, curr_structure_side)
                # assert name_side_tuple in self.structure_volumes, \
                # "structure_volumes does not have %s. Need to reconstruct this structure first." % str(name_side_tuple)
                #
                # if name_side_tuple in self.gscenes['main_sagittal'].uncertainty_lines:
                #     print "Remove uncertainty line"
                #     for gscene in self.gscenes.itervalues():
                #         gscene.hide_uncertainty_line(name_side_tuple)
                # else:
                #     print "Add uncertainty line"
                #     if curr_structure_side == 'S':
                #         name = curr_structure_label
                #     else:
                #         name = curr_structure_label + '_' + curr_structure_side
                #     current_structure_hessians = DataManager.load_confidence(stack_m='atlasV3', stack_f=self.stack, classifier_setting_m=37, classifier_setting_f=37, warp_setting=8,
                #     param_suffix=name, what='hessians')
                #     H, fmax = current_structure_hessians[84.64]
                #
                #     U, S, UT = np.linalg.svd(H)
                #     flattest_dir = U[:,-1]
                #
                #     current_structure_peakwidth = DataManager.load_confidence(stack_m='atlasV3', stack_f=self.stack, classifier_setting_m=37, classifier_setting_f=37, warp_setting=8,
                #     param_suffix=name, what='peak_radius')
                #     pw_max_um, _, _ = current_structure_peakwidth[118.75][84.64]
                #     len_lossless_res = pw_max_um / convert_resolution_string_to_voxel_size(stack=self.gui.stack, resolution='lossless')
                #
                #     vol = self.structure_volumes[name_side_tuple]['volume']
                #     bbox = self.structure_volumes[name_side_tuple]['bbox']
                #     c_vol_res_gl = np.mean(np.where(vol), axis=1)[[1,0,2]] + (bbox[0], bbox[2], bbox[4])
                #
                #     e1 = c_vol_res_gl * self.volume_downsample_factor - len_lossless_res * flattest_dir
                #     e2 = c_vol_res_gl * self.volume_downsample_factor + len_lossless_res * flattest_dir
                #
                #     for gscene in self.gscenes.itervalues():
                #         e1_gscene = point3d_to_point2d(e1, gscene)
                #         e2_gscene = point3d_to_point2d(e2, gscene)
                #         print gscene.id, e1_gscene, e2_gscene
                #         gscene.set_uncertainty_line(name_side_tuple, e1_gscene, e2_gscene)
                #
                # if name_side_tuple in self.gscenes['main_sagittal'].structure_onscreen_messages:
                #     self.gscenes['main_sagittal'].hide_structure_onscreen_message(name_side_tuple)
                # else:
                #     current_structure_zscores = DataManager.load_confidence(stack_m='atlasV3', stack_f=self.stack, classifier_setting_m=37, classifier_setting_f=37, warp_setting=8,
                #     param_suffix=name, what='zscores')
                #     zscore, fmax, mean, std = current_structure_zscores[118.75]
                #     print str((zscore, fmax, mean, std)), np.array(e1_gscene + e2_gscene)/2
                #     self.gscenes['main_sagittal'].set_structure_onscreen_message(name_side_tuple, "zscore = %.2f" % zscore, (e1_gscene + e2_gscene)/2)

            elif key == Qt.Key_O:
                self.DISABLE_UPDATE_MAIN_SCENE = not self.DISABLE_UPDATE_MAIN_SCENE
                sys.stderr.write("DISABLE_UPDATE_MAIN_SCENE = %s\n" % self.DISABLE_UPDATE_MAIN_SCENE)

        elif event.type() == QEvent.KeyRelease:
            key = event.key()
            if key == Qt.Key_Space:
                if not event.isAutoRepeat():
                    for gscene in self.gscenes.itervalues():
                        gscene.set_mode('idle')

        return False

# def point3d_to_point2d(pt3d, gscene):
#     """
#     Convert a 3D point to 2D point on a gscene.
#
#     Args:
#         pt3d ((3,)-ndarray): a point coordinate in lossless-resolution coordinate.
#         gscene (QGraphicScene)
#     """
#     pt3d_gscene_res = pt3d / gscene.data_feeder.downsample
#     if gscene.id == 'main_sagittal' or gscene.id == 'tb_sagittal':
#         pt2d = (pt3d_gscene_res[0], pt3d_gscene_res[1])
#     elif gscene.id == 'tb_coronal':
#         pt2d = (gscene.data_feeder.z_dim - 1 - pt3d_gscene_res[2], pt3d_gscene_res[1])
#     elif gscene.id == 'tb_horizontal':
#         pt2d = (pt3d_gscene_res[0], gscene.data_feeder.z_dim - 1 - pt3d_gscene_res[2])
#     return np.array(pt2d)

def load_structure_names(fn):
    """
    Load structure names from a file.

    Args:
        fn (str): a file containing rows of structure names.
    Returns:
        (list of str)
    """
    names = {}
    with open(fn, 'r') as f:
        for ln in f.readlines():
            abbr, fullname = ln.split('\t')
            names[abbr] = fullname.strip()
    return names


if __name__ == "__main__":

    import argparse
    from sys import argv, exit

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Launch brain labeling GUI.')

    parser.add_argument("stack_name", type=str, help="Stack name")
    # parser.add_argument("-f", "--first_sec", type=int, help="First section")
    # parser.add_argument("-l", "--last_sec", type=int, help="Last section")
    # parser.add_argument("-v", "--img_version", type=str, help="Image version. Default = %(default)s.", default='jpeg')
    parser.add_argument("-v", "--img_version", type=str, help="Image version (jpeg or grayJpeg). Default = %(default)s.", default='grayJpeg')
    parser.add_argument("-r", "--resolution", type=str, help="Resolution of image displayed in main scene. Default = %(default)s.", default='raw')
    parser.add_argument("-p", "--prep", type=int, help="Frame identifier of image displayed in main scene (0 for no alignment or crop, 2 for brainstem crop, 3 for thalamus crop). Default = %(default)d.", default=2)
    args = parser.parse_args()

    appl = QApplication(argv)

    stack = args.stack_name
    resolution = args.resolution
    img_version = args.img_version
    prep_id = args.prep

    # default_first_sec, default_last_sec = DataManager.load_cropbox(stack, prep_id=prep_id)[4:]

    # first_sec = default_first_sec if args.first_sec is None else args.first_sec
    # last_sec = default_last_sec if args.last_sec is None else args.last_sec

    m = BrainLabelingGUI(stack=stack,
    # first_sec=first_sec, last_sec=last_sec,
    resolution=resolution, img_version=img_version, prep_id=prep_id)

    m.showMaximized()
    m.raise_()
    exit(appl.exec_())
