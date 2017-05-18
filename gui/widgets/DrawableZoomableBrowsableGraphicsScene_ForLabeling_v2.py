import sys, os
from collections import defaultdict
from datetime import datetime

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from multiprocess import Pool

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from data_manager import DataManager
from metadata import *
from annotation_utilities import *
from registration_utilities import *
from gui_utilities import *

from DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene

CROSSLINE_PEN_WIDTH = 2
CROSSLINE_RED_PEN = QPen(Qt.red)
CROSSLINE_RED_PEN.setWidth(CROSSLINE_PEN_WIDTH)

reference_resources = {
'5N': {'BrainInfo': 'http://braininfo.rprc.washington.edu/centraldirectory.aspx?ID=559',
        'PubMed': 'https://www.ncbi.nlm.nih.gov/pubmed/query.fcgi?cmd=search&db=PubMed&term=%22motor+nucleus%22+OR+%22motor+nucleus+of+trigeminal+nerve%22+OR+%22motor+trigeminal+nucleus%22+OR+%22Nucleus+motorius+nervi+trigemini%22+OR+%22trigeminal+motor+nucleus%22&dispmax=20&relentrezdate=No+Limit',
        'Allen Reference Atlas (Sagittal)': 'http://atlas.brain-map.org/atlas?atlas=2&plate=100883869#atlas=2&plate=100883869&resolution=6.98&x=10959.666748046875&y=5154.666748046875&zoom=-2&structure=621',
        'Allen Reference Atlas (Coronal)': 'http://atlas.brain-map.org/atlas?atlas=1#atlas=1&structure=621&resolution=8.38&x=4728&y=3720&zoom=-2&plate=100960192'},
'7N': {'BrainInfo': 'http://braininfo.rprc.washington.edu/centraldirectory.aspx?ID=586',
        'PubMed': 'https://www.ncbi.nlm.nih.gov/pubmed/query.fcgi?cmd=search&db=PubMed&term=%22facial+motor+nucleus%22+OR+%22facial+nucleus%22+OR+%22Nucleus+facialis%22+OR+%22Nucleus+nervi+facialis%22+AND+facial+nucleus&dispmax=20&relentrezdate=No+Limit',
        'Allen Reference Atlas (Sagittal)': 'http://atlas.brain-map.org/atlas?atlas=2&plate=100883869#atlas=2&plate=100883869&resolution=6.98&x=10959.666748046875&y=5154.666748046875&zoom=-2&structure=661',
        'Allen Reference Atlas (Coronal)': 'http://atlas.brain-map.org/atlas?atlas=1#atlas=1&structure=661&resolution=6.98&x=6039.549458821615&y=4468.1439208984375&zoom=-2&plate=100960181'}
}

class DrawableZoomableBrowsableGraphicsScene_ForLabeling(DrawableZoomableBrowsableGraphicsScene):
    """
    Used for annotation GUI.
    """

    drawings_updated = pyqtSignal(object)
    crossline_updated = pyqtSignal(int, int, int, str)
    update_structure_volume_requested = pyqtSignal(object)
    active_image_updated = pyqtSignal()

    def __init__(self, id, gui=None, gview=None, parent=None):
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).__init__(id=id, gview=gview, parent=parent)

        # self.pixmapItem = QGraphicsPixmapItem()
        # self.addItem(self.pixmapItem)

        self.gui = gui

        # self.gview = gview
        # self.id = id

        # self.polygonElements = defaultdict(dict)
        # self.qimages = None
        # self.active_section = None
        # self.active_i = None
        # self.active_dataset = None

        # self.installEventFilter(self)

        self.showing_which = 'histology'

        self.dont_add_vertex = False

        # self.drawings = defaultdict(list)

        self.hline = QGraphicsLineItem()
        self.hline.setPen(CROSSLINE_RED_PEN)
        self.vline = QGraphicsLineItem()
        self.vline.setPen(CROSSLINE_RED_PEN)
        self.addItem(self.hline)
        self.addItem(self.vline)
        self.hline.setVisible(False)
        self.vline.setVisible(False)

        # self.gview.setMouseTracking(False)
        # self.gview.setVerticalScrollBarPolicy( Qt.ScrollBarAlwaysOff )
        # self.gview.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOff )
        # self.gview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # self.gview.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # Important! default is AnchorViewCenter.
        # self.gview.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # self.gview.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.gview.setDragMode(QGraphicsView.ScrollHandDrag)
        # gview.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # if not hasattr(self, 'contextMenu_set') or (hasattr(self, 'contextMenu_set') and not self.contextMenu_set):
        #     self.gview.customContextMenuRequested.connect(self.show_context_menu)

        # self.gview.installEventFilter(self)
        # self.gview.viewport().installEventFilter(self)

        # self.set_mode('idle')

    # def set_mode(self, mode):

        # if hasattr(self, 'mode'):
        #     print 'Mode change:', self.mode, '=>', mode
        #
        # if mode == 'delete vertices':
        #     self.gview.setDragMode(QGraphicsView.RubberBandDrag)
        #
        # elif mode == 'add vertices consecutively':
        #     self.gview.setDragMode(QGraphicsView.NoDrag)
        #     # for p in self.drawings[self.active_i]:
        #         # p.setFlag(QGraphicsItem.ItemIsMovable, False)
        #
        # elif mode == 'idle':
        #     self.gview.setDragMode(QGraphicsView.ScrollHandDrag)
        #     if hasattr(self, 'mode') and self.mode == 'crossline':
        #         self.hline.setVisible(False)
        #         self.vline.setVisible(False)
        #         # self.removeItem(self.hline)
        #         # self.removeItem(self.vline)
        #     # for p in self.drawings[self.active_i]:
        #     #     p.setFlag(QGraphicsItem.ItemIsMovable, True)
        #
        # elif mode == 'crossline':
        #     self.hline.setVisible(True)
        #     self.vline.setVisible(True)
        #
        # self.mode = mode

    # def set_data_feeder(self, feeder):
    #     if hasattr(self, 'data_feeder') and self.data_feeder == feeder:
    #         return
    #
    #     self.data_feeder = feeder
    #
    #     if self.data_feeder.downsample == 32:
    #         self.set_vertex_radius(4)
    #     elif self.data_feeder.downsample == 1:
    #         self.set_vertex_radius(20)
    #
    #     if self.data_feeder.downsample == 32:
    #         self.set_line_width(3)
    #     elif self.data_feeder.downsample == 1:
    #         self.set_line_width(10)
    #
    #     self.active_section = None
    #     self.active_i = None
    #
    #     # if hasattr(self, 'active_i') and self.active_i is not None:
    #     #     self.update_image()
    #     #     self.active_image_updated.emit()


    def set_structure_volumes(self, structure_volumes):
        self.structure_volumes = structure_volumes

    # def check_structure_side_properties(self):
    #     return True

    def update_drawings_from_structure_volume(self, name_u, side):
        """
        Based on reconstructed 3D structure,
        """

        volume, bbox = self.structure_volumes[name_u]
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        print 'volume', volume.shape, xmin, xmax, ymin, ymax, zmin, zmax

        volume_downsample_factor = self.gui.volume_downsample_factor
        # xmin_lossless, xmax_lossless, ymin_lossless, ymax_lossless, zmin_lossless, zmax_lossless = np.array(bbox) * downsample
        bbox_lossless = np.array(bbox) * volume_downsample_factor

        downsample = self.data_feeder.downsample

        if volume_downsample_factor <= downsample:

            volume_downsampled = volume[::downsample/volume_downsample_factor, ::downsample/volume_downsample_factor, ::downsample/volume_downsample_factor]
            xmin_ds, xmax_ds, ymin_ds, ymax_ds, zmin_ds, zmax_ds = np.array(bbox_lossless) / downsample

            print 'volume_downsampled', volume_downsampled.shape, xmin_ds, xmax_ds, ymin_ds, ymax_ds, zmin_ds, zmax_ds

        matched_confirmed_polygons = [(i, p) for i, polygons in self.drawings.iteritems()
                                    for p in polygons if p.label == name_u and p.type != 'interpolated' and p.side == side]

        # matched_unconfirmed_polygons = [(i, p) for i, polygons in self.drawings.iteritems() for p in polygons if p.label == name_u and p.type == 'interpolated']
        # for i, p in matched_unconfirmed_polygons:
        #     if i == self.active_i:
        #         self.removeItem(p)
        #     self.drawings[i].remove(p)

        # if self.data_feeder.orientation == 'sagittal':
        if hasattr(self.data_feeder, 'sections'):
            assert self.data_feeder.orientation == 'sagittal'
            # sec_min = DataManager.convert_z_to_section(stack=self.data_feeder.stack, z=zmin, downsample=downsample)
            # matched_confirmed_sections = [self.data_feeder.sections[i] for i, p in matched_confirmed_polygons]
            matched_confirmed_sections = [self.data_feeder.sections[i] for i, p in matched_confirmed_polygons]

            if len(matched_confirmed_sections) > 0:
                min_sec = np.min(matched_confirmed_sections)
                max_sec = np.max(matched_confirmed_sections)
            else:
                min_sec = DataManager.convert_z_to_section(stack=self.data_feeder.stack, z=zmin, downsample=volume_downsample_factor)
                max_sec = DataManager.convert_z_to_section(stack=self.data_feeder.stack, z=zmax, downsample=volume_downsample_factor)

            for sec in range(min_sec, max_sec+1):

                # remove if this section has interpolated polygon
                # if sec not in self.data_feeder.sections:
                if sec not in self.data_feeder.sections:
                    sys.stderr.write('Section %d is not loaded.\n' % sec)
                    continue

                # i = self.data_feeder.sections.index(sec)
                i = self.data_feeder.sections.index(sec)
                matched_unconfirmed_polygons_to_remove = [p for p in self.drawings[i] if p.label == name_u and p.type == 'interpolated' and p.side == side]
                for p in matched_unconfirmed_polygons_to_remove:
                    self.drawings[i].remove(p)
                    if i == self.active_i:
                        self.removeItem(p)

                if sec in matched_confirmed_sections:
                    continue


                z0, z1 = DataManager.convert_section_to_z(stack=self.data_feeder.stack, sec=sec, downsample=downsample)
                # z_currResol = int(np.round((z0 + z1)/2))
                z_currResol = .5 * z0 + .5 * z1
                z_volResol = int(np.round(z_currResol * downsample / volume_downsample_factor))
                # (int(np.ceil(z0)) + int(np.floor(z1))) / 2
                # z_volResol = z_currResol * downsample / volume_downsample_factor
                print sec, z0, z1, z_currResol, z_volResol, zmin
                # if downsample == 32:
                cnts_volResol = find_contour_points(volume[:, :, z_volResol - zmin].astype(np.uint8), sample_every=20)

                # print cnts_volResol

                if len(cnts_volResol) > 0 and 1 in cnts_volResol:
                    # print x_ds
                    xys_volResol = np.array(cnts_volResol[1][0])
                    gscene_xs_volResol = xys_volResol[:,0] + xmin # the coordinate on gscene's x axis
                    gscene_ys_volResol = xys_volResol[:,1] + ymin
                    gscene_points_volResol = np.c_[gscene_xs_volResol, gscene_ys_volResol]
                    gscene_points_currResol = gscene_points_volResol * volume_downsample_factor / downsample
                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_points_currResol),
                                                            label=name_u, linecolor='g', section=sec, type='interpolated',
                                                            side=side,
                                                            side_manually_assigned=False)
        else:
            # raise Exception('Sagittal interpolation on volume data is not implemented.')

            matched_confirmed_positions = [i for i, p in matched_confirmed_polygons]

            if self.data_feeder.orientation == 'sagittal':
                posmin_ds = zmin_ds
                posmax_ds = zmin_ds + volume_downsampled.shape[2] - 1
            elif self.data_feeder.orientation == 'coronal':
                posmin_ds = xmin_ds
                posmax_ds = xmin_ds + volume_downsampled.shape[1] - 1
            elif self.data_feeder.orientation == 'horizontal':
                posmin_ds = ymin_ds
                posmax_ds = ymin_ds + volume_downsampled.shape[0] - 1

            for pos_ds in range(posmin_ds, posmax_ds+1):

                # if pos_ds in matched_confirmed_positions:
                #     continue

                # remove if this section has interpolated polygon
                matched_unconfirmed_polygons_to_remove = [p for p in self.drawings[pos_ds] if p.label == name_u and p.type == 'interpolated' and p.side == side]
                for p in matched_unconfirmed_polygons_to_remove:
                    self.drawings[pos_ds].remove(p)
                    if pos_ds == self.active_i:
                        self.removeItem(p)

                if self.data_feeder.orientation == 'sagittal':
                    raise Exception('Not implemented.')

                elif self.data_feeder.orientation == 'coronal':

                    cnts = find_contour_points(volume_downsampled[:, pos_ds-posmin_ds, :].astype(np.uint8), sample_every=max(20/downsample, 10))
                    if len(cnts) == 0 or 1 not in cnts:
                        sys.stderr.write('%s: Contour not found with reconstructed volume.\n' % self.id)
                        continue
                        # Contour for label 1 (which is the only label in the boolean volume)
                    zys = np.array(cnts[1][0])
                    gscene_xs = self.data_feeder.z_dim - 1 - (zys[:,0] + zmin_ds) # the coordinate on gscene's x axis
                    gscene_ys = zys[:,1] + ymin_ds

                elif self.data_feeder.orientation == 'horizontal':

                    cnts = find_contour_points(volume_downsampled[pos_ds-posmin_ds, :, :].astype(np.uint8), sample_every=max(20/downsample, 10))
                    if len(cnts) == 0 or 1 not in cnts:
                        sys.stderr.write('%s: Contour not found with reconstructed volume.\n' % self.id)
                        continue

                    zxs = np.array(cnts[1][0])
                    gscene_xs = zxs[:,1] + xmin_ds
                    gscene_ys = self.data_feeder.z_dim - 1 - (zxs[:,0] + zmin_ds) # the coordinate on gscene's x axis

                pts_on_gscene = np.c_[gscene_xs, gscene_ys]
                self.add_polygon_with_circles_and_label(path=vertices_to_path(pts_on_gscene), label=name_u,
                                                        linecolor='g', vertex_radius=1, linewidth=2, index=pos_ds,
                                                        type='interpolated',
                                                        side=side,
                                                        side_manually_assigned=False)

    # def set_active_i(self, i, update_crossline=True, emit_changed_signal=True):
    #
    #     if i == self.active_i:
    #         return
    #
    #     old_i = self.active_i
    #
    #     print self.id, ': Set active index to', i, ', update_crossline', update_crossline
    #
    #     self.active_i = i
    #     if hasattr(self.data_feeder, 'sections'):
    #         self.active_section = self.data_feeder.sections[self.active_i]
    #
    #     try:
    #         self.update_image()
    #     except Exception as e: # if failed, do not change active_i or active_section
    #         if old_i is not None:
    #             self.active_i = old_i
    #             if hasattr(self.data_feeder, 'sections'):
    #                 self.active_section = self.data_feeder.sections[old_i]
    #         raise e
    #
    #     for polygon in self.drawings[old_i]:
    #         self.removeItem(polygon)
    #
    #     for polygon in self.drawings[i]:
    #         self.addItem(polygon)
    #
    #     if emit_changed_signal:
    #         self.active_image_updated.emit()
    #
    #     # if update_crossline and self.mode == 'crossline':
    #
    #     if update_crossline and hasattr(self, 'cross_x_lossless'):
    #         if hasattr(self.data_feeder, 'sections'):
    #             d1, d2 = self.convert_section_to_z(sec=self.active_section, downsample=1)
    #             cross_depth_lossless = .5 * d1 + .5 * d2
    #             # print 'cross_depth_lossless 1', cross_depth_lossless
    #         else:
    #             print 'active_i =', self.active_i, 'downsample =', self.data_feeder.downsample
    #             cross_depth_lossless = self.active_i * self.data_feeder.downsample
    #             # print 'cross_depth_lossless 2', cross_depth_lossless
    #
    #         if self.data_feeder.orientation == 'sagittal':
    #             self.cross_z_lossless = cross_depth_lossless
    #         elif self.data_feeder.orientation == 'coronal':
    #             self.cross_x_lossless = cross_depth_lossless
    #         elif self.data_feeder.orientation == 'horizontal':
    #             self.cross_y_lossless = cross_depth_lossless
    #
    #         print self.id, ': emit', self.cross_x_lossless, self.cross_y_lossless, self.cross_z_lossless
    #         self.crossline_updated.emit(self.cross_x_lossless, self.cross_y_lossless, self.cross_z_lossless, self.id)
    #         # self.crossline_updated.emit(int(np.round(self.cross_x_lossless)), int(np.round(self.cross_y_lossless)), int(np.round(self.cross_z_lossless)), self.id)


    # def set_active_section(self, sec, emit_changed_signal=True, update_crossline=True):
    #
    #     if sec == self.active_section:
    #         return
    #
    #     print self.id, ': Set active section to', sec
    #
    #     if hasattr(self.data_feeder, 'sections'):
    #         assert sec in self.data_feeder.sections, 'Section %s is not loaded.' % str(sec)
    #         i = self.data_feeder.sections.index(sec)
    #         self.set_active_i(i, emit_changed_signal=emit_changed_signal, update_crossline=update_crossline)
    #
    #     self.active_section = sec

    def update_image(self):
        if self.showing_which == 'histology':
            self.load_histology()
        elif self.showing_which == 'scoremap':
            self.load_scoremap()

    def set_active_structure(self, name_s=None, name_u=None):
        if name_s is not None:
            self.active_structure = name_s
        else:
            self.active_structure = name_u

    def set_downsample_factor(self, downsample):
        if self.data_feeder.downsample == downsample:
            return
        # if self.downsample == downsample:
        #     return
        #
        # self.downsample = downsample
        self.data_feeder.set_downsample_factor(downsample)
        self.update_image()


    def load_scoremap(self, name_u=None, sec=None):
        if sec is None:
            assert self.active_section is not None
            sec = self.active_section

        if name_u is None:
            assert self.active_structure is not None
            name_u = labelMap_sidedToUnsided[self.active_structure]

        scoremap_viz_fn = '/home/yuncong/CSHL_scoremapViz_svm_Sat16ClassFinetuned_v3/%(stack)s/%(sec)04d/%(stack)s_%(sec)04d_roi1_scoremapViz_%(name)s.jpg' % \
        {'sec': self.active_section, 'stack': self.gui.stack, 'name': name_u}

        w, h = DataManager.get_image_dimension(self.gui.stack)
        scoremap_pixmap = QPixmap(scoremap_viz_fn).scaled(w, h)
        self.pixmapItem.setPixmap(scoremap_pixmap)
        self.showing_which = 'scoremap'


    def load_histology(self, i=None, sec=None):

        if i is None:
            i = self.active_i
            # assert i >= 0 and i < len(self.data_feeder.sections)
        elif hasattr(self.data_feeder, 'sections') and self.data_feeder.sections is not None:
        # elif self.data_feeder.sections is not None:
            if sec is None:
                sec = self.active_section
            # i = self.data_feeder.sections.index(sec)
            assert sec in self.data_feeder.sections
            i = self.data_feeder.sections.index(sec)

        print i
        image = self.data_feeder.retrive_i(i=i)

        histology_pixmap = QPixmap.fromImage(image)

        # histology_pixmap = QPixmap.fromImage(self.qimages[sec])
        self.pixmapItem.setPixmap(histology_pixmap)
        self.pixmapItem.setVisible(True)
        self.showing_which = 'histology'

        self.set_active_i(i)

    # def show_next(self, cycle=False):
    #     # if self.indexing_mode == 'section':
    #     #     self.set_active_section(min(self.active_section + 1, self.data_feeder.last_sec))
    #     # elif self.indexing_mode == 'voxel':
    #     #     self.set_active_i(min(self.active_i + 1, self.data_feeder.n - 1))
    #
    #     if cycle:
    #         self.set_active_i((self.active_i + 1) % self.data_feeder.n)
    #     else:
    #         # If next image is not valid, show the one after the next
    #         t = 1
    #         while self.active_i + t <= self.data_feeder.n - 1:
    #             try:
    #                 self.set_active_i(min(self.active_i + t, self.data_feeder.n - 1))
    #                 # self.set_active_i(min(self.active_i + t, np.max(self.data_feeder.sections)))
    #                 break
    #             except:
    #                 t += 1

            # self.set_active_i(min(self.active_i + 1, self.data_feeder.n - 1)

    # def show_previous(self, cycle=False):
    #     # if self.indexing_mode == 'section':
    #     #     self.set_active_section(max(self.active_section - 1, self.data_feeder.first_sec))
    #     # elif self.indexing_mode == 'voxel':
    #     #     self.set_active_i(max(self.active_i - 1, 0))
    #
    #     if cycle:
    #         self.set_active_i((self.active_i - 1) % self.data_feeder.n)
    #     else:
    #         # If previous image is not valid, show the one before the previous
    #         t = 1
    #         while self.active_i - t >= 0:
    #             try:
    #                 self.set_active_i(max(self.active_i - t, 0))
    #                 break
    #             except:
    #                 t += 1
    #
    #         # self.set_active_i(max(self.active_i - 1, 0))

    def infer_side(self):

        label_section_lookup = self.get_label_section_lookup()

        structure_ranges = get_landmark_range_limits_v2(stack=self.data_feeder.stack, label_section_lookup=label_section_lookup)

        print 'structure_ranges', structure_ranges

        for section_index, polygons in self.drawings.iteritems():
            for p in polygons:
                if p.label in structure_ranges:
                    assert p.label in singular_structures, 'Label %s is in structure_ranges, but it is not singular.' % p.label
                    if section_index >= structure_ranges[p.label][0] and section_index <= structure_ranges[p.label][1]:
                        if p.side is None or not p.side_manually_assigned:
                            p.set_side('S', side_manually_assigned=False)
                    else:
                        raise Exception('Polygon is on a section not in structure_range.')
                else:
                    lname = convert_to_left_name(p.label)
                    if lname in structure_ranges:
                        if section_index >= structure_ranges[lname][0] and section_index <= structure_ranges[lname][1]:
                            if p.side is None or not p.side_manually_assigned:
                                p.set_side('L', side_manually_assigned=False)
                                sys.stderr.write('%d, %d %s set to L\n' % (section_index, self.data_feeder.sections[section_index], p.label))

                    rname = convert_to_right_name(p.label)
                    if rname in structure_ranges:
                        if section_index >= structure_ranges[rname][0] and section_index <= structure_ranges[rname][1]:
                            if p.side is None or not p.side_manually_assigned:
                                p.set_side('R', side_manually_assigned=False)
                                sys.stderr.write('%d, %d %s set to R\n' % (section_index, self.data_feeder.sections[section_index], p.label))


    def set_conversion_func_section_to_z(self, func):
        self.convert_section_to_z = func

    def set_conversion_func_z_to_section(self, func):
        self.convert_z_to_section = func


    # def add_polygon_with_circles_and_label(self, path, linecolor='r', linewidth=None, vertex_color='b', vertex_radius=None,
    #                                         label='unknown', section=None, label_pos=None, index=None, type=None,
    #                                         edit_history=[], side=None, side_manually_assigned=None,
    #                                         contour_id=None):
    #     '''
    #     Function for adding polygon, along with vertex circles.
    #     Step 1: create polygon
    #     Step 2: create vertices
    #     Step 3: reorder overlapping polygons if any
    #     Step 4: add label
    #     '''
    #
    #     ## CHECK THIS!!!!
    #
    #     # self.history_allSections[sec].append({
    #     #     'type': 'add_polygon_by_vertices_label_begin'
    #     #     })
    #
    #     if index is None and section is None:
    #         index = self.active_i
    #     elif section is not None:
    #         # if section in self.data_feeder.sections:
    #         if section in self.data_feeder.sections:
    #             # index = self.data_feeder.sections.index(section)
    #             index = self.data_feeder.sections.index(section)
    #         else:
    #             sys.stderr.write('Trying to add polygon, but section %d is not loaded - add polygon anyway.\n' % section)
    #             raise Exception('Not implemented.') # CANNOT ASSUME ALL HAVE INDEX ...
    #
    #     polygon = self.add_polygon(path, color=linecolor, linewidth=linewidth, index=index, section=section)
    #
    #     if vertex_radius is None:
    #         vertex_radius = self.vertex_radius
    #
    #     polygon.add_circles_for_all_vertices(radius=vertex_radius, color=vertex_color)
    #
    #     polygon.set_label(label, label_pos)
    #     polygon.set_closed(True)
    #     polygon.set_type(type)
    #     polygon.set_side(side=side, side_manually_assigned=side_manually_assigned)
    #
    #     if edit_history is None or len(edit_history) == 0:
    #         polygon.add_edit(editor=self.gui.get_username())
    #     else:
    #         polygon.set_edit_history(edit_history)
    #
    #     polygon.set_contour_id(contour_id) # Could be None - will be generated new in convert_drawings_to_entries()
    #     return polygon


    # def add_polygon(self, path=QPainterPath(), color='r', linewidth=None, z_value=50,
    #                 uncertain=False, section=None, index=None, vertex_radius=None):
    #     '''
    #     Add a polygon.
    #
    #     Args:
    #         path (QPainterPath): path of the polygon
    #         pen (QPen): pen used to draw polygon
    #
    #     Returns:
    #         QGraphicsPathItemModified: added polygon
    #
    #     '''
    #
    #     # if path is None:
    #     #     path = QPainterPath()
    #
    #     if color == 'r':
    #         pen = QPen(Qt.red)
    #     elif color == 'g':
    #         pen = QPen(Qt.green)
    #     elif color == 'b':
    #         pen = QPen(Qt.blue)
    #
    #     if linewidth is None:
    #         linewidth = self.line_width
    #
    #     pen.setWidth(linewidth)
    #
    #     if index is None and section is None:
    #         if hasattr(self, 'active_i'):
    #             index = self.active_i
    #     elif section is not None:
    #         # if section in self.data_feeder.sections:
    #         if section in self.data_feeder.sections:
    #             # index = self.data_feeder.sections.index(section)
    #             index = self.data_feeder.sections.index(section)
    #         else:
    #             raise Exception('Not implemented.')
    #
    #     if hasattr(self.data_feeder, 'sections'):
    #         # sec = self.data_feeder.sections[index]
    #         sec = self.data_feeder.sections[index]
    #         z0, z1 = self.convert_section_to_z(sec=sec, downsample=self.data_feeder.downsample)
    #         pos = (z0 + z1) / 2
    #         # if len(z) == 2: # a section corresponds to more than one z values; returned result is a pair indicating first and last z's.
    #         #     pos = (z[0]+z[1])/2
    #         # else:
    #         #     pos = z
    #     else:
    #         pos = index
    #
    #     if vertex_radius is None:
    #         vertex_radius = self.vertex_radius
    #
    #     polygon = QGraphicsPathItemModified(path, gscene=self, index=index, orientation=self.data_feeder.orientation, position=pos, vertex_radius=vertex_radius)
    #
    #     polygon.setPen(pen)
    #     polygon.setZValue(z_value)
    #     # polygon.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
    #     polygon.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
    #     polygon.setFlag(QGraphicsItem.ItemIsMovable, False)
    #
    #     polygon.signal_emitter.press.connect(self.polygon_press)
    #     polygon.signal_emitter.release.connect(self.polygon_release)
    #     # polygon.moved.connect(self.polygon_moved)
    #     # polygon.o.released.connect(self.polygon_released)
    #
    #     # polygon.signal_emitter.clicked.connect(self.polygon_pressed)
    #     # polygon.signal_emitter.moved.connect(self.polygon_moved)
    #     # polygon.signal_emitter.released.connect(self.polygon_released)
    #     # polygon.signal_emitter.vertex_added.connect(self.vertex_added)
    #
    #     polygon.signal_emitter.vertex_added.connect(self.vertex_added)
    #     polygon.signal_emitter.label_added.connect(self.label_added)
    #     polygon.signal_emitter.evoke_label_selection.connect(self.label_selection_evoked)
    #     polygon.signal_emitter.polygon_completed.connect(self.polygon_completed)
    #
    #     self.drawings[index].append(polygon)
    #
    #     # if adding polygon to current section
    #     # if sec == self.active_section:
    #     if index == self.active_i:
    #         print 'polygon added.'
    #         self.addItem(polygon)
    #     # self.polygonElements[sec][polygon] = {'vertexCircles': [], 'uncertain': uncertain}
    #
    #     # self.map_polygon_to_section[polygon] = sec
    #
    #     # self.history_allSections[sec].append({
    #     #     'type': 'add_polygon',
    #     #     'polygon': polygon
    #     #     })
    #
    #     return polygon


    def open_label_selection_dialog(self):

        # Put recent labels in the front of the list.
        # from collections import OrderedDict

        # if len(self.gui.recent_labels) > 0:
        #     tuples_recent = [(abbr, fullname) for abbr, fullname in self.gui.structure_names.iteritems() if abbr in self.gui.recent_labels]
        #     tuples_nonrecent = [(abbr, fullname) for abbr, fullname in self.gui.structure_names.iteritems() if abbr not in self.gui.recent_labels]
        #     structure_names = OrderedDict(tuples_recent + tuples_nonrecent)
        # else:
        #     structure_names = self.gui.structure_names

        rearranged_labels = [abbr + '(' + fullname + ')' for abbr, fullname in self.gui.structure_names.iteritems() if abbr in self.gui.recent_labels] + \
                            [abbr + '(' + fullname + ')' for abbr, fullname in self.gui.structure_names.iteritems() if abbr not in self.gui.recent_labels]

        self.label_selection_dialog = AutoCompleteInputDialog(parent=self.gui, labels=rearranged_labels)
        self.label_selection_dialog.setWindowTitle('Select Structure Name')

        # if hasattr(self, 'invalid_labelname'):
        #     print 'invalid_labelname', self.invalid_labelname
        # else:
        #     print 'no labelname set'

        if hasattr(self.active_polygon, 'label'):
            abbr_unsided = self.active_polygon.label
            # if '_' in abbr: # if side has been set
            #     abbr = abbr[:-2]
            # abbr_unsided = abbr[:-2] if '_L' in abbr or '_R' in abbr else abbr

            self.label_selection_dialog.comboBox.setEditText( abbr_unsided + ' (' + self.gui.structure_names[abbr_unsided] + ')')

        # else:
        #     self.label = ''

        self.label_selection_dialog.set_test_callback(self.label_confirmed) # Callback when OK is clicked or Enter is pressed.

        # self.label_selection_dialog.accepted.connect(self.label_dialog_text_changed)
        # self.label_selection_dialog.textValueSelected.connect(self.label_dialog_text_changed)

        self.label_selection_dialog.exec_()

        # choose left or right side

        # self.left_right_selection_dialog = QInputDialog(self)
        # self.left_right_selection_dialog.setLabelText('Enter L or R, or leave blank for single structure')
        #
        # if self.selected_section < (self.first_sec + self.last_sec)/2:
        #     self.left_right_selection_dialog.setTextValue(QString('L'))
        # else:
        #     self.left_right_selection_dialog.setTextValue(QString('R'))
        #
        # self.left_right_selection_dialog.exec_()
        #
        # left_right = str(self.left_right_selection_dialog.textValue())
        #
        # if left_right == 'L' or left_right == 'R':
        #     abbr = self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label']
        #     abbr_sided = abbr + '_' + left_right
        #     self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label'] = abbr_sided
        #     self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'].setText(abbr_sided)

    def label_confirmed(self):

        text = str(self.label_selection_dialog.comboBox.currentText())

        # Parse the name text.
        import re
        m = re.match('^(.+?)\s*\((.+)\)$', text)

        if m is None:
            QMessageBox.warning(self.gview, 'oops', 'structure name must be of the form "abbreviation (full description)"')
            return

        else:
            abbr, fullname = m.groups()
            if not (abbr in self.gui.structure_names.keys() and fullname in self.gui.structure_names.values()):  # new label
                if abbr in self.gui.structure_names:
                    QMessageBox.warning(self.gview, 'oops', 'structure with abbreviation %s already exists: %s' % (abbr, fullname))
                    return
                else:
                    self.gui.structure_names[abbr] = fullname
                    self.gui.new_labelnames[abbr] = fullname

        # print self.accepted_proposals_allSections.keys()
        # print self.selected_section

        # self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label'] = abbr
        self.active_polygon.set_label(abbr)

        # if 'labelTextArtist' in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon] and \
        #         self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'] is not None:
        #     # label exists
        #     self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'].setText(abbr)
        # else:
        #     # label not exist, create
        #     self.add_label_to_polygon(self.selected_polygon, abbr)

        if abbr in self.gui.recent_labels:
            self.gui.recent_labels = [abbr] + [x for x in self.gui.recent_labels if x != abbr]
        else:
            self.gui.recent_labels.insert(0, abbr)

        self.label_selection_dialog.accept()


    def load_drawings(self, contours, append=False):
        """
        Load annotation contours and place drawings.
        """

        CONTOUR_IS_INTERPOLATED = 1

        if not append:
            self.drawings = defaultdict(list)

        endorser_contour_lookup = defaultdict(set)
        for cnt_id, contour in contours.iterrows():
            for editor in set([edit['username'] for edit in contour['edits']]):
                endorser_contour_lookup[editor].add(cnt_id)
            endorser_contour_lookup[contour['creator']].add(cnt_id)

        grouped = contours.groupby('section')

        for sec, group in grouped:
            for contour_id, contour in group.iterrows():
                vertices = contour['vertices']
                contour_type = 'interpolated' if contour['flags'] & CONTOUR_IS_INTERPOLATED else None
                # endorsers = set([edit['username'] for edit in contour['edits']] + [contour['creator']])
                if sec == 201:
                    print  contour['name'], sec, metadata_cache['sections_to_filenames'][self.data_feeder.stack][sec]
                self.add_polygon_with_circles_and_label(path=vertices_to_path(vertices), label=contour['name'], label_pos=contour['label_position'],
                                                        linecolor='r', section=sec, type=contour_type,
                                                        side=contour['side'],
                                                        # side_manually_assigned=contour['side_manually_assigned'] if 'side_manually_assigned' in contour else False,
                                                        side_manually_assigned=contour['side_manually_assigned'],
                                                        edit_history=[{'username': contour['creator'], 'timestamp': contour['time_created']}] + contour['edits'],
                                                        contour_id=contour_id)


    def convert_drawings_to_entries(self, timestamp, username):
        """
        Return dict, key=polygon_id, value=contour entry.
        """

        import uuid

        CONTOUR_IS_INTERPOLATED = 1

        contour_entries = {}

        for idx, polygons in self.drawings.iteritems():
            for polygon in polygons:
                if hasattr(polygon, 'contour_id') and polygon.contour_id is not None:
                    polygon_id = polygon.contour_id
                else:
                    polygon_id = str(uuid.uuid4().fields[-1])

                vertices = []
                for c in polygon.vertex_circles:
                    pos = c.scenePos()
                    vertices.append((pos.x(), pos.y()))

                pos = polygon.label_textItem.scenePos()

                contour_entry = {'name': polygon.label,
                            'label_position': (pos.x(), pos.y()),
                           'side': polygon.side,
                           'creator': polygon.edit_history[0]['username'],
                           'time_created': polygon.edit_history[0]['timestamp'],
                            'edits': polygon.edit_history + [{'username':username, 'timestamp':timestamp}],
                            'vertices': vertices,
                            'downsample': self.data_feeder.downsample,
                           'flags': CONTOUR_IS_INTERPOLATED if polygon.type == 'interpolated' else 0,
                            'section': self.data_feeder.sections[idx],
                            # 'position': None,
                            'orientation': self.data_feeder.orientation,
                            'parent_structure': [],
                            'side_manually_assigned': polygon.side_manually_assigned,
                            'id': polygon_id}

        #     contour_entries.append(contour_entry)
                # assert polygon_id not in contour_entries
                contour_entries[polygon_id] = contour_entry

        return contour_entries


    def save_drawings(self, fn_template, timestamp, username):
        return

        import cPickle as pickle

        # Cannot pickle QT objects, so need to extract the data and put in dict.

        # If no labeling is loaded, create a new one
        if not hasattr(self, 'labelings'):
            self.labelings = {'polygons': defaultdict(list)}

            if hasattr(self.data_feeder, 'sections'):
                self.labelings['indexing_scheme'] = 'section'
            else:
                self.labelings['indexing_scheme'] = 'index'

            self.labelings['orientation'] = self.data_feeder.orientation
            self.labelings['downsample'] = self.data_feeder.downsample
            self.labelings['timestamp'] = timestamp
            self.labelings['username'] = username

        # print self.drawings

        for i, polygons in self.drawings.iteritems():

            # Erase the labelings on a loaded section - because we will add those later as they currently appear.
            if hasattr(self.data_feeder, 'sections'):
                # sec = self.data_feeder.sections[i]
                sec = self.data_feeder.sections[i]
                self.labelings['polygons'][sec] = []
            else:
                self.labelings['polygons'][i] = []

            # Add polygons as they currently appear
            for polygon in polygons:
                try:
                    polygon_labeling = {'vertices': []}
                    for c in polygon.vertex_circles:
                        pos = c.scenePos()
                        polygon_labeling['vertices'].append((pos.x(), pos.y()))

                    polygon_labeling['label'] = polygon.label

                    label_pos = polygon.label_textItem.scenePos()
                    polygon_labeling['labelPos'] = (label_pos.x(), label_pos.y())

                    if hasattr(self.data_feeder, 'sections'):
                        # polygon_labeling['section'] = self.data_feeder.sections[i]
                        self.labelings['polygons'][sec].append(polygon_labeling)
                    else:
                        self.labelings['polygons'][i].append(polygon_labeling)

                    polygon_labeling['side'] = None
                    polygon_labeling['type'] = polygon.type

                except Exception as e:
                    print e
                    with open('log.txt', 'w') as f:
                        f.write('ERROR:' + self.id + ' ' + str(i) + '\n')

        fn = fn_template % dict(stack=self.data_feeder.stack, orientation=self.data_feeder.orientation,
                                downsample=self.data_feeder.downsample, username=username, timestamp=timestamp)
        pickle.dump(self.labelings, open(fn, 'w'))
        sys.stderr.write('Labeling saved to %s.\n' % fn)


    def get_label_section_lookup(self):

        label_section_lookup = defaultdict(list)

        for section_index, polygons in self.drawings.iteritems():
            for p in polygons:
                if p.side_manually_assigned:
                    if p.side is None:
                        label = p.label
                    elif p.side == 'S':
                        label = p.label
                    elif p.side == 'L':
                        label = convert_to_left_name(p.label)
                    elif p.side == 'R':
                        label = convert_to_right_name(p.label)
                    else:
                        raise Exception('Side property must be None, L or R.')
                else:
                    label = p.label

                label_section_lookup[label].append(section_index)

        label_section_lookup.default_factory = None
        return label_section_lookup

    # @pyqtSlot()
    # def polygon_completed(self):
    #     polygon = self.sender().parent
    #     self.set_mode('idle')
    #
    #     self.drawings_updated.emit(polygon)

    # @pyqtSlot(object)
    # def polygon_closed(self, polygon):
    #     self.mode = 'idle'

    @pyqtSlot()
    def label_selection_evoked(self):
        self.active_polygon = self.sender().parent
        self.open_label_selection_dialog()

    # @pyqtSlot(object)
    # def label_added(self, text_item):
    #     polygon = self.sender().parent
    #     if polygon.index == self.active_i:
    #         print 'label added.'
            # self.addItem(text_item)

    # @pyqtSlot(QGraphicsEllipseItemModified)
    # def vertex_added(self, circle):
    #     polygon = self.sender().parent
    #     if polygon.index == self.active_i:
    #         pass
    #         # print 'circle added.'
    #         # self.addItem(circle)
    #         # circle.signal_emitter.moved.connect(self.vertex_moved)
    #         # circle.signal_emitter.clicked.connect(self.vertex_clicked)
    #         # circle.signal_emitter.released.connect(self.vertex_released)

    # @pyqtSlot(object)
    # def polygon_press(self, polygon):
    #
    #     print 'polygon pressed'
    #     print self.mode
    #
    #     if self.mode == 'add vertices consecutively':
    #         # if we are adding vertices, do nothing when the click triggers a polygon.
    #         pass
    #     else:
    #         self.active_polygon = polygon
    #         self.polygon_is_moved = False
    #         print 'active polygon selected', self.active_polygon

    # @pyqtSlot()
    # def polygon_release(self):
    #     pass


    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)

        action_newPolygon = myMenu.addAction("New polygon")
        action_deletePolygon = myMenu.addAction("Delete polygon")
        action_insertVertex = myMenu.addAction("Insert vertex")
        action_deleteVertices = myMenu.addAction("Delete vertices")

        myMenu.addSeparator()

        setSide_menu = QMenu("Set hemisphere", myMenu)
        myMenu.addMenu(setSide_menu)
        action_assignS = setSide_menu.addAction('Singular')
        action_assignL = setSide_menu.addAction('Left')
        action_assignR = setSide_menu.addAction('Right')
        action_sides = {action_assignS: 'S', action_assignL: 'L', action_assignR: 'R'}
        if hasattr(self, 'active_polygon') and self.active_polygon.properties['side'] is not None:
            if self.active_polygon.properties['side_manually_assigned']:
                how_str = '(manual)'
            else:
                how_str = '(inferred)'
            if self.active_polygon.properties['side'] == 'L':
                action_assignL.setText('Left ' + how_str)
            elif self.active_polygon.properties['side'] == 'R':
                action_assignR.setText('Right ' + how_str)
            elif self.active_polygon.properties['side'] == 'S':
                action_assignS.setText('Singular '+ how_str)

        action_setLabel = myMenu.addAction("Set label")

        action_confirmPolygon = myMenu.addAction("Confirm this polygon")
        if hasattr(self, 'active_polygon') and self.active_polygon.properties['type'] != 'interpolated':
            action_confirmPolygon.setVisible(False)

        action_reconstruct = myMenu.addAction("Update 3D structure")
        action_showInfo = myMenu.addAction("Show contour information")
        action_showReferences = myMenu.addAction("Show reference resources")

        myMenu.addSeparator()

        resolution_menu = QMenu("Change resolution", myMenu)
        myMenu.addMenu(resolution_menu)
        action_resolutions = {}
        for d in self.data_feeder.supported_downsample_factors:
            action = resolution_menu.addAction(str(d))
            action_resolutions[action] = d

        # action_setUncertain = myMenu.addAction("Set uncertain segment")
        # action_deleteROIDup = myMenu.addAction("Delete vertices in ROI (duplicate)")
        # action_deleteROIMerge = myMenu.addAction("Delete vertices in ROI (merge)")
        # action_deleteBetween = myMenu.addAction("Delete edges between two vertices")
        # action_closePolygon = myMenu.addAction("Close polygon")
        # action_insertVertex = myMenu.addAction("Insert vertex")
        # action_appendVertex = myMenu.addAction("Append vertex")
        # action_connectVertex = myMenu.addAction("Connect vertex")

        # action_showScoremap = myMenu.addAction("Show scoremap")
        # action_showHistology = myMenu.addAction("Show histology")

        # action_doneDrawing = myMenu.addAction("Done drawing")

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))

        # if selected_action == action_changeIndexMode:
        #     if self.indexing_mode == 'section':
        #         self.indexing_mode = 'voxel'
        #     elif self.indexing_mode == 'voxel':
        #         self.indexing_mode = 'section'


        if selected_action == action_deleteVertices:
            self.set_mode('delete vertices')

        elif selected_action in action_sides:
            self.active_polygon.set_side(action_sides[selected_action], side_manually_assigned=True)

        elif selected_action == action_showInfo:

            contour_info_text = "Abbreviation: %(name)s\n" % {'name': self.active_polygon.label}
            contour_info_text += "Fullname: %(fullname)s\n" % {'fullname': self.gui.structure_names[self.active_polygon.label]}

            if self.active_polygon.properties['side'] is None:
                side_string = ''
            else:
                if self.active_polygon.properties['side'] == 'S':
                    side_string = 'singular'
                elif self.active_polygon.properties['side'] == 'L':
                    side_string = 'left'
                elif self.active_polygon.properties['side'] == 'R':
                    side_string = 'right'
                else:
                    raise Exception('Side property must be one of S, L or R.')

                if self.active_polygon.properties['side_manually_assigned'] is not None:
                    if self.active_polygon.properties['side_manually_assigned']:
                        side_string += ' (manual)'
                    else:
                        side_string += ' (inferred)'

            contour_info_text += "Side: %(side)s\n" % {'side': side_string}

            first_edit = self.active_polygon.properties['edit_history'][0]
            contour_info_text += "Created by %(creator)s at %(timestamp)s\n" % \
            {'creator': first_edit['username'],
            'timestamp':  datetime.strftime(datetime.strptime(first_edit['timestamp'], "%m%d%Y%H%M%S"), '%Y/%m/%d %H:%M')
            }

            # editors = set(x['username'] for x in self.active_polygon.edit_history[1:])
            # if len(editors) > 0:
            #     contour_info_text += "Edited by %(editors)s\n" % \
            #     {'editors': ' '.join(set(x['username'] for x in self.active_polygon.edit_history[1:]))}

            print self.active_polygon.properties['edit_history']

            last_edit = self.active_polygon.properties['edit_history'][-1]
            contour_info_text += "Last edited by %(editor)s at %(timestamp)s\n" % \
            {'editor': last_edit['username'],
            'timestamp':  datetime.strftime(datetime.strptime(last_edit['timestamp'], "%m%d%Y%H%M%S"), '%Y/%m/%d %H:%M')
            }

            contour_info_text += "Type: %(type)s\n" % {'type': self.active_polygon.properties['type']}

            QMessageBox.information(self.gview, "Information", contour_info_text)

        elif selected_action == action_showReferences:

            reference_text = ''
            # for resource_name, resource_url in reference_resources[self.active_polygon.label].iteritems():
            for resource_name in ['BrainInfo', 'PubMed', 'Allen Reference Atlas (Saggittal)', 'Allen Reference Atlas (Coronal)']:
                resource_url = reference_resources[self.active_polygon.properties['label']][resource_name]
                reference_text += "<a href=\"%(resource_url)s\">%(resource_name)s</a><br>" % dict(resource_url=resource_url, resource_name=resource_name)

            msgBox = QMessageBox(self.gview)
            msgBox.setWindowTitle("Reference Resources")
            msgBox.setTextFormat(Qt.RichText)
            msgBox.setText(reference_text)
            msgBox.exec_()

        elif selected_action == action_confirmPolygon:
            self.active_polygon.set_type(None)

        elif selected_action == action_deletePolygon:
            self.drawings[self.active_i].remove(self.active_polygon)
            self.removeItem(self.active_polygon)

        elif selected_action == action_reconstruct:
            self.update_structure_volume_requested.emit(self.active_polygon)

        elif selected_action in action_resolutions:
            selected_downsample_factor = action_resolutions[selected_action]
            self.set_downsample_factor(selected_downsample_factor)

        elif selected_action == action_newPolygon:
            # self.disable_elements()
            self.close_curr_polygon = False
            self.active_polygon = self.add_polygon(QPainterPath(), color='r', index=self.active_i)
            self.active_polygon.add_edit(editor=self.gui.get_username())
            self.active_polygon.set_type(None)
            self.active_polygon.set_side(side=None, side_manually_assigned=False)

            # self.set_mode(Mode.ADDING_VERTICES_CONSECUTIVELY)
            self.set_mode('add vertices consecutively')

        elif selected_action == action_insertVertex:
            self.set_mode('add vertices randomly')

        # elif selected_action == action_deletePolygon:
        #     self.remove_polygon(self.selected_polygon)
        #
        elif selected_action == action_setLabel:
            self.open_label_selection_dialog()

    # @pyqtSlot()
    # def vertex_clicked(self):
    #     # pass
    #     circle = self.sender().parent
    #     print 'vertex clicked:', circle


    # @pyqtSlot()
    # def vertex_released(self):
    #     # print self.sender().parent, 'released'
    #
    #     clicked_vertex = self.sender().parent
    #
    #     if self.mode == 'moving vertex' and self.vertex_is_moved:
    #         # self.history_allSections[self.active_section].append({'type': 'drag_vertex', 'polygon': self.active_polygon, 'vertex': clicked_vertex, \
    #         #                      'mouse_moved': (clicked_vertex.release_scene_x - clicked_vertex.press_scene_x, \
    #         #                          clicked_vertex.release_scene_y - clicked_vertex.press_scene_y), \
    #         #                      'label': self.accepted_proposals_allSections[self.active_section][self.active_polygon]['label']})
    #
    #         self.vertex_is_moved = False
    #         # self.print_history()

    def update_cross(self, cross_x_lossless, cross_y_lossless, cross_z_lossless):

        print self.id, ': cross_lossless', cross_x_lossless, cross_y_lossless, cross_z_lossless

        # self.hline.setVisible(True)
        # self.vline.setVisible(True)

        self.cross_x_lossless = cross_x_lossless
        self.cross_y_lossless = cross_y_lossless
        self.cross_z_lossless = cross_z_lossless

        downsample = self.data_feeder.downsample
        cross_x_ds = cross_x_lossless / downsample
        cross_y_ds = cross_y_lossless / downsample
        cross_z_ds = cross_z_lossless / downsample

        if self.data_feeder.orientation == 'sagittal':

            self.hline.setLine(0, cross_y_ds, self.data_feeder.x_dim-1, cross_y_ds)
            self.vline.setLine(cross_x_ds, 0, cross_x_ds, self.data_feeder.y_dim-1)

            if hasattr(self.data_feeder, 'sections'):
                sec = self.convert_z_to_section(z=cross_z_ds, downsample=downsample)
                print 'cross_z', cross_z_ds, 'sec', sec, 'reverse z', self.convert_section_to_z(sec=sec, downsample=downsample)
                self.set_active_section(sec, update_crossline=False)

        elif self.data_feeder.orientation == 'coronal':

            self.hline.setLine(0, cross_y_ds, self.data_feeder.z_dim-1, cross_y_ds)
            self.vline.setLine(self.data_feeder.z_dim-1-cross_z_ds, 0, self.data_feeder.z_dim-1-cross_z_ds, self.data_feeder.y_dim-1)

            self.set_active_i(cross_x_ds, update_crossline=False)

        elif self.data_feeder.orientation == 'horizontal':

            self.hline.setLine(0, self.data_feeder.z_dim-1-cross_z_ds, self.data_feeder.x_dim-1, self.data_feeder.z_dim-1-cross_z_ds)
            self.vline.setLine(cross_x_ds, 0, cross_x_ds, self.data_feeder.z_dim-1)

            self.set_active_i(cross_y_ds, update_crossline=False)


    def eventFilter(self, obj, event):
        # print obj.metaObject().className(), event.type()
        # http://doc.qt.io/qt-4.8/qevent.html#Type-enum


        if event.type() == QEvent.KeyPress:
            key = event.key()

            if key == Qt.Key_Escape:
                self.set_mode('idle')
                return True

            elif (key == Qt.Key_Enter or key == Qt.Key_Return) and self.mode == 'add vertices consecutively': # CLose polygon
                first_circ = self.active_polygon.vertex_circles[0]
                first_circ.signal_emitter.press.emit(first_circ)
                return False
            elif key == Qt.Key_V: # Toggle all vertex circles
                for i, polygons in self.drawings.iteritems():
                    for p in polygons:
                        for c in p.vertex_circles:
                            if c.isVisible():
                                c.setVisible(False)
                            else:
                                c.setVisible(True)
            elif key == Qt.Key_C: # Toggle all contours
                for i, polygons in self.drawings.iteritems():
                    for p in polygons:
                        if p.isVisible():
                            p.setVisible(False)
                        else:
                            p.setVisible(True)

            elif key == Qt.Key_Control:
                if not event.isAutoRepeat():
                    # for polygon in self.drawings[self.active_i]:
                    #     polygon.setFlag(QGraphicsItem.ItemIsMovable, True)
                    self.active_polygon.setFlag(QGraphicsItem.ItemIsMovable, True)

        elif event.type() == QEvent.KeyRelease:
            key = event.key()

            if key == Qt.Key_Control:
                if not event.isAutoRepeat():
                    # for polygon in self.drawings[self.active_i]:
                    #     polygon.setFlag(QGraphicsItem.ItemIsMovable, False)
                    self.active_polygon.setFlag(QGraphicsItem.ItemIsMovable, False)

        elif event.type() == QEvent.Wheel:
            # eat wheel event from gview viewport. default behavior is to trigger down scroll

            out_factor = .9
            in_factor = 1. / out_factor

            if event.delta() < 0: # negative means towards user
                self.gview.scale(out_factor, out_factor)
            else:
                self.gview.scale(in_factor, in_factor)

            # gview_x = event.x()
            # gview_y = event.y()
            # delta = event.delta()
            # print delta
            # # self.zoom_scene(gscene_x, gscene_y, delta)
            # self.zoom_scene(gview_x, gview_y, delta)

            return True

        # if event.type() == QEvent.GraphicsSceneWheel:
        #     return True
        #     gscene_x = event.scenePos().x()
        #     gscene_y = event.scenePos().y()
        #     delta = event.delta()
        #     self.zoom_scene(gscene_x, gscene_y, delta)
        #     print 2
        #     return True

        if event.type() == QEvent.GraphicsSceneMousePress:

            self.gui.active_gscene = self

            pos = event.scenePos()
            gscene_x = pos.x()
            gscene_y = pos.y()

            if event.button() == Qt.RightButton:
                obj.mousePressEvent(event)

            if self.mode == 'idle':
                # pass the event down
                obj.mousePressEvent(event)

                self.press_screen_x = gscene_x
                self.press_screen_y = gscene_y
                print self.press_screen_x, self.press_screen_y
                self.pressed = True
                return True

            elif self.mode == 'crossline':

                downsample = self.data_feeder.downsample

                gscene_y_lossless = gscene_y * downsample
                gscene_x_lossless = gscene_x * downsample

                if hasattr(self.data_feeder, 'sections'):
                    z0, z1 = self.convert_section_to_z(sec=self.active_section, downsample=1) # Note that the returned result is a pair of z limits.
                    gscene_z_lossless = .5 * z0 + .5 * z1
                else:
                    gscene_z_lossless = self.active_i * downsample

                if self.data_feeder.orientation == 'sagittal':
                    cross_x_lossless = gscene_x_lossless
                    cross_y_lossless = gscene_y_lossless
                    cross_z_lossless = gscene_z_lossless

                elif self.data_feeder.orientation == 'coronal':
                    cross_z_lossless = self.data_feeder.z_dim * downsample - 1 - gscene_x_lossless
                    cross_y_lossless = gscene_y_lossless
                    cross_x_lossless = gscene_z_lossless

                elif self.data_feeder.orientation == 'horizontal':
                    cross_x_lossless = gscene_x_lossless
                    cross_z_lossless = self.data_feeder.z_dim * downsample - 1 - gscene_y_lossless
                    cross_y_lossless = gscene_z_lossless

                print self.id, ': emit', cross_x_lossless, cross_y_lossless, cross_z_lossless
                self.crossline_updated.emit(cross_x_lossless, cross_y_lossless, cross_z_lossless, self.id)
                return True


            elif self.mode == 'add vertices consecutively':
                # if in add vertices mode, left mouse press means:
                # - closing a polygon, or
                # - adding a vertex

                if event.button() == Qt.LeftButton:

                    obj.mousePressEvent(event)

                    # if hasattr(self, 'close_curr_polygon') and self.close_curr_polygon:
                    #
                    #     print 'close curr polygon UNSET'
                    #     self.close_curr_polygon = False
                    #     self.close_polygon()
                    #
                    #     # if 'label' not in self.drawings[self.active_section][self.active_polygon]:
                    #     #     self.open_label_selection_dialog()
                    #
                    #     self.set_mode('idle')
                    #
                    # else:

                        # polygon_goto(self.active_polygon, gscene_x, gscene_y)
                        # self.add_vertex_to_polygon(self.active_polygon, gscene_x, gscene_y)
                        # self.print_history()
                    if not self.active_polygon.closed:
                        self.active_polygon.add_vertex(gscene_x, gscene_y)

                    return True

            elif self.mode == 'add vertices randomly':
                if event.button() == Qt.LeftButton:
                    obj.mousePressEvent(event)

                    assert self.active_polygon.closed, 'Insertion is not allowed if polygon is not closed.'
                    new_index = find_vertex_insert_position(self.active_polygon, gscene_x, gscene_y)
                    self.active_polygon.add_vertex(gscene_x, gscene_y, new_index)

                    return True

            elif self.mode == 'delete vertices':
                items_in_rubberband = self.analyze_rubberband_selection()

                for polygon, vertex_indices in items_in_rubberband.iteritems():
                    polygon.delete_vertices(vertex_indices)
                    # if self.mode == Mode.DELETE_ROI_DUPLICATE:
                    #     self.delete_vertices(polygon, vertex_indices)
                    # elif self.mode == Mode.DELETE_ROI_MERGE:
                    #     self.delete_vertices(polygon, vertex_indices, merge=True)

                # self.set_mode('idle')
                # self.set_mode(Mode.IDLE)

        return False
