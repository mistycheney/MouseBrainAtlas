import sys, os
from collections import defaultdict
from datetime import datetime

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from qimage2ndarray import recarray_view, array2qimage

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from data_manager import DataManager
from metadata import *
from annotation_utilities import *
from registration_utilities import *
from gui_utilities import *

from custom_widgets import AutoCompleteInputDialog
from DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene

CROSSLINE_PEN_WIDTH = 2
CROSSLINE_RED_PEN = QPen(Qt.red)
CROSSLINE_RED_PEN.setWidth(CROSSLINE_PEN_WIDTH)

MARKER_COLOR_CHAR = 'w'

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

    crossline_updated = pyqtSignal(int, int, int, str)
    structure_volume_updated =  pyqtSignal(str, str, bool, bool)

    def __init__(self, id, gui=None, gview=None, parent=None):
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).__init__(id=id, gview=gview, parent=parent)
        self.gui = gui
        self.showing_which = 'histology'
        self.per_channel_pixmap_cached = {}
        self.per_channel_pixmap_cache_section = None

        self.hline = QGraphicsLineItem()
        self.hline.setPen(CROSSLINE_RED_PEN)
        self.vline = QGraphicsLineItem()
        self.vline.setPen(CROSSLINE_RED_PEN)
        self.addItem(self.hline)
        self.addItem(self.vline)
        self.hline.setVisible(False)
        self.vline.setVisible(False)

        self.uncertainty_lines = {}
        self.structure_onscreen_messages = {}
        self.default_name = None

    def set_mode(self, mode):
        """
        Extend inherited method by:
        - showing or hiding two cross-lines.
        """

        if mode == 'add vertices once':
            print "\nPress N and then click to put a marker. Label of the first marker is set as default for subsequent markers. Press B to clear default label.\n"

        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).set_mode(mode)
        if mode == 'crossline':
            self.hline.setVisible(True)
            self.vline.setVisible(True)
        # elif mode == 'rotate3d':
        #     pass
        # elif mode == 'shift3d':
        #     pass
        elif mode == 'idle':
            self.hline.setVisible(False)
            self.vline.setVisible(False)

    def set_structure_volumes(self, structure_volumes):
        """
        Args:
            structure_volumes (dict): {structure name: (volume, bbox)}.
            The volume dimension is the bounding box of the structure. Thumbnail resolution.
            Bbox coordinates are relative to the cropped volume.
        """

        self.structure_volumes = structure_volumes

    def set_structure_volumes_downscale_factor(self, downscale):
        sys.stderr.write('Set structure volumes downscale to %d\n' % downscale)
        self.structure_volumes_downscale_factor = downscale

    def update_drawings_from_structure_volume(self, name_u, side):
        """
        Update drawings based on `self.structure_volumes`.

        Args:
            name_u (str): structure name, unsided
            side (str): L, R or S
        """

        print "%s: Updating drawings based on structure volume of %s, %s" % (self.id, name_u, side)

        volume = self.structure_volumes[(name_u, side)]['volume_in_bbox']
        bbox = self.structure_volumes[(name_u, side)]['bbox']
        # for x in range(volume.shape[1]):
        #     imsave('/tmp/vol_%d.png' % x, (volume[:, x, :]*255).astype(np.uint8))

        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        print 'volume', volume.shape, xmin, xmax, ymin, ymax, zmin, zmax

        volume_downsample_factor = self.structure_volumes_downscale_factor
        bbox_lossless = np.array(bbox) * volume_downsample_factor

        data_vol_downsample_ratio = float(self.data_feeder.downsample) / volume_downsample_factor

        if data_vol_downsample_ratio > 1:
            volume_data_resol = volume[::data_vol_downsample_ratio, ::data_vol_downsample_ratio, ::data_vol_downsample_ratio]
            xmin_ds, xmax_ds, ymin_ds, ymax_ds, zmin_ds, zmax_ds = np.array(bbox_lossless) / self.data_feeder.downsample
            print 'volume at data resol', volume_data_resol.shape, xmin_ds, xmax_ds, ymin_ds, ymax_ds, zmin_ds, zmax_ds

        if hasattr(self.data_feeder, 'sections'):

            assert self.data_feeder.orientation == 'sagittal', "Current implementation only considers sagittal sections."

            print "Removing all unconfirmed polygons..."

            matched_unconfirmed_polygons_to_remove = {i: [p for p in self.drawings[i] \
            if p.properties['label'] == name_u and p.properties['side'] == side and \
            p.properties['type'] != 'confirmed']
            for i in range(len(self.data_feeder.sections))}

            for i in range(len(self.data_feeder.sections)):
                for p in matched_unconfirmed_polygons_to_remove[i]:
                    self.drawings[i].remove(p)
                    if i == self.active_i:
                        self.removeItem(p)

            sections_used = []
            positions_rel_vol_resol = []
            for sec in self.data_feeder.sections:
                pos_gl_vol_resol = np.mean(self.convert_section_to_z(sec=sec, downsample=volume_downsample_factor))
                pos_rel_vol_resol = int(np.round(pos_gl_vol_resol - zmin))
                if pos_rel_vol_resol >= 0 and pos_rel_vol_resol < volume.shape[2]:
                    positions_rel_vol_resol.append(pos_rel_vol_resol)
                    sections_used.append(sec)

            if self.data_feeder.downsample == 1:
                gscene_pts_rel_vol_resol_allpos = find_contour_points_3d(volume, along_direction='z', sample_every=20, positions=positions_rel_vol_resol)
            else:
                gscene_pts_rel_vol_resol_allpos = find_contour_points_3d(volume, along_direction='z', sample_every=1, positions=positions_rel_vol_resol)
            m = dict(zip(positions_rel_vol_resol, sections_used))
            gscene_pts_rel_vol_resol_allsec = {m[pos]: pts for pos, pts in gscene_pts_rel_vol_resol_allpos.iteritems()}

            for sec, gscene_pts_rel_vol_resol in gscene_pts_rel_vol_resol_allsec.iteritems():

                # if this section already has a confirmed contour, do not add a new one.
                if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                p.properties['type'] == 'confirmed' for p in self.drawings[self.data_feeder.sections.index(sec)]]):
                    continue

                gscene_xs_gl_vol_resol = gscene_pts_rel_vol_resol[:,0] + xmin
                gscene_ys_gl_vol_resol = gscene_pts_rel_vol_resol[:,1] + ymin
                gscene_pts_gl_vol_resol = np.c_[gscene_xs_gl_vol_resol, gscene_ys_gl_vol_resol]
                gscene_pts_gl_data_resol = gscene_pts_gl_vol_resol / data_vol_downsample_ratio
                # t = time.time()
                self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_gl_data_resol), label=name_u,
                                                    linecolor='r', vertex_radius=8, linewidth=5, section=sec,
                                                    type='intersected',
                                                    side=side,
                                                    side_manually_assigned=False)
                # sys.stderr.write("Add polygon and vertices: %.2f seconds.\n" % (time.time()-t))
        else:

            if self.data_feeder.orientation == 'sagittal':
                pos_start_ds = 0
                pos_end_ds = self.data_feeder.z_dim - 1
            elif self.data_feeder.orientation == 'coronal':
                pos_start_ds = 0
                pos_end_ds = self.data_feeder.x_dim - 1
            elif self.data_feeder.orientation == 'horizontal':
                pos_start_ds = 0
                pos_end_ds = self.data_feeder.y_dim - 1

            print "Removing all unconfirmed polygons..."
            for pos_ds in range(pos_start_ds, pos_end_ds+1):
                matched_unconfirmed_polygons_to_remove = [p for p in self.drawings[pos_ds] \
                if p.properties['label'] == name_u and p.properties['side'] == side and \
                p.properties['type'] != 'confirmed']
                for p in matched_unconfirmed_polygons_to_remove:
                    self.drawings[pos_ds].remove(p)
                    if pos_ds == self.active_i:
                        self.removeItem(p)

            # volume_data_resol is the structure in bbox.
            if self.data_feeder.orientation == 'coronal':
                gscene_pts_allpos = find_contour_points_3d(volume_data_resol, along_direction='x', sample_every=1)
                for pos, gscene_pts in gscene_pts_allpos.iteritems():

                    # if this position already has a confirmed contour, do not add a new one.
                    if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                    p.properties['type'] == 'confirmed' for p in self.drawings[pos]]):
                        continue

                    gscene_xs = self.data_feeder.z_dim - 1 - (gscene_pts[:,0] + zmin_ds)
                    gscene_ys = gscene_pts[:,1] + ymin_ds
                    gscene_pts = np.c_[gscene_xs, gscene_ys]
                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts), label=name_u,
                                                    linecolor='g', vertex_radius=1, linewidth=2, index=pos+xmin_ds,
                                                    type='intersected',
                                                    side=side,
                                                    side_manually_assigned=False)
            elif self.data_feeder.orientation == 'horizontal':
                gscene_pts_allpos = find_contour_points_3d(volume_data_resol, along_direction='y', sample_every=1)
                for pos, gscene_pts in gscene_pts_allpos.iteritems():

                    # if this position already has a confirmed contour, do not add a new one.
                    if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                    p.properties['type'] == 'confirmed' for p in self.drawings[pos]]):
                        continue

                    gscene_xs = gscene_pts[:,1] + xmin_ds
                    gscene_ys = self.data_feeder.z_dim - 1 - (gscene_pts[:,0] + zmin_ds)
                    gscene_pts = np.c_[gscene_xs, gscene_ys]
                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts), label=name_u,
                                                        linecolor='g', vertex_radius=1, linewidth=2, index=pos+ymin_ds,
                                                        type='intersected',
                                                        side=side,
                                                        side_manually_assigned=False)
            elif self.data_feeder.orientation == 'sagittal':
                gscene_pts_allpos = find_contour_points_3d(volume_data_resol, along_direction='z', sample_every=1)
                for pos, gscene_pts in gscene_pts_allpos.iteritems():

                    # if this position already has a confirmed contour, do not add a new one.
                    if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                    p.properties['type'] == 'confirmed' for p in self.drawings[pos]]):
                        continue

                    gscene_xs = gscene_pts[:,0] + xmin_ds
                    gscene_ys = gscene_pts[:,1] + ymin_ds
                    gscene_pts = np.c_[gscene_xs, gscene_ys]
                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts), label=name_u,
                                                        linecolor='g', vertex_radius=1, linewidth=2, index=pos+zmin_ds,
                                                        type='intersected',
                                                        side=side,
                                                        side_manually_assigned=False)


    def update_image(self, i=None, sec=None):
        i, sec = self.get_requested_index_and_section(i=i, sec=sec)

        if self.showing_which == 'histology':
            image = self.data_feeder.retrieve_i(i=i)
            histology_pixmap = QPixmap.fromImage(image)
            self.pixmapItem.setPixmap(histology_pixmap)
            self.pixmapItem.setVisible(True)
        elif self.showing_which == 'scoremap':
            assert self.active_polygon is not None, 'Must have an active polygon first.'
            name_u = self.active_polygon.properties['label']
            scoremap_viz_fp = DataManager.get_scoremap_viz_filepath(stack=self.gui.stack, downscale=8, section=sec, structure=name_u, detector_id=1, prep_id=2)
            download_from_s3(scoremap_viz_fp)
            w, h = metadata_cache['image_shapes'][self.gui.stack]
            scoremap_pixmap = QPixmap(scoremap_viz_fp).scaled(w, h)
            self.pixmapItem.setPixmap(scoremap_pixmap)
            self.pixmapItem.setVisible(True)

        elif self.showing_which == 'blue_only' or self.showing_which == 'red_only' or self.showing_which == 'green_only' :
            if self.per_channel_pixmap_cache_section != sec:
                # qimage = self.data_feeder.retrieve_i(i=i)
                # t = time.time()
                # img = recarray_view(qimage)
                # print 'recarray_view', time.time() - t
                # img_shape = img["b"].shape
                # img_dtype = img["b"].dtype
                # img_blue = np.dstack([np.zeros(img_shape, img_dtype), np.zeros(img_shape, img_dtype), img["b"]])
                # img_red = np.dstack([img["r"], np.zeros(img_shape, img_dtype), np.zeros(img_shape, img_dtype)])
                # img_green = np.dstack([np.zeros(img_shape, img_dtype), img["g"], np.zeros(img_shape, img_dtype)])
                # t = time.time()
                # qimage_blue = array2qimage(img_blue)
                # qimage_red = array2qimage(img_red)
                # qimage_green = array2qimage(img_green)
                # print 'array2qimage', time.time() - t

                t = time.time()
                blue_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='contrastStretchedBlue', ext='jpg')
                qimage_blue = QImage(blue_fp)
                # green_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='contrastStretchedGreen', ext='jpg')
                # qimage_green = QImage(green_fp)
                # red_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='contrastStretchedRed', ext='jpg')
                # qimage_red = QImage(red_fp)
                print 'read images', time.time() - t # 9s for first read, 4s for subsequent

                if 'b' in self.per_channel_pixmap_cached:
                    del self.per_channel_pixmap_cached['b']
                self.per_channel_pixmap_cached['b'] = QPixmap.fromImage(qimage_blue)
                # self.per_channel_pixmap_cached['g'] = QPixmap.fromImage(qimage_green)
                # self.per_channel_pixmap_cached['r'] = QPixmap.fromImage(qimage_red)
                self.per_channel_pixmap_cache_section = sec

            if self.showing_which == 'blue_only':
                self.pixmapItem.setPixmap(self.per_channel_pixmap_cached['b'])
            # elif self.showing_which == 'red_only':
            #     self.pixmapItem.setPixmap(self.per_channel_pixmap_cached['r'])
            # elif self.showing_which == 'green_only':
            #     self.pixmapItem.setPixmap(self.per_channel_pixmap_cached['g'])

            self.pixmapItem.setVisible(True)

        else:
            raise Exception("Show option %s is not recognized." % self.showing_which)

    def infer_side(self):

        label_section_lookup = self.get_label_section_lookup()

        structure_ranges = get_landmark_range_limits_v2(stack=self.data_feeder.stack, label_section_lookup=label_section_lookup)

        print 'structure_ranges', structure_ranges

        for section_index, polygons in self.drawings.iteritems():
            for p in polygons:
                if p.properties['label'] in structure_ranges:
                    assert p.properties['label'] in singular_structures, 'Label %s is in structure_ranges, but it is not singular.' % p.properties['label']
                    if section_index >= structure_ranges[p.properties['label']][0] and section_index <= structure_ranges[p.properties['label']][1]:
                        if p.properties['side'] is None or not p.properties['side_manually_assigned']:
                            p.set_properties('side', 'S')
                            p.set_properties('side_manually_assigned', False)
                    else:
                        raise Exception('Polygon is on a section not in structure_range.')
                else:
                    lname = convert_to_left_name(p.properties['label'])
                    if lname in structure_ranges:
                        if section_index >= structure_ranges[lname][0] and section_index <= structure_ranges[lname][1]:
                            if p.properties['side'] is None or not p.properties['side_manually_assigned']:
                                p.set_properties('side', 'L')
                                p.set_properties('side_manually_assigned', False)
                                # sys.stderr.write('%d, %d %s set to L\n' % (section_index, self.data_feeder.sections[section_index], p.properties['label']))

                    rname = convert_to_right_name(p.properties['label'])
                    if rname in structure_ranges:
                        if section_index >= structure_ranges[rname][0] and section_index <= structure_ranges[rname][1]:
                            if p.properties['side'] is None or not p.properties['side_manually_assigned']:
                                p.set_properties('side', 'R')
                                p.set_properties('side_manually_assigned', False)
                                # sys.stderr.write('%d, %d %s set to R\n' % (section_index, self.data_feeder.sections[section_index], p.properties['label']))


    def set_conversion_func_section_to_z(self, func):
        """
        Set the conversion function that converts section index to voxel position.
        """
        self.convert_section_to_z = func

    def set_conversion_func_z_to_section(self, func):
        """
        Set the conversion function that converts voxel position to section index.
        """
        self.convert_z_to_section = func

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
        # self.active_polygon.set_label(abbr)
        self.active_polygon.set_properties('label', abbr)
        print self.active_polygon.properties

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


    def load_drawings(self, contours, append=False, vertex_color=None):
        """
        Load annotation contours and place drawings.

        Args:
            vertex_color (str or (3,)-array of 255-based int):

        """

        if not append:
            self.drawings = defaultdict(list)

        endorser_contour_lookup = defaultdict(set)
        for cnt_id, contour in contours.iterrows():
            for editor in set([edit['username'] for edit in contour['edits']]):
                endorser_contour_lookup[editor].add(cnt_id)
            endorser_contour_lookup[contour['creator']].add(cnt_id)

        grouped = contours.groupby('section')

        for sec, group in grouped:
            # assert sec in metadata_cache['valid_sections'][self.gui.stack], "Section %d is labeled but the section is not valid." % sec
            if sec not in metadata_cache['valid_sections'][self.gui.stack]:
                sys.stderr.write( "Section %d is labeled but the section is not valid.\n" % sec)
                continue

            for contour_id, contour in group.iterrows():
                vertices = contour['vertices']
                if 'type' in contours and contour['type'] is not None:
                    contour_type = contour['type']
                else:
                    contour_type = None

                if 'class' in contours and contour['class'] is not None:
                    contour_class = contour['class']
                else:
                    contour_class = 'contour'
                    # contour_class = None

                self.add_polygon_with_circles_and_label(path=vertices_to_path(vertices),
                                                        label=contour['name'], label_pos=contour['label_position'],
                                                        linecolor='r', vertex_color=vertex_color,
                                                        section=sec, type=contour_type,
                                                        side=contour['side'],
                                                        side_manually_assigned=contour['side_manually_assigned'],
                                                        edits=[{'username': contour['creator'], 'timestamp': contour['time_created']}] + contour['edits'],
                                                        contour_id=contour_id,
                                                        category=contour_class)

    def convert_drawings_to_entries(self, timestamp, username, classes=['contour']):
        """
        Args:
            classes (list of str): list of classes to gather. Default is contour.

        Returns:
            dict: {polygon_id: contour information entry}
        """

        import uuid
        # CONTOUR_IS_INTERPOLATED = 1
        contour_entries = {}
        for idx, polygons in self.drawings.iteritems():
            for polygon in polygons:
                if 'class' not in polygon.properties or ('class' in polygon.properties and polygon.properties['class'] not in classes):
                    # raise Exception("polygon has no class: %d, %s" % (self.data_feeder.sections[idx], polygon.properties['label']))
                    sys.stderr.write("Polygon has no class: %d, %s. Skip." % (self.data_feeder.sections[idx], polygon.properties['label']))
                    continue

                if hasattr(polygon, 'contour_id') and polygon.contour_id is not None:
                    polygon_id = polygon.contour_id
                else:
                    polygon_id = str(uuid.uuid4().fields[-1])

                vertices = []
                for c in polygon.vertex_circles:
                    pos = c.scenePos()
                    vertices.append((pos.x(), pos.y()))

                if len(vertices) == 0:
                    sys.stderr.write("Polygon has no vertices.\n")
                    continue

                if polygon.properties['class'] == 'neuron':
                    # labeled neuron markers
                    contour_entry = {'name': polygon.properties['label'],
                            'label_position': None,
                               'side': polygon.properties['side'],
                               'creator': polygon.properties['edits'][0]['username'],
                               'time_created': polygon.properties['edits'][0]['timestamp'],
                                'edits': polygon.properties['edits'],
                                'vertices': vertices,
                                'downsample': self.data_feeder.downsample,
                                'type': None,
                                'orientation': self.data_feeder.orientation,
                                'parent_structure': [],
                                'side_manually_assigned': polygon.properties['side_manually_assigned'],
                                'id': polygon_id,
                                'class': polygon.properties['class']}
                    assert hasattr(self.data_feeder, 'sections')
                    contour_entry['section'] = self.data_feeder.sections[idx]
                else:
                    # structure boundaries
                    label_pos = polygon.properties['label_textItem'].scenePos()
                    contour_entry = {'name': polygon.properties['label'],
                                'label_position': (label_pos.x(), label_pos.y()),
                               'side': polygon.properties['side'],
                               'creator': polygon.properties['edits'][0]['username'],
                               'time_created': polygon.properties['edits'][0]['timestamp'],
                                'edits': polygon.properties['edits'] + [{'username':username, 'timestamp':timestamp}],
                                'vertices': vertices,
                                'downsample': self.data_feeder.downsample,
                            #    'flags': 0 if polygon.properties['type'] == 'confirmed' else 1,
                                'type': polygon.properties['type'],
                                'orientation': self.data_feeder.orientation,
                                'parent_structure': [],
                                'side_manually_assigned': polygon.properties['side_manually_assigned'],
                                'id': polygon_id,
                                'class': polygon.properties['class']}

                    if hasattr(self.data_feeder, 'sections'):
                        contour_entry['section'] = self.data_feeder.sections[idx]
                    else:
                        contour_entry['voxel_position'] = idx

                contour_entries[polygon_id] = contour_entry

        return contour_entries

    def get_label_section_lookup(self):

        label_section_lookup = defaultdict(list)

        for section_index, polygons in self.drawings.iteritems():
            for p in polygons:
                if p.properties['side_manually_assigned']:
                    if p.side is None:
                        label = p.properties['label']
                    elif p.side == 'S':
                        label = p.properties['label']
                    elif p.side == 'L':
                        label = convert_to_left_name(p.properties['label'])
                    elif p.side == 'R':
                        label = convert_to_right_name(p.properties['label'])
                    else:
                        raise Exception('Side property must be None, L or R.')
                else:
                    label = p.properties['label']

                label_section_lookup[label].append(section_index)

        label_section_lookup.default_factory = None
        return label_section_lookup

    @pyqtSlot()
    def polygon_completed_callback(self):
        polygon = self.sender().parent
        if self.mode == 'add vertices once':
            if self.default_name is None:
                self.open_label_selection_dialog()
                if 'label' in self.active_polygon.properties and self.active_polygon.properties['label'] is not None:
                    self.default_name = self.active_polygon.properties['label']
                    print '\nself.default_name =', self.default_name, '\n'
            else:
                self.active_polygon.set_properties('label', self.default_name)
        else:
            self.open_label_selection_dialog()
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).polygon_completed_callback()

    @pyqtSlot(object)
    def polygon_pressed_callback(self, polygon):
        # polygon = self.sender().parent
        if self.mode == 'remove marker':
            self.drawings[self.active_i].remove(polygon)
            self.removeItem(polygon)
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).polygon_pressed_callback(polygon)

    # @pyqtSlot(object)
    # def polygon_closed(self, polygon):
    #     self.mode = 'idle'

    # @pyqtSlot()
    # def label_selection_evoked(self):
    #     self.active_polygon = self.sender().parent
    #     self.open_label_selection_dialog()

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
        action_newMarker = myMenu.addAction("New marker")
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
        if hasattr(self, 'active_polygon') and self.active_polygon.properties['type'] == 'confirmed':
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
            self.active_polygon.set_properties('side', action_sides[selected_action])
            self.active_polygon.set_properties('side_manually_assigned', True)
            # self.active_polygon.set_side(action_sides[selected_action], side_manually_assigned=True)

        elif selected_action == action_showInfo:
            self.show_information_box()

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
            # self.active_polygon.set_type(None)
            self.active_polygon.set_properties('type', 'confirmed')

        elif selected_action == action_deletePolygon:
            self.drawings[self.active_i].remove(self.active_polygon)
            self.removeItem(self.active_polygon)

        elif selected_action == action_reconstruct:
            assert 'side' in self.active_polygon.properties and self.active_polygon.properties['side'] is not None, 'Must specify side first.'
            self.structure_volume_updated.emit(self.active_polygon.properties['label'], self.active_polygon.properties['side'], True, True)

        elif selected_action in action_resolutions:
            selected_downsample_factor = action_resolutions[selected_action]
            self.set_downsample_factor(selected_downsample_factor)

        elif selected_action == action_newPolygon:
            self.set_mode('add vertices consecutively')
            self.start_new_polygon(init_properties={'class': 'contour'})

        elif selected_action == action_insertVertex:
            self.set_mode('add vertices randomly')

        elif selected_action == action_setLabel:
            self.open_label_selection_dialog()

        elif selected_action == action_newMarker:
            self.set_mode('add vertices once')
            self.start_new_polygon(init_properties={'class': 'neuron'}, color=MARKER_COLOR)

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

    def start_new_polygon(self, init_properties=None, color='r'):
        """
        Args:
            init_properties (dict): {property_name: property_value}
        """
        # self.disable_elements()
        # self.close_curr_polygon = False
        self.active_polygon = self.add_polygon(QPainterPath(), color=color, index=self.active_i)

        self.active_polygon.set_properties('orientation', self.data_feeder.orientation)

        self.active_polygon.set_properties('type', 'confirmed')
        self.active_polygon.set_properties('side', None)
        self.active_polygon.set_properties('side_manually_assigned', False)
        self.active_polygon.set_properties('contour_id', None) # Could be None - will be generated new in convert_drawings_to_entries()
        self.active_polygon.set_properties('edits',
        [{'username': self.gui.get_username(), 'timestamp': datetime.now().strftime("%m%d%Y%H%M%S")}])

        if hasattr(self.data_feeder, 'sections'):
            self.active_polygon.set_properties('section', self.active_section)
            d_voxel = np.mean(self.convert_section_to_z(sec=self.active_section, downsample=self.data_feeder.downsample))
            d_um = d_voxel * XY_PIXEL_DISTANCE_LOSSLESS * self.data_feeder.downsample
            self.active_polygon.set_properties('position_um', d_um)
            # print 'd_voxel', d_voxel, 'position_um', d_um
        else:
            self.active_polygon.set_properties('voxel_position', self.active_i)
            d_um = self.active_i * XY_PIXEL_DISTANCE_LOSSLESS * self.data_feeder.downsample
            self.active_polygon.set_properties('position_um', d_um)
            # print 'index', index, 'position_um', d_um

        if init_properties is not None:
            for k, v in init_properties.iteritems():
                self.active_polygon.set_properties(k, v)

    def show_information_box(self):
        assert self.active_polygon is not None, 'Must choose an active polygon first.'

        contour_info_text = "Abbreviation: %(name)s\n" % {'name': self.active_polygon.properties['label']}
        contour_info_text += "Fullname: %(fullname)s\n" % {'fullname': self.gui.structure_names[self.active_polygon.properties['label']]}

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

        if len(self.active_polygon.properties['edits']) > 0:
            print self.active_polygon.properties['edits']
            first_edit = self.active_polygon.properties['edits'][0]
            contour_info_text += "Created by %(creator)s at %(timestamp)s\n" % \
            {'creator': first_edit['username'],
            'timestamp':  datetime.strftime(datetime.strptime(first_edit['timestamp'], "%m%d%Y%H%M%S"), '%Y/%m/%d %H:%M')
            }

            last_edit = self.active_polygon.properties['edits'][-1]
            contour_info_text += "Last edited by %(editor)s at %(timestamp)s\n" % \
            {'editor': last_edit['username'],
            'timestamp':  datetime.strftime(datetime.strptime(last_edit['timestamp'], "%m%d%Y%H%M%S"), '%Y/%m/%d %H:%M')
            }
            print self.active_polygon.properties['edits']
        else:
            sys.stderr.write('No edit history.\n')

        contour_info_text += "Type: %(type)s\n" % {'type': self.active_polygon.properties['type']}
        contour_info_text += "Class: %(class)s\n" % {'class': self.active_polygon.properties['class']}

        QMessageBox.information(self.gview, "Information", contour_info_text)

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
            else:
                self.set_active_i(cross_z_ds, update_crossline=False)

        elif self.data_feeder.orientation == 'coronal':

            self.hline.setLine(0, cross_y_ds, self.data_feeder.z_dim-1, cross_y_ds)
            self.vline.setLine(self.data_feeder.z_dim-1-cross_z_ds, 0, self.data_feeder.z_dim-1-cross_z_ds, self.data_feeder.y_dim-1)

            self.set_active_i(cross_x_ds, update_crossline=False)

        elif self.data_feeder.orientation == 'horizontal':

            self.hline.setLine(0, self.data_feeder.z_dim-1-cross_z_ds, self.data_feeder.x_dim-1, self.data_feeder.z_dim-1-cross_z_ds)
            self.vline.setLine(cross_x_ds, 0, cross_x_ds, self.data_feeder.z_dim-1)

            self.set_active_i(cross_y_ds, update_crossline=False)

    def set_active_i(self, index, emit_changed_signal=True, update_crossline=True):
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).set_active_i(i=index, emit_changed_signal=emit_changed_signal)

        if update_crossline and hasattr(self, 'cross_x_lossless'):
            print 'update_crossline', update_crossline
            if hasattr(self.data_feeder, 'sections'):
                d1, d2 = self.convert_section_to_z(sec=self.active_section, downsample=1)
                cross_depth_lossless = .5 * d1 + .5 * d2
            else:
                print 'active_i =', self.active_i, 'downsample =', self.data_feeder.downsample
                cross_depth_lossless = self.active_i * self.data_feeder.downsample

            if self.data_feeder.orientation == 'sagittal':
                self.cross_z_lossless = cross_depth_lossless
            elif self.data_feeder.orientation == 'coronal':
                self.cross_x_lossless = cross_depth_lossless
            elif self.data_feeder.orientation == 'horizontal':
                self.cross_y_lossless = cross_depth_lossless

            print self.id, ': emit', self.cross_x_lossless, self.cross_y_lossless, self.cross_z_lossless
            self.crossline_updated.emit(self.cross_x_lossless, self.cross_y_lossless, self.cross_z_lossless, self.id)


    def set_active_section(self, section, emit_changed_signal=True, update_crossline=True):
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).set_active_section(sec=section, emit_changed_signal=emit_changed_signal)

        # if update_crossline and hasattr(self, 'cross_x_lossless'):
        #     if hasattr(self.data_feeder, 'sections'):
        #         d1, d2 = self.convert_section_to_z(sec=self.active_section, downsample=1)
        #         cross_depth_lossless = .5 * d1 + .5 * d2
        #     else:
        #         print 'active_i =', self.active_i, 'downsample =', self.data_feeder.downsample
        #         cross_depth_lossless = self.active_i * self.data_feeder.downsample
        #
        #     if self.data_feeder.orientation == 'sagittal':
        #         self.cross_z_lossless = cross_depth_lossless
        #     elif self.data_feeder.orientation == 'coronal':
        #         self.cross_x_lossless = cross_depth_lossless
        #     elif self.data_feeder.orientation == 'horizontal':
        #         self.cross_y_lossless = cross_depth_lossless
        #
        #     print self.id, ': emit', self.cross_x_lossless, self.cross_y_lossless, self.cross_z_lossless
        #     self.crossline_updated.emit(self.cross_x_lossless, self.cross_y_lossless, self.cross_z_lossless, self.id)


    def set_uncertainty_line(self, structure, e1, e2):
        if structure in self.uncertainty_lines:
            self.removeItem(self.uncertainty_lines[structure])
        self.uncertainty_lines[structure] = \
        self.addLine(e1[0], e1[1], e2[0], e2[1], QPen(QBrush(QColor(0, 0, 255, int(.3*255))), 20))

    def hide_uncertainty_line(self, structure):
        if structure in self.uncertainty_lines:
            self.removeItem(self.uncertainty_lines[structure])
        self.uncertainty_lines.pop(structure)

    def set_structure_onscreen_message(self, structure, msg, pos):
        if structure in self.structure_onscreen_messages:
            self.removeItem(self.structure_onscreen_messages[structure])
        message_text_item = self.addSimpleText(msg)
        message_text_item.setPos(pos[0], pos[1])
        message_text_item.setScale(1.5)
        self.structure_onscreen_messages[structure] = message_text_item

    def hide_structure_onscreen_message(self, structure):
        if structure in self.structure_onscreen_messages:
            self.removeItem(self.structure_onscreen_messages[structure])
        self.structure_onscreen_messages.pop(structure)


    def show_next(self, cycle=False):
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).show_next(cycle=cycle)
        assert all(['label' in p.properties for p in self.drawings[self.active_i]])

    def eventFilter(self, obj, event):
        # print obj.metaObject().className(), event.type()
        # http://doc.qt.io/qt-4.8/qevent.html#Type-enum

        if event.type() == QEvent.KeyPress:
            key = event.key()

            if key == Qt.Key_Escape:
                self.set_mode('idle')
                return True

            elif key == Qt.Key_S:
                if self.showing_which == 'scoremap':
                    self.showing_which = 'histology'
                else:
                    self.showing_which = 'scoremap'
                self.update_image()
                return True

            elif key == Qt.Key_X:
                if self.showing_which != 'red_only':
                    self.showing_which = 'red_only'
                else:
                    self.showing_which = 'histology'
                self.update_image()
                return True


            elif key == Qt.Key_Y:
                if self.showing_which != 'green_only':
                    self.showing_which = 'green_only'
                else:
                    self.showing_which = 'histology'
                self.update_image()
                return True

            elif key == Qt.Key_Z:
                if self.showing_which != 'blue_only':
                    self.showing_which = 'blue_only'
                else:
                    self.showing_which = 'histology'
                self.update_image()
                return True


            elif key == Qt.Key_I:
                self.show_information_box()
                return True

            elif key == Qt.Key_W:
                self.set_mode('rotate3d')
                return True

            elif key == Qt.Key_Q:
                self.set_mode('shift3d')
                return True

            elif (key == Qt.Key_Enter or key == Qt.Key_Return) and self.mode == 'add vertices consecutively': # Close polygon
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

            elif key == Qt.Key_M: # Toggle labeled cell markers
                for i, polygons in self.drawings.iteritems():
                    for p in polygons:
                        if 'class' in p.properties and p.properties['class'] == 'neuron':
                            if p.isVisible():
                                p.setVisible(False)
                            else:
                                p.setVisible(True)

            elif key == Qt.Key_N: # New marker
                self.set_mode('add vertices once')
                self.start_new_polygon(init_properties={'class': 'neuron'}, color=MARKER_COLOR_CHAR)

            elif key == Qt.Key_R: # Remove marker
                self.set_mode('remove marker')

            elif key == Qt.Key_B: # Clear default label name.
                self.default_name = None

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

        if event.type() == QEvent.GraphicsSceneMouseRelease:

            pos = event.scenePos()
            gscene_x = pos.x()
            gscene_y = pos.y()

            # Transform the current structure volume.
            # Notify GUI to use the new volume to update contours on all gscenes.
            if self.mode == 'rotate3d' or self.mode == 'shift3d':

                name_side_tuple = (self.active_polygon.properties['label'], self.active_polygon.properties['side'])
                assert name_side_tuple in self.structure_volumes, \
                "structure_volumes does not have %s. Need to reconstruct this structure first." % str(name_side_tuple)
                vol = self.structure_volumes[name_side_tuple]['volume_in_bbox']
                bbox = self.structure_volumes[name_side_tuple]['bbox']
                print 'vol', vol.shape, 'bbox', bbox
                ys, xs, zs = np.where(vol)
                cx_volResol = np.mean(xs)
                cy_volResol = np.mean(ys)
                cz_volResol = np.mean(zs)

                if self.mode == 'rotate3d':

                    cx_vol_resol_gl, cy_vol_resol_gl, cz_vol_resol_gl = (bbox[0] + cx_volResol, bbox[2] + cy_volResol, bbox[4] + cz_volResol)
                    if self.id == 'sagittal' or self.id == 'sagittal_tb':
                        active_structure_center_2d_vol_resol = np.array((cx_vol_resol_gl, cy_vol_resol_gl))
                        active_structure_center_2d_gscene_resol = active_structure_center_2d_vol_resol * self.structure_volumes_downscale_factor / self.data_feeder.downsample
                    elif self.id == 'coronal':
                        active_structure_center_2d_gscene_resol = np.array((self.data_feeder.z_dim - 1 - cz_vol_resol_gl * self.structure_volumes_downscale_factor / self.data_feeder.downsample,
                                                                            cy_vol_resol_gl * self.structure_volumes_downscale_factor / self.data_feeder.downsample))
                    elif self.id == 'horizontal':
                        active_structure_center_2d_gscene_resol = np.array((cx_vol_resol_gl * self.structure_volumes_downscale_factor / self.data_feeder.downsample,
                                                                            self.data_feeder.z_dim - 1 - cz_vol_resol_gl * self.structure_volumes_downscale_factor / self.data_feeder.downsample))

                    vec2 = np.array((gscene_x - active_structure_center_2d_gscene_resol[0], gscene_y - active_structure_center_2d_gscene_resol[1]))
                    vec1 = np.array((self.press_screen_x - active_structure_center_2d_gscene_resol[0], self.press_screen_y - active_structure_center_2d_gscene_resol[1]))
                    vec1n = np.sqrt(vec1[0]**2 + vec1[1]**2)
                    x2 = np.dot(vec2, vec1/vec1n)
                    y2 = (vec2 - x2*vec1/vec1n)[1]
                    theta_ccwise = np.arctan2(y2, x2)
                    print active_structure_center_2d_gscene_resol, vec2, vec1, x2, y2
                    print theta_ccwise, np.rad2deg(theta_ccwise)
                    if self.id == 'sagittal' or self.id == 'sagittal_tb':
                        tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xy=theta_ccwise)
                    elif self.id == 'coronal':
                        tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_yz=theta_ccwise)
                    elif self.id == 'horizontal':
                        tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xz=-theta_ccwise)

                elif self.mode == 'shift3d':

                    # shift_2d is in gscene resolution, which differs for each gscene.
                    shift_2d_gscene_resol = np.array((gscene_x - self.press_screen_x, gscene_y - self.press_screen_y))
                    shift_2d_orig_resol = shift_2d_gscene_resol * self.data_feeder.downsample
                    shift_2d_vol_resol = shift_2d_orig_resol / float(self.structure_volumes_downscale_factor)
                    print 'shift_2d_vol_resol', shift_2d_vol_resol
                    if self.id == 'sagittal' or self.id == 'sagittal_tb':
                        tf = affine_components_to_vector(tx=shift_2d_vol_resol[0],ty=shift_2d_vol_resol[1],tz=0)
                    elif self.id == 'coronal':
                        tf = affine_components_to_vector(tx=0,ty=shift_2d_vol_resol[1],tz=-shift_2d_vol_resol[0])
                    elif self.id == 'horizontal':
                        tf = affine_components_to_vector(tx=shift_2d_vol_resol[0],ty=0,tz=-shift_2d_vol_resol[1])

                tfed_structure_volume, tfed_structure_volume_bbox_rel = transform_volume_v2(vol.astype(np.int), tf,
                centroid_m=(cx_volResol, cy_volResol, cz_volResol), centroid_f=(cx_volResol, cy_volResol, cz_volResol))
                print 'tfed_structure_volume_bbox_rel', tfed_structure_volume_bbox_rel
                tfed_structure_volume_bbox = (tfed_structure_volume_bbox_rel[0] + bbox[0],
                                                tfed_structure_volume_bbox_rel[1] + bbox[0],
                                                tfed_structure_volume_bbox_rel[2] + bbox[2],
                                                tfed_structure_volume_bbox_rel[3] + bbox[2],
                                                tfed_structure_volume_bbox_rel[4] + bbox[4],
                                                tfed_structure_volume_bbox_rel[5] + bbox[4])
                print 'tfed_structure_volume.shape', tfed_structure_volume.shape, 'tfed_structure_volume_bbox', tfed_structure_volume_bbox

                self.structure_volumes[name_side_tuple]['volume_in_bbox'] = tfed_structure_volume
                self.structure_volumes[name_side_tuple]['bbox'] = tfed_structure_volume_bbox
                if self.mode == 'shift3d':
                    self.structure_volumes[name_side_tuple]['edits'].append(('shift3d', tf, (cx_volResol, cy_volResol, cz_volResol), (cx_volResol, cy_volResol, cz_volResol)))
                elif self.mode == 'rotate3d':
                    self.structure_volumes[name_side_tuple]['edits'].append(('rotate3d', tf, (cx_volResol, cy_volResol, cz_volResol), (cx_volResol, cy_volResol, cz_volResol)))

                self.structure_volume_updated.emit(self.active_polygon.properties['label'], self.active_polygon.properties['side'], False, False)
                self.set_mode('idle')

        if event.type() == QEvent.GraphicsSceneMousePress:

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

            elif self.mode == 'rotate3d':
                self.press_screen_x = gscene_x
                self.press_screen_y = gscene_y

            elif self.mode == 'shift3d':
                self.press_screen_x = gscene_x
                self.press_screen_y = gscene_y

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

            elif self.mode == 'add vertices once':
                if event.button() == Qt.LeftButton:
                    obj.mousePressEvent(event)
                    if not self.active_polygon.closed:
                        assert 'class' in self.active_polygon.properties and self.active_polygon.properties['class'] == 'neuron'
                        self.active_polygon.add_vertex(gscene_x, gscene_y, color=MARKER_COLOR_CHAR)
                    first_circ = self.active_polygon.vertex_circles[0]
                    first_circ.signal_emitter.press.emit(first_circ)
                    return False

            elif self.mode == 'add vertices consecutively':

                if event.button() == Qt.LeftButton:

                    obj.mousePressEvent(event)

                    if not self.active_polygon.closed:
                        if 'class' in self.active_polygon.properties and self.active_polygon.properties['class'] == 'neuron':
                            vertex_color = 'r'
                        else:
                            vertex_color = 'b'
                        self.active_polygon.add_vertex(gscene_x, gscene_y, color=vertex_color)

                    return True

            elif self.mode == 'add vertices randomly':
                if event.button() == Qt.LeftButton:
                    obj.mousePressEvent(event)

                    assert self.active_polygon.closed, 'Insertion is not allowed if polygon is not closed.'
                    new_index = find_vertex_insert_position(self.active_polygon, gscene_x, gscene_y)
                    if 'class' in self.active_polygon.properties and self.active_polygon.properties['class'] == 'neuron':
                        vertex_color = MARKER_COLOR_CHAR
                    else:
                        vertex_color = 'b'
                    self.active_polygon.add_vertex(gscene_x, gscene_y, new_index, color=vertex_color)

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
