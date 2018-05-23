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

# from memory_profiler import profile

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

    crossline_updated = pyqtSignal(object)
    structure_volume_updated = pyqtSignal(str, str, bool, bool)
    # prob_structure_volume_updated = pyqtSignal(str, str)
    global_transform_updated = pyqtSignal(object)

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

        self.histology_pixmap = QPixmap()

        self.rotation_center_wrt_wholebrain_volResol = None

        self.converter = None # coordinate converter

    def set_coordinates_converter(self, converter):
        """
        Set the converter used to convert between coordinates expressed in different frames and resolutions.
        """
        self.converter = converter

        if hasattr(self, 'data_feeder') and self.data_feeder is not None:
            self.converter.set_data(data_feeder=self.data_feeder, stack=self.gui.stack)

    def set_data_feeder(self, data_feeder):
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).set_data_feeder(data_feeder)
        if hasattr(self, 'converter') and self.converter is not None:
            self.converter.set_data(data_feeder=self.data_feeder, stack=self.gui.stack)

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

        elif mode == 'global_rotate3d' or mode == 'global_shift3d':
            self.gview.setDragMode(QGraphicsView.NoDrag)

    def set_structure_volumes(self, structure_volumes):
        """
        """
        self.structure_volumes = structure_volumes

    def set_structure_volumes_resolution(self, um):
        sys.stderr.write('%s: Set probabilistic structure volumes resolution to %.1f um\n' % (self.id, um))
        self.structure_volumes_resolution_um = um

    def update_drawings_from_structure_volume(self, name_s, levels, set_name):
        """
        Update drawings based on reconstructed structure volumes, which is a reference to the GUI's `prob_structure_volumes`.
        Polygons created by this function has the type "derived_from_atlas" if set name is "aligned_atlas" and type "intersected" if set name is "handdrawn".

        Args:
            name_s (str): sided structure name.
            levels (list of float): contours of these levels will be drawn onto the graphics scene.
            set_name (str): name of the annotation set, e.g. aligned_atlas or handdrawn.
        """

        # self.drawings = defaultdict(list) # Clear the internal variable `drawings`, and let `load_drawings` append to an empty set.

        print "%s: Updating drawings based on structure volume of %s" % (self.id, name_s)

        # If the structure volume has not been created, give up.
        if self.structure_volumes[set_name][name_s]['volume'] is None:
            sys.stderr.write("%s: Structure %s has not yet been reconstructed.\n" % (self.id, name_s))
            return

        structure_volume_volResol = self.structure_volumes[set_name][name_s]['volume']
        structure_origin_wrt_wholebrain_volResol = np.array(self.structure_volumes[set_name][name_s]['origin'])

        ###############################
        ## Remove existing contours. ##
        ###############################

        # Identify contours that will be removed.
        if set_name == 'aligned_atlas':
            types_to_remove = ['derived_from_atlas']
        elif set_name == 'handdrawn':
            types_to_remove = ['intersected']
        else:
            raise Exception("Set name %s does not exist." % set_name)

        name_u, side = parse_label(name_s, singular_as_s=True)[:2]
        polygons_to_remove = {i: [p for p in polygons \
                                                    if p.properties['label'] == name_u and \
                                                    p.properties['side'] == side and \
                                                    p.properties['type'] in types_to_remove]
                                                for i, polygons in self.drawings.iteritems()}

        # Remove the contours from drawings list and graphics scene.
        # t = time.time()
        for i in self.drawings.keys():
            for p in polygons_to_remove[i]:
                self.drawings[i].remove(p)
                # if i == self.active_i:
                self.removeItem(p)
        # sys.stderr.write("Remove unconfirmed polygons: %.2f seconds\n" % (time.time()-t))

        for level in levels:

            print "level", level

            structure_wrt = (name_s, self.data_feeder.orientation)

            if hasattr(self.data_feeder, 'sections'):

                structure_ddim = structure_volume_volResol.shape[2]

                in_pts = np.array(self.data_feeder.sections)

                positions_of_all_sections_wrt_structureVolume = self.converter.convert_frame_and_resolution(
                p=in_pts[:,None],
                in_wrt=('data', self.data_feeder.orientation), in_resolution='section',
                out_wrt=structure_wrt, out_resolution='volume')[..., 2].flatten()

                valid_mask = (positions_of_all_sections_wrt_structureVolume >= 0) & (positions_of_all_sections_wrt_structureVolume < structure_ddim)
                if np.count_nonzero(valid_mask) == 0:
                    sys.stderr.write("valid_mask is empty.\n")
                    continue

                positions_of_all_sections_wrt_structureVolume = positions_of_all_sections_wrt_structureVolume[valid_mask]
                # index_of_all_sections = in_pts[valid_mask]

                index_of_all_sections = \
                self.converter.convert_frame_and_resolution(
                p=in_pts[valid_mask][:,None],
                in_wrt=('data', self.data_feeder.orientation), in_resolution='section',
                out_wrt=('data', self.data_feeder.orientation), out_resolution='index').flatten()

            else:
                if self.data_feeder.orientation == 'sagittal':
                    ddim = self.data_feeder.z_dim
                    structure_ddim = structure_volume_volResol.shape[2]
                elif self.data_feeder.orientation == 'horizontal':
                    ddim = self.data_feeder.y_dim
                    structure_ddim = structure_volume_volResol.shape[0]
                elif self.data_feeder.orientation == 'coronal':
                    ddim = self.data_feeder.x_dim
                    structure_ddim = structure_volume_volResol.shape[1]
                else:
                    raise

                in_pts = np.column_stack([np.nan * np.ones((ddim,)), np.nan * np.ones((ddim,)), np.arange(0, ddim)])

                positions_of_all_sections_wrt_structureVolume = self.converter.convert_frame_and_resolution(
                p=in_pts,
                in_wrt=('data', self.data_feeder.orientation), in_resolution='image',
                out_wrt=structure_wrt, out_resolution='volume')[..., 2]

                valid_mask = (positions_of_all_sections_wrt_structureVolume >= 0) & (positions_of_all_sections_wrt_structureVolume < structure_ddim)
                if np.count_nonzero(valid_mask) == 0:
                    sys.stderr.write("valid_mask is empty.\n")
                    continue

                positions_of_all_sections_wrt_structureVolume = positions_of_all_sections_wrt_structureVolume[valid_mask]
                index_of_all_sections = in_pts[valid_mask, 2]

            positions_of_all_sections_wrt_structureVolume = np.round(positions_of_all_sections_wrt_structureVolume).astype(np.int)

            # print 'index_of_all_sections', index_of_all_sections
            # print 'positions_of_all_sections_wrt_structureVolume', positions_of_all_sections_wrt_structureVolume

            # If image resolution > 10 um
            if convert_resolution_string_to_um(resolution=self.data_feeder.resolution, stack=self.gui.stack) > 10.:
                sample_every = 5
            else:
                sample_every = 5

            contour_2d_wrt_structureVolume_sectionPositions_volResol = find_contour_points_3d(structure_volume_volResol >= level,
                                                            along_direction=self.data_feeder.orientation,
                                                            sample_every=sample_every,
                                                            positions=positions_of_all_sections_wrt_structureVolume)

            for d_wrt_structureVolume, cnt_uv_wrt_structureVolume in contour_2d_wrt_structureVolume_sectionPositions_volResol.iteritems():

                print
                print 'd_wrt_structureVolume', d_wrt_structureVolume

                contour_3d_wrt_structureVolume_volResol = np.column_stack([cnt_uv_wrt_structureVolume, np.ones((len(cnt_uv_wrt_structureVolume),)) * d_wrt_structureVolume])

                print 'contour_3d_wrt_structureVolume_volResol', contour_3d_wrt_structureVolume_volResol.mean(axis=0)

                contour_3d_wrt_dataVolume_uv_dataResol_index = self.converter.convert_frame_and_resolution(
                p=contour_3d_wrt_structureVolume_volResol,
                in_wrt=structure_wrt, in_resolution='volume',
                out_wrt=('data', self.data_feeder.orientation), out_resolution='image_image_index')

                uv_wrt_dataVolume_dataResol = contour_3d_wrt_dataVolume_uv_dataResol_index[..., :2]
                # Cannot use index_wrt_dataVolume as contour_3d_wrt_dataVolume_uv_dataResol_index[..., 2] because the rounding off will cause multiple contours to be placed on the same image
                index_wrt_dataVolume = int(index_of_all_sections[list(positions_of_all_sections_wrt_structureVolume).index(d_wrt_structureVolume)])

                # If this position already has a polygon for this structure, do not add a new one.
                if any([p.properties['label'] == name_u \
                # and p.properties['side'] == side \
                and p.properties['type'] == 'confirmed' \
                and p.properties['set_name'] == set_name
                        for p in self.drawings[index_wrt_dataVolume]]):
                    sys.stderr.write("d_wrt_structureVolume = %.1f already has a confirmed polygon for %s (set %s) on %s. Skip adding interpolated polygon.\n" % (d_wrt_structureVolume, name_u, set_name, self.id))
                    continue

                print 'index_wrt_dataVolume', index_wrt_dataVolume

                # # If image resolution > 10 um
                # if convert_resolution_string_to_um(resolution=self.data_feeder.resolution, stack=self.gui.stack) > 10.:
                #     # thumbnail graphics scenes
                #     linewidth = 1
                #     vertex_radius = .2
                # else:
                #     # raw graphics scenes
                #     linewidth = 10
                #     vertex_radius = 15

                if set_name == 'aligned_atlas':
                    new_polygon_type = 'derived_from_atlas'
                elif set_name == 'handdrawn':
                    new_polygon_type = 'intersected'
                else:
                    raise Exception("Set name %s does not exist." % set_name)

                try:
                    # if index_wrt_dataVolume  is None:
                    #     sys.exit()
                    self.add_polygon_with_circles_and_label(path=vertices_to_path(uv_wrt_dataVolume_dataResol),
                                                            index=index_wrt_dataVolume,
                                                            label=name_u,
                                                            # linewidth=linewidth,
                                                            linecolor=LEVEL_TO_COLOR_LINE[level] if set_name == 'handdrawn'else LEVEL_TO_COLOR_LINE2[level],
                                                            # vertex_radius=vertex_radius,
                                                            vertex_color=LEVEL_TO_COLOR_VERTEX[level] if set_name == 'handdrawn'else LEVEL_TO_COLOR_VERTEX2[level],
                                                            type=new_polygon_type,
                                                            level=level,
                                                            side=side,
                                                            side_manually_assigned=False,
                                                            set_name=set_name)
                    sys.stderr.write("%s: %s: Added polygon to d_wrt_structureVolume = %d, index_wrt_dataVolume = %d\n" % (self.id, name_u, d_wrt_structureVolume, index_wrt_dataVolume))
                except Exception as e:
                    sys.stderr.write("Failed adding polygon: %s\n" % e)

    def update_image(self, i=None, sec=None):
        """
        Update the image displayed in the graphics scene at the given index/section.
        """

        i, sec = self.get_requested_index_and_section(i=i, sec=sec)

        if self.showing_which == 'histology':

            t = time.time()
            qimage = self.data_feeder.retrieve_i(i=i)
            sys.stderr.write("data_feeder.retrieve: %.2f seconds.\n" % (time.time() - t))
            # histology_pixmap = QPixmap.fromImage(qimage)

            t = time.time()
            self.histology_pixmap.convertFromImage(qimage) # Keeping a global pixmap avoids creating a new pixmap every time which is tha case if using the static method QPixmap.fromImage(image)
            sys.stderr.write("convertFromImage: %.2f seconds.\n" % (time.time() - t))

            t = time.time()
            self.pixmapItem.setPixmap(self.histology_pixmap)
            sys.stderr.write("setPixmap: %.2f seconds.\n" % (time.time() - t))

            t = time.time()
            self.pixmapItem.setVisible(True)
            sys.stderr.write("setVisible: %.2f seconds.\n" % (time.time() - t))

        elif self.showing_which == 'scoremap':
            assert self.active_polygon is not None, 'Must have an active polygon first.'
            name_u = self.active_polygon.properties['label']
            # name_s = compose_label(self.active_polygon.properties['label'], self.active_polygon.properties['side'])
            # scoremap_viz_fp = DataManager.get_scoremap_viz_filepath(stack=self.gui.stack, downscale=8, section=sec, structure=name_u, detector_id=15, prep_id=2)

            if self.gui.stack in ['CHATM2', 'CHATM3']:
                scoremap_viz_fp = DataManager.get_scoremap_viz_filepath_v2(stack=self.gui.stack, out_resolution='10.0um', detector_id=799, prep_id=2, section=sec, structure=name_u)
            download_from_s3(scoremap_viz_fp)
            w, h = metadata_cache['image_shape'][self.gui.stack]
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
                # blue_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='contrastStretchedBlue', ext='jpg')
                # qimage_blue = QImage(blue_fp)

                # blue_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='ChatJpeg', ext='jpg')
                blue_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='raw', version='CHATJpeg')
                qimage_blue = QImage(blue_fp)

                # if self.data_feeder.downsample != 1:
                #     # Downsample the image for CryoJane data, which is too large and exceeds QPixmap size limit.
                #     raw_width, raw_height = (qimage_blue.width(), qimage_blue.height())
                #     new_width, new_height = (raw_width / self.data_feeder.downsample, raw_height / self.data_feeder.downsample)
                #     qimage_blue = qimage_blue.scaled(new_width, new_height)
                #     sys.stderr.write("Downsampling image by %.2f from size (w=%d,h=%d) to (w=%d,h=%d)\n" % (self.data_feeder.downsample, raw_width, raw_height, new_width, new_height))

                # green_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='contrastStretchedGreen', ext='jpg')
                # qimage_green = QImage(green_fp)
                # red_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='contrastStretchedRed', ext='jpg')
                # qimage_red = QImage(red_fp)
                sys.stderr.write('read images: %.2fs\n' % (time.time() - t))

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


    def clear_side(self):
        for section_index, polygons in self.drawings.iteritems():
            for p in polygons:
                if not p.properties['side_manually_assigned']:
                    p.set_properties('side', None)

    def infer_side(self):
        """
        Automatically assigns side to each polygon in this graphics scene.
        """

        assert hasattr(self.data_feeder, 'sections')
        first_sec, last_sec = DataManager.load_cropbox(self.gui.stack)[4:]
        mid_sec = int(round((first_sec + last_sec) / 2))

        label_indices_lookup = self.get_label_indices_lookup()
        # label_sections_lookup = {l: [self.data_feeder.sections[idx] for idx in indices] for l, indices in label_indices_lookup.iteritems()}

        index_ranges_all_structures = get_landmark_range_limits_v3(stack=self.data_feeder.stack, label_section_lookup=label_indices_lookup,
        mid_index=self.data_feeder.sections.index(mid_sec))
        print 'index_ranges_all_structures', sorted(index_ranges_all_structures.items())

        for idx, polygons in self.drawings.iteritems():
            for p in polygons:
                if p.properties['set_name'] == 'handdrawn':
                    name_u = p.properties['label']
                    if name_u in index_ranges_all_structures:
                        assert name_u in singular_structures, 'Label %s is in index_ranges_all_structures, but it is not singular.' % name_u

                        if idx >= index_ranges_all_structures[name_u][0] and idx <= index_ranges_all_structures[name_u][1]:
                            if p.properties['side'] is None or not p.properties['side_manually_assigned']:
                                p.set_properties('side', 'S')
                                p.set_properties('side_manually_assigned', False)
                        else:
                            raise Exception('Polygon for %s is on image index %d, but is not in structure_range.' % (name_u, idx ))
                    else:
                        lname = convert_to_left_name(name_u)
                        if lname in index_ranges_all_structures:
                            if index_ranges_all_structures[lname][1] is not None:
                                if idx <= index_ranges_all_structures[lname][1]:
                                    if p.properties['side'] is None or not p.properties['side_manually_assigned']:
                                        p.set_properties('side', 'L')
                                        p.set_properties('side_manually_assigned', False)
                                        # sys.stderr.write('%d, %d %s set to L\n' % (idx, self.data_feeder.sections[idx], p.properties['label']))

                        rname = convert_to_right_name(p.properties['label'])
                        if rname in index_ranges_all_structures:
                            if index_ranges_all_structures[rname][0] is not None:
                                if idx >= index_ranges_all_structures[rname][0]:
                                    if p.properties['side'] is None or not p.properties['side_manually_assigned']:
                                        p.set_properties('side', 'R')
                                        p.set_properties('side_manually_assigned', False)
                                        # sys.stderr.write('%d, %d %s set to R\n' % (idx, self.data_feeder.sections[idx], p.properties['label']))

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


    def load_drawings(self, contours, append=False, vertex_color=None, linecolor='r'):
        """
        Load annotation contours and place them onto graphicsScenes.

        Args:
            contours (DataFrame): row indices are random polygon_ids, and columns include vertices, edits, creator, section, type, name, side, class, label_position, time_created
            vertex_color (str or (3,)-array of 255-based int):
        """

        if not append:
            self.drawings = defaultdict(list) # keys are image indices

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

                print contour['name'], contour['side'], contour['section']

                vertices = contour['vertices']

                # type = confirmed or intersected
                if 'type' in contour:
                    if contour['type'] is not None:
                        if contour['name'] == 'SC' or contour['name'] == 'IC':
                            contour_type = 'intersected'
                        else:
                            contour_type = contour['type']
                    else:
                        contour_type = 'confirmed'
                else:
                    # contour_type = None
                    contour_type = 'confirmed'
                    # contour_type = 'intersected'

                # class = neuron or contour
                if 'class' in contour and contour['class'] is not None:
                    contour_class = contour['class']
                else:
                    contour_class = 'contour'

                print self.id, convert_resolution_string_to_voxel_size(resolution=self.data_feeder.resolution, stack=self.gui.stack)

                # if convert_resolution_string_to_voxel_size(resolution=self.data_feeder.resolution, stack=self.gui.stack) > 10.:
                #     linewidth = 1
                # else:
                #     linewidth = 30

                self.add_polygon_with_circles_and_label(path=vertices_to_path(vertices),
                                                        label=contour['name'], label_pos=contour['label_position'],
                                                        # linewidth=linewidth,
                                                        linecolor=linecolor,
                                                        vertex_color=vertex_color,
                                                        section=sec, type=contour_type,
                                                        side=contour['side'],
                                                        side_manually_assigned=contour['side_manually_assigned'],
                                                        edits=[{'username': contour['creator'], 'timestamp': contour['time_created']}] + contour['edits'],
                                                        contour_id=contour_id,
                                                        category=contour_class,
                                                        set_name='handdrawn')

    def convert_drawings_to_entries(self, timestamp, username, classes=None, types=None):
        """
        Convert polygon objects into dict records.

        Args:
            classes (list of str): list of classes to gather. Default is contour.

        Returns:
            dict: {polygon_id: contour entry}
        """

        import uuid
        # CONTOUR_IS_INTERPOLATED = 1
        contour_entries = {}
        for idx, polygons in self.drawings.iteritems():
            for polygon in polygons:

                if 'label' not in polygon.properties:
                    sys.stderr.write("A polygon in section %d has no labels. Ignore.\n" % self.data_feeder.sections[idx])
                    continue

                if classes is not None:
                    if 'class' not in polygon.properties or ('class' in polygon.properties and polygon.properties['class'] not in classes):
                        # raise Exception("polygon has no class: %d, %s" % (self.data_feeder.sections[idx], polygon.properties['label']))
                        sys.stderr.write("Polygon has no class: %d, %s. Skip." % (self.data_feeder.sections[idx], polygon.properties['label']))
                        continue

                if types is not None:
                    if polygon.properties['type'] not in types:
                        sys.stderr.write("Type %s of polygon %d, %s is not in list of desired types. Skip." % (polygon.properties['type'], self.data_feeder.sections[idx], polygon.properties['label']))
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
                                # 'downsample': self.data_feeder.downsample,
                                'resolution': self.data_feeder.resolution,
                                'type': None,
                                'orientation': self.data_feeder.orientation,
                                'parent_structure': [],
                                'side_manually_assigned': polygon.properties['side_manually_assigned'],
                                'id': polygon_id,
                                'class': polygon.properties['class']}
                    assert hasattr(self.data_feeder, 'sections')
                    contour_entry['section'] = self.data_feeder.sections[idx]
                    contour_entry['filename'] = metadata_cache['sections_to_filenames'][self.gui.stack][contour_entry['section']]
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
                                'resolution': self.data_feeder.resolution,
                                # 'downsample': self.data_feeder.downsample,
                            #    'flags': 0 if polygon.properties['type'] == 'confirmed' else 1,
                                'type': polygon.properties['type'],
                                'orientation': self.data_feeder.orientation,
                                'parent_structure': [],
                                'side_manually_assigned': polygon.properties['side_manually_assigned'],
                                'id': polygon_id,
                                'class': polygon.properties['class']}

                    if hasattr(self.data_feeder, 'sections'):
                        contour_entry['section'] = self.data_feeder.sections[idx]
                        contour_entry['filename'] = metadata_cache['sections_to_filenames'][self.gui.stack][contour_entry['section']]
                        if is_invalid(contour_entry['filename']):
                            continue
                    else:
                        contour_entry['voxel_position'] = idx

                contour_entries[polygon_id] = contour_entry

        return contour_entries

    # def get_label_section_lookup(self):
    def get_label_indices_lookup(self, only_use_confirmed_sides=True):
        """
        Returns:
            dict: {label: index list}. Labels are sided if the sides are confirmed. Otherwise labels are unsided. Values are list of image indices, NOT sections.
        """

        label_section_lookup = defaultdict(list)

        for section_index, polygons in self.drawings.iteritems():
            for p in polygons:
                if p.properties['side'] is None:
                    label = p.properties['label']
                elif p.properties['side'] == 'S':
                    label = p.properties['label']
                elif p.properties['side'] == 'L':
                    if (only_use_confirmed_sides and p.properties['side_manually_assigned']) or not only_use_confirmed_sides:
                        label = convert_to_left_name(p.properties['label'])
                    else:
                        label = p.properties['label']
                elif p.properties['side'] == 'R':
                    if (only_use_confirmed_sides and p.properties['side_manually_assigned']) or not only_use_confirmed_sides:
                        label = convert_to_right_name(p.properties['label'])
                    else:
                        label = p.properties['label']
                else:
                    raise Exception('Side property must be None, L or R.')

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

        # if self.mode == 'rotate2d' or self.mode == 'rotate3d':
        #     tf = self.active_polygon.transform()
        #     self.init_tf_2d = np.array([[tf.m11(), tf.m21(), tf.m31()],
        #                 [tf.m12(), tf.m22(), tf.m32()],
        #                 [tf.m13(), tf.m23(), tf.m33()]])

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


    def show_context_menu(self, pos_wrt_gview):

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

        action_confirmPolygon = myMenu.addAction("Set type to CONFIRMED")
        if hasattr(self, 'active_polygon') and self.active_polygon.properties['type'] == 'confirmed':
            print self.active_polygon.properties['type'],
            action_confirmPolygon.setText('Set type to INTERSECTED')
            # action_confirmPolygon.setVisible(False)

        reconstruct_menu = QMenu("Update 3D structure (using only confirmed contours)", myMenu)
        myMenu.addMenu(reconstruct_menu)
        reconstruct_menu_actions = \
        [reconstruct_menu.addAction('All annotated')] + \
        [reconstruct_menu.addAction('All atlas structures')] + \
        [reconstruct_menu.addAction(name_s) for name_s in sorted(all_known_structures_sided)]

        reconstructUnconfirmed_menu = QMenu("Update 3D structure (using all contours)", myMenu)
        myMenu.addMenu(reconstructUnconfirmed_menu)
        reconstructUnconfirmed_menu_actions = \
        [reconstructUnconfirmed_menu.addAction('All annotated')] + \
        [reconstructUnconfirmed_menu.addAction('All atlas structures')] + \
        [reconstructUnconfirmed_menu.addAction(name_s) for name_s in sorted(all_known_structures_sided)]

        # action_reconstruct = myMenu.addAction("Update 3D structure (using only confirmed contours)")
        # action_reconstructUsingUnconfirmed = myMenu.addAction("Update 3D structure (using all contours)")

        action_showInfo = myMenu.addAction("Show contour information")
        action_showReferences = myMenu.addAction("Show reference resources")
        action_syncCrossline = myMenu.addAction("Sync crossline")

        myMenu.addSeparator()

        action_gotoNextConfirmed = myMenu.addAction("Go to next confirmed")
        action_gotoPreviousConfirmed = myMenu.addAction("Go to previous confirmed")

        myMenu.addSeparator()

        # action_alignAtlasToManualStructureGlobal = myMenu.addAction("Align atlas structure to manual structure (global)")
        # action_alignAtlasToManualStructureLocal = myMenu.addAction("Align atlas structure to manual structure (local)")
        # action_moveAtlasStructureHereGlobal = myMenu.addAction("Move atlas structure here (global)")
        # action_moveAtlasStructureHereLocal = myMenu.addAction("Move atlas structure here (local)")

        moveAtlasStructureToHanddrawnGlobal_menu = QMenu("Move atlas structure to fit handdrawn annotation (global)", myMenu)
        myMenu.addMenu(moveAtlasStructureToHanddrawnGlobal_menu)
        moveAtlasStructureToHanddrawnGlobal_menu_actions = \
        [moveAtlasStructureToHanddrawnGlobal_menu.addAction(name_s) for name_s in sorted(all_known_structures_sided)]

        moveAtlasStructureToHanddrawnLocal_menu = QMenu("Move atlas structure to fit handdrawn annotation (local)", myMenu)
        myMenu.addMenu(moveAtlasStructureToHanddrawnLocal_menu)
        moveAtlasStructureToHanddrawnLocal_menu_actions = \
        [moveAtlasStructureToHanddrawnLocal_menu.addAction(name_s) for name_s in sorted(all_known_structures_sided)]

        moveAtlasStructureToMouseClickGlobal_menu = QMenu("Move atlas structure to here (global)", myMenu)
        myMenu.addMenu(moveAtlasStructureToMouseClickGlobal_menu)
        moveAtlasStructureToMouseClickGlobal_menu_actions = \
        [moveAtlasStructureToMouseClickGlobal_menu.addAction(name_s) for name_s in sorted(all_known_structures_sided)]

        moveAtlasStructureToMouseClickLocal_menu = QMenu("Move atlas structure to here (local)", myMenu)
        myMenu.addMenu(moveAtlasStructureToMouseClickLocal_menu)
        moveAtlasStructureToMouseClickLocal_menu_actions = \
        [moveAtlasStructureToMouseClickLocal_menu.addAction(name_s) for name_s in sorted(all_known_structures_sided)]

        myMenu.addSeparator()

        # resolution_menu = QMenu("Change resolution", myMenu)
        # myMenu.addMenu(resolution_menu)
        # action_resolutions = {}
        # for d in self.data_feeder.supported_downsample_factors:
        #     action = resolution_menu.addAction(str(d))
        #     action_resolutions[action] = d

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

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos_wrt_gview))

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

        elif selected_action == action_syncCrossline:
            pos_wrt_gscene = self.gview.mapToScene((pos_wrt_gview))
            self.compute_crossline(pos_wrt_gscene.x(), pos_wrt_gscene.y())

        elif selected_action == action_confirmPolygon:
            # self.active_polygon.set_type(None)
            if hasattr(self, 'active_polygon'):
                if self.active_polygon.properties['type'] != 'confirmed':
                    self.active_polygon.set_properties('type', 'confirmed')
                else:
                    self.active_polygon.set_properties('type', 'intersected')

        elif selected_action == action_deletePolygon:
            self.drawings[self.active_i].remove(self.active_polygon)
            self.removeItem(self.active_polygon)

        # elif selected_action == action_reconstruct:
        elif selected_action in reconstruct_menu_actions:
            # name_s = compose_label(self.active_polygon.properties['label'], self.active_polygon.properties['side'])
            # assert 'side' in self.active_polygon.properties and self.active_polygon.properties['side'] is not None, 'Must specify side first.'
            name_s = str(selected_action.text())

            if name_s == 'All annotated':
                label_indices_lookup = self.get_label_indices_lookup(only_use_confirmed_sides=False)
                for s in label_indices_lookup.keys():
                    self.structure_volume_updated.emit('handdrawn', s, True, True)
            elif name_s == 'All atlas structures':
                label_indices_lookup = self.get_label_indices_lookup(only_use_confirmed_sides=False)
                for s in set(label_indices_lookup.keys()) & set(all_known_structures_sided):
                    self.structure_volume_updated.emit('handdrawn', s, True, True)
            else:
                self.structure_volume_updated.emit('handdrawn', name_s, True, True)

        # elif selected_action == action_reconstructUsingUnconfirmed:
        elif selected_action in reconstructUnconfirmed_menu_actions:
            # name_s = compose_label(self.active_polygon.properties['label'], self.active_polygon.properties['side'])
            # assert 'side' in self.active_polygon.properties and self.active_polygon.properties['side'] is not None, 'Must specify side first.'
            name_s = str(selected_action.text())
            if name_s == 'All annotated':
                label_indices_lookup = self.get_label_indices_lookup(only_use_confirmed_sides=False)
                for s in label_indices_lookup.keys():
                    self.structure_volume_updated.emit('handdrawn', s, False, True)
            elif name_s == 'All atlas structures':
                label_indices_lookup = self.get_label_indices_lookup(only_use_confirmed_sides=False)
                for s in set(label_indices_lookup.keys()) & set(all_known_structures_sided):
                    self.structure_volume_updated.emit('handdrawn', s, True, True)
            else:
                self.structure_volume_updated.emit('handdrawn', name_s, False, True)

        # elif selected_action in action_resolutions:
        #     selected_downsample_factor = action_resolutions[selected_action]
        #     self.set_downsample_factor(selected_downsample_factor)

        elif selected_action == action_newPolygon:
            self.set_mode('add vertices consecutively')
            self.start_new_polygon(init_properties={'class': 'contour'})

        elif selected_action == action_insertVertex:
            self.set_mode('add vertices randomly')

        elif selected_action == action_setLabel:
            self.open_label_selection_dialog()

        elif selected_action == action_newMarker:
            self.set_mode('add vertices once')
            self.start_new_polygon(init_properties={'class': 'neuron'}, color=MARKER_COLOR_CHAR)

        elif selected_action in moveAtlasStructureToHanddrawnGlobal_menu_actions:
            name_s = str(selected_action.text())
            self.move_atlas_structure_to(name_s=name_s, global_instead_of_local=True, where='handdrawn_centroid')

        elif selected_action in moveAtlasStructureToHanddrawnLocal_menu_actions:
            name_s = str(selected_action.text())
            self.move_atlas_structure_to(name_s=name_s, global_instead_of_local=False, where='handdrawn_centroid')

        elif selected_action in moveAtlasStructureToMouseClickGlobal_menu_actions:
            name_s = str(selected_action.text())
            self.move_atlas_structure_to(name_s=name_s, global_instead_of_local=True, where='mouse_click')

        elif selected_action in moveAtlasStructureToMouseClickLocal_menu_actions:
            name_s = str(selected_action.text())
            self.move_atlas_structure_to(name_s=name_s, global_instead_of_local=False, where='mouse_click')

        elif selected_action == action_gotoNextConfirmed:

            items = all_known_structures_sided
            item, ok = QInputDialog.getItem(self.gui, "Which structure?", "list of structures", items, 0, False)
            if ok and item:
                name_u = convert_to_original_name(str(item))
            else:
                return

            all_indices_with_confirmed_contours = []
            for ind, polygons in self.drawings.iteritems():
                for p in polygons:
                    if p.properties['label'] == name_u and p.properties['type'] == 'confirmed':
                        all_indices_with_confirmed_contours.append(ind)
            all_indices_with_confirmed_contours = np.array(sorted(all_indices_with_confirmed_contours))
            print all_indices_with_confirmed_contours
            diffs = all_indices_with_confirmed_contours - self.active_i
            print diffs

            next_index_args = np.where(diffs > 0)[0]
            print next_index_args
            if len(next_index_args) == 0:
                print 'No more next confirmed contour for %s' % name_u
                return
            else:
                next_index_with_confirmed_contour = all_indices_with_confirmed_contours[next_index_args[0]]
                print next_index_with_confirmed_contour
                self.set_active_i(next_index_with_confirmed_contour)

        elif selected_action == action_gotoPreviousConfirmed:

            items = all_known_structures_sided
            item, ok = QInputDialog.getItem(self.gui, "Which structure?", "list of structures", items, 0, False)
            if ok and item:
                name_u = convert_to_original_name(str(item))
            else:
                return

            all_indices_with_confirmed_contours = []
            for ind, polygons in self.drawings.iteritems():
                for p in polygons:
                    if p.properties['label'] == name_u and p.properties['type'] == 'confirmed':
                        all_indices_with_confirmed_contours.append(ind)
            all_indices_with_confirmed_contours = np.array(sorted(all_indices_with_confirmed_contours))
            print all_indices_with_confirmed_contours
            diffs = all_indices_with_confirmed_contours - self.active_i
            print diffs

            previous_index_args = np.where(diffs < 0)[0]
            print previous_index_args
            if len(previous_index_args) == 0:
                print 'No more previous confirmed contour for %s' % name_u
                return
            else:
                previous_index_with_confirmed_contour = all_indices_with_confirmed_contours[previous_index_args[-1]]
                print previous_index_with_confirmed_contour
                self.set_active_i(previous_index_with_confirmed_contour)

        # elif selected_action == action_alignAtlasToManualStructureGlobal:
        #     self.move_atlas_structure_to(name_s=compose_label(self.active_polygon.properties['label'], side=self.active_polygon.properties['side']), global_instead_of_local=True, where='handdrawn_centroid')
        #
        # elif selected_action == action_alignAtlasToManualStructureLocal:
        #     self.move_atlas_structure_to(name_s=compose_label(self.active_polygon.properties['label'], side=self.active_polygon.properties['side']), global_instead_of_local=False, where='handdrawn_centroid')
        #
        # elif selected_action == action_moveAtlasStructureHereGlobal:
        #     self.move_atlas_structure_to(name_s=compose_label(self.active_polygon.properties['label'], side=self.active_polygon.properties['side']), global_instead_of_local=True, where='mouse_click')
        #
        # elif selected_action == action_moveAtlasStructureHereLocal:
        #     self.move_atlas_structure_to(name_s=compose_label(self.active_polygon.properties['label'], side=self.active_polygon.properties['side']), global_instead_of_local=False, where='mouse_click')


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

        # print 'before', self.structure_volumes['aligned_atlas'][name_s]['origin']
        # self.structure_volumes['aligned_atlas'][name_s]['origin'] = self.structure_volumes['aligned_atlas'][name_s]['origin'] - prob_structure_centroid3d_wrt_wholebrain_volResol + manual_structure_centroid3d_wrt_wholebrain_volResol
        # print 'after', self.structure_volumes['aligned_atlas'][name_s]['origin']

        # self.structure_volume_updated.emit('aligned_atlas', name_s, False, False)

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

        self.active_polygon.set_properties('set_name', 'handdrawn')

        if hasattr(self.data_feeder, 'sections'):
            self.active_polygon.set_properties('section', self.active_section)
            # d_voxel = DataManager.convert_section_to_z(sec=self.active_section, resolution=self.data_feeder.resolution, mid=True,
            # stack=self.gui.stack)
            # d_um = d_voxel * convert_resolution_string_to_voxel_size(stack=self.gui.stack, resolution='lossless') * self.data_feeder.downsample
            # d_um = d_voxel * convert_resolution_string_to_voxel_size(stack=self.gui.stack, resolution=self.data_feeder.resolution)

            d_um = self.converter.convert_frame_and_resolution([[self.active_section]],
            in_wrt=('data', self.data_feeder.orientation), in_resolution='section',
            out_wrt=('wholebrain', self.data_feeder.orientation), out_resolution='um')[0,0]

            self.active_polygon.set_properties('position_um', d_um)
            # print 'd_voxel', d_voxel, 'position_um', d_um
        else:
            self.active_polygon.set_properties('voxel_position', self.active_i)
            # d_um = self.active_i * convert_resolution_string_to_voxel_size(stack=self.gui.stack, resolution='lossless') * self.data_feeder.downsample
            d_um = self.active_i * convert_resolution_string_to_voxel_size(stack=self.gui.stack, resolution=self.data_feeder.resolution)
            self.active_polygon.set_properties('position_um', d_um)
            # print 'index', index, 'position_um', d_um

        if init_properties is not None:
            for k, v in init_properties.iteritems():
                self.active_polygon.set_properties(k, v)

    def show_information_box(self):
        """
        Show the information box of a polygon contour.
        """

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
        if 'level' in self.active_polygon.properties:
            contour_info_text += 'Prob level: %.2f\n' % self.active_polygon.properties['level']

        contour_info_text += "Class: %(class)s\n" % {'class': self.active_polygon.properties['class']}
        contour_info_text += "Position: %(position_um).2f microns (from origin of whole brain aligned and padded volume)\n" % {'position_um': self.active_polygon.properties['position_um']}

        contour_info_text += "Set name: %s\n" % self.active_polygon.properties['set_name']

        QMessageBox.information(self.gview, "Information", contour_info_text)

    def get_uv_dimension(self, plane):
        """
        in image resolution.
        """
        if self.data_feeder.orientation == 'sagittal':
            return self.data_feeder.x_dim - 1, self.data_feeder.y_dim - 1
        elif self.data_feeder.orientation == 'coronal':
            return self.data_feeder.z_dim - 1, self.data_feeder.y_dim - 1
        elif self.data_feeder.orientation == 'horizontal':
            return self.data_feeder.x_dim - 1, self.data_feeder.z_dim - 1
        else:
            print self.data_feeder.orientation
            raise

    def update_cross(self, cross):
        """
        Update positions of the two crosslines in this scene to the given `cross`.

        Args:
            cross (3-vector): intersection of the cross wrt wholebrain in raw resolution.
        """

        print self.id, ': update cross to', cross
        # self.hline.setVisible(True)
        # self.vline.setVisible(True)

        if hasattr(self.data_feeder, 'sections'):
            u, v, sec = self.converter.convert_frame_and_resolution([cross],
            in_wrt='wholebrain', in_resolution='raw',
            out_wrt=('data', self.data_feeder.orientation), out_resolution='image_image_section')[0]
            self.set_active_section(sec)
        else:
            u, v, d = self.converter.convert_frame_and_resolution([cross],
            in_wrt='wholebrain', in_resolution='raw',
            out_wrt=('data', self.data_feeder.orientation), out_resolution='image_image_index')[0]
            self.set_active_i(int(np.ceil(d)))

        udim, vdim = self.get_uv_dimension(plane=self.data_feeder.orientation)
        self.vline.setLine(u, 0, u, vdim)
        self.hline.setLine(0, v, udim, v)

        # if hasattr(self.data_feeder, 'sections'):
        #
        #     sec = self.converter.convert_frame_and_resolution([[d]],
        #     in_wrt=('data', self.data_feeder.orientation), in_resolution='image',
        #     out_wrt=('data', self.data_feeder.orientation), out_resolution='section')[0,0]

            # sec = DataManager.convert_z_to_section(z=d, resolution=self.data_feeder.resolution, stack=self.gui.stack,
            # z_first_sec=0)
            # sec = self.converter.convert_frame_and_resolution([[d]],
            # in_wrt=('data', self.data_feeder.orientation), in_resolution='image',
            # out_wrt=('data', self.data_feeder.orientation), out_resolution='section')[0,0]

            # self.set_active_section(sec)

            # z = DataManager.convert_section_to_z(sec=sec, resolution=self.data_feeder.resolution, stack=self.gui.stack, mid=True)
            # print d, sec, z
        # else:
        #     self.set_active_i(int(np.ceil(d)))

    # def set_active_i(self, index, emit_changed_signal=True, update_crossline=False):
    #     super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).set_active_i(i=index, emit_changed_signal=emit_changed_signal)
    #
    # def set_active_section(self, section, emit_changed_signal=True, update_crossline=True):
    #     super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).set_active_section(sec=section, emit_changed_signal=emit_changed_signal)

    def set_uncertainty_line(self, structure, e1, e2):
        if structure in self.uncertainty_lines:
            self.removeItem(self.uncertainty_lines[structure])
        self.uncertainty_lines[structure] = self.addLine(e1[0], e1[1], e2[0], e2[1], QPen(QBrush(QColor(0, 0, 255, int(.3*255))), 20))

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

    def compute_crossline(self, gscene_x, gscene_y):
        """
        Compute crossline position based on mouse click.
        Emit signal to update all scenes.
        """

        if hasattr(self.data_feeder, 'sections'):
            # d = DataManager.convert_section_to_z(sec=self.active_section, resolution=self.data_feeder.resolution, mid=True, stack=self.gui.stack)
            # d = self.converter.convert_frame_and_resolution(p=[[self.active_section]],
            # in_wrt=('data', self.data_feeder.orientation), in_resolution='section',
            # out_wrt=('data', self.data_feeder.orientation), out_resolution='index')[0][0]
            cross_wrt_wholebrain_rawResol = self.converter.convert_frame_and_resolution([(gscene_x, gscene_y, self.active_section)],
            in_wrt=('data', self.data_feeder.orientation), in_resolution='image_image_section',
            out_wrt='wholebrain', out_resolution='raw')[0]
        else:
            cross_wrt_wholebrain_rawResol = self.converter.convert_frame_and_resolution([(gscene_x, gscene_y, self.active_i)],
            in_wrt=('data', self.data_feeder.orientation), in_resolution='image',
            out_wrt='wholebrain', out_resolution='raw')[0]

        # print "(gscene_x, gscene_y, d) =", (gscene_x, gscene_y, d)

        # cross_wrt_wholebrain_rawResol = self.converter.convert_frame_and_resolution([(gscene_x, gscene_y, d)],
        # in_wrt=('data', self.data_feeder.orientation), in_resolution='image_image_index',
        # out_wrt='wholebrain', out_resolution='raw')[0]

        print self.id, ': emit crossline_updated signal', '; cross_wrt_wholebrain_rawResol', cross_wrt_wholebrain_rawResol
        self.crossline_updated.emit(cross_wrt_wholebrain_rawResol)
        return True

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
                print 'x'
                if self.showing_which != 'red_only':
                    self.showing_which = 'red_only'
                else:
                    self.showing_which = 'histology'
                self.update_image()
                return True

            elif key == Qt.Key_Y:
                print 'y'
                if self.showing_which != 'green_only':
                    self.showing_which = 'green_only'
                else:
                    self.showing_which = 'histology'
                self.update_image()
                return True

            elif key == Qt.Key_Z:
                print 'z'
                if self.showing_which != 'blue_only':
                    self.showing_which = 'blue_only'
                else:
                    self.showing_which = 'histology'
                self.update_image()
                return True

            elif key == Qt.Key_I:
                self.show_information_box()
                return True

            elif key == Qt.Key_Q:
                self.set_mode('shift3d')
                self.active_polygon.setFlag(QGraphicsItem.ItemIsMovable, True)
                return True

            elif key == Qt.Key_W:
                self.set_mode('rotate3d')
                return True

            elif key == Qt.Key_T:
                modifiers = QApplication.keyboardModifiers()
                if modifiers & Qt.AltModifier:
                    self.set_mode('global_shift3d')
                else:
                    self.set_mode('prob_shift3d')
                    self.active_polygon.setFlag(QGraphicsItem.ItemIsMovable, True)
                return True

            elif key == Qt.Key_R:
                modifiers = QApplication.keyboardModifiers()
                if modifiers & Qt.AltModifier: # hold alt
                    self.set_mode('global_rotate3d')
                else:
                    self.set_mode('prob_rotate3d')
                return True

            elif key == Qt.Key_J:
                self.set_mode('place_rotation_center')
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

            elif key == Qt.Key_L: # Toggle all labels
                for i, polygons in self.drawings.iteritems():
                    for p in polygons:
                        textitem = p.properties['label_textItem']
                        if textitem.isVisible():
                            textitem.setVisible(False)
                        else:
                            textitem.setVisible(True)

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

            elif key == Qt.Key_Control: # for moving a single 2d contour.
                if not event.isAutoRepeat():
                    # for polygon in self.drawings[self.active_i]:
                    #     polygon.setFlag(QGraphicsItem.ItemIsMovable, True)
                    self.active_polygon.setFlag(QGraphicsItem.ItemIsMovable, True)
                    self.set_mode('shift2d')

            elif key == Qt.Key_Shift:
                if not event.isAutoRepeat(): # Ignore events that are auto repeats of the original SHIFT keypress
                    # if mode is rotate3d, do not change it, because at mouserelease, we want to be in rotate3d state to execute the 3d structure transform.
                    if self.mode == 'rotate3d':
                        self.set_mode('rotate3d')
                    else:
                        self.set_mode('rotate2d')

        elif event.type() == QEvent.KeyRelease:
            key = event.key()

            if key == Qt.Key_Control:
                if not event.isAutoRepeat():
                    # for polygon in self.drawings[self.active_i]:
                    #     polygon.setFlag(QGraphicsItem.ItemIsMovable, False)
                    self.active_polygon.setFlag(QGraphicsItem.ItemIsMovable, False)
                    self.set_mode('idle')

            elif key == Qt.Key_Shift:
                if not event.isAutoRepeat():
                    self.set_mode('idle')

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

        elif event.type() == QEvent.GraphicsSceneMouseMove:

            pos = event.scenePos()
            curr_mouse_x_wrt_imageData_gsceneResol = pos.x()
            curr_mouse_y_wrt_imageData_gsceneResol = pos.y()

            if self.mode == 'rotate2d' or self.mode == 'rotate3d' or self.mode == 'prob_rotate3d':
                # This only moves the single contour on the current image.
                # Those contours of the same structure but on other sections are not affected.

                if self.mouse_under_press:

                    active_polygon_vertices = vertices_from_polygon(polygon=self.active_polygon)
                    polygon_cx_wrt_imageData_gsceneResol, polygon_cy_wrt_imageData_gsceneResol = np.mean(active_polygon_vertices, axis=0)

                    vec2 = np.array([curr_mouse_x_wrt_imageData_gsceneResol - polygon_cx_wrt_imageData_gsceneResol, curr_mouse_y_wrt_imageData_gsceneResol - polygon_cy_wrt_imageData_gsceneResol])
                    vec1 = np.array([self.press_x_wrt_imageData_gsceneResol - polygon_cx_wrt_imageData_gsceneResol, self.press_y_wrt_imageData_gsceneResol - polygon_cy_wrt_imageData_gsceneResol])
                    theta_ccwise = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])

                    # Note: Using 3d tranforms is not necessary. Can use 2-d versions.
                    tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xy=theta_ccwise)
                    A = consolidate(tf,
                    centroid_m=[polygon_cx_wrt_imageData_gsceneResol, polygon_cy_wrt_imageData_gsceneResol, 0],
                    centroid_f=[polygon_cx_wrt_imageData_gsceneResol, polygon_cy_wrt_imageData_gsceneResol, 0])
                    tf_mat_combined = A
                    xform = QTransform(tf_mat_combined[0,0], tf_mat_combined[1,0], 0.,
                                        tf_mat_combined[0,1], tf_mat_combined[1,1], 0.,
                                        tf_mat_combined[0,3], tf_mat_combined[1,3], 1.)
                    self.active_polygon.setTransform(xform, combine=False)

        elif event.type() == QEvent.GraphicsSceneMousePress:

            # Notice that if self.active_polygon has not been set when enter this,
            # it will be set only after all the actions specified here have been executed.
            # So any action below that requires self.active_polygon must ensure
            # self.active_polygon has already been set AND is pointing to the correct polygon.

            pos = event.scenePos()
            gscene_x = pos.x()
            gscene_y = pos.y()

            # for compatibility purpose; Will clean up later...
            self.press_x_wrt_imageData_gsceneResol = gscene_x
            self.press_y_wrt_imageData_gsceneResol = gscene_y

            self.mouse_under_press = True

            if event.button() == Qt.RightButton:
                obj.mousePressEvent(event)

            if self.mode == 'idle':
                # pass the event down
                obj.mousePressEvent(event)
                self.pressed = True
                return True

            elif self.mode == 'crossline':
                # Enter here if a user left-clicks while in crossline mode (holding down space bar).
                self.compute_crossline(gscene_x, gscene_y)

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

            elif self.mode == 'place_rotation_center':
                obj.mousePressEvent(event)

                self.rotation_center_wrt_wholebrain_volResol = self.converter.convert_frame_and_resolution(p=[(gscene_x, gscene_y, 0)],
                in_wrt=('data', self.data_feeder.orientation), in_resolution='image',
                out_wrt='wholebrain', out_resolution='volume')[0]
                print "Set rotation_center_wrt_wholebrain_volResol =",  self.rotation_center_wrt_wholebrain_volResol
                self.set_mode('idle')
                return True

        elif event.type() == QEvent.GraphicsSceneMouseRelease:

            pos = event.scenePos()
            self.gscene_x = pos.x()
            self.gscene_y = pos.y()

            self.mouse_under_press = False

            if self.mode == 'shift2d':
                scene_tf_mat = qtransform_to_matrix2d(self.active_polygon.sceneTransform())
                # local_tf_mat = qtransform_to_matrix2d(self.active_polygon.transform())
                print 'scene_tf_mat', scene_tf_mat
                # print 'local_tf_mat', local_tf_mat
                # pre_tf_vertices = vertices_from_polygon(polygon=self.active_polygon)
                # scene_tf_mat = qtransform_to_matrix2d(self.active_polygon.sceneTransform())
                # local_tf_mat = qtransform_to_matrix2d(self.active_polygon.transform())
                # tf_mat = np.dot(local_tf_mat, scene_tf_mat)
                # post_tf_vertices = np.dot(tf_mat[:2,:2], pre_tf_vertices.T).T + tf_mat[:2,2]
                #
                # polygon_to_delete = self.active_polygon
                # self.add_polygon_with_circles_and_label(path=vertices_to_path(post_tf_vertices, closed=True),
                #                                         label=polygon_to_delete.properties['label'], linecolor='r', vertex_radius=8, linewidth=5,
                #                                         section=polygon_to_delete.properties['section'],
                #                                         type=polygon_to_delete.properties['type'],
                #                                         side=polygon_to_delete.properties['side'],
                #                                         side_manually_assigned=polygon_to_delete.properties['side_manually_assigned'])
                # self.delete_polygon(polygon=polygon_to_delete)

            elif self.mode == 'rotate2d':
                pre_tf_vertices = vertices_from_polygon(polygon=self.active_polygon)
                # Note that local transforms are different from scene transform.
                # scene transform includes both scene's global position and local transforms.

                scene_tf_mat = qtransform_to_matrix2d(self.active_polygon.sceneTransform())
                # local_tf_mat = qtransform_to_matrix2d(self.active_polygon.transform())
                print 'scene_tf_mat', scene_tf_mat
                # print 'local_tf_mat', local_tf_mat
                # tf_mat = np.dot(scene_tf_mat, local_tf_mat)
                tf_mat = scene_tf_mat
                post_tf_vertices = np.dot(tf_mat[:2,:2], pre_tf_vertices.T).T + tf_mat[:2,2]

                polygon_to_delete = self.active_polygon
                self.add_polygon_with_circles_and_label(path=vertices_to_path(post_tf_vertices, closed=True),
                                                        label=polygon_to_delete.properties['label'], linecolor='r',
                                                        # vertex_radius=8, linewidth=5,
                                                        section=polygon_to_delete.properties['section'],
                                                        type=polygon_to_delete.properties['type'],
                                                        side=polygon_to_delete.properties['side'],
                                                        side_manually_assigned=polygon_to_delete.properties['side_manually_assigned'])
                self.delete_polygon(polygon=polygon_to_delete)

            # Transform the current structure volume.
            # Notify GUI to use the new volume to update contours on all gscenes.
            elif self.mode == 'rotate3d' or self.mode == 'shift3d':

                # name_side_tuple = (self.active_polygon.properties['label'], self.active_polygon.properties['side'])
                # self.transform_structure(name=self.active_polygon.properties['label'], side=self.active_polygon.properties['side'])
                # self.set_mode('idle')
                pass

            elif self.mode == 'global_shift3d' or self.mode == 'global_rotate3d':

                if self.active_polygon.properties['set_name'] == 'aligned_atlas':

                    # Apply the given global transform to every structure.
                    for name_s in self.structure_volumes['aligned_atlas'].iterkeys():
                        self.transform_structure(name_s=name_s, set_name='aligned_atlas')
                    # Nullify the rotation center after using it.
                    self.rotation_center_wrt_wholebrain_volResol = None
                    # sys.stderr.write("nullify\n")
                else:
                    sys.stderr.write("Global adjustment can only be applied to polygons in the set aligned_atlas. Do nothing.\n")

                self.set_mode('idle')

                # self.global_transform_updated.emit(self.get_global_transform())

            elif self.mode == 'prob_shift3d' or self.mode == 'prob_rotate3d':

                self.transform_structure(name_s=compose_label(self.active_polygon.properties['label'], side=self.active_polygon.properties['side']), set_name=self.active_polygon.properties['set_name'])
                # Nullify the rotation center after using it.
                self.rotation_center_wrt_wholebrain_volResol = None
                # sys.stderr.write("nullify\n")
                self.set_mode('idle')

            elif self.mode == 'delete vertices':
                items_in_rubberband = self.analyze_rubberband_selection()

                for polygon, vertex_indices in items_in_rubberband.iteritems():
                    polygon.delete_vertices(vertex_indices, merge=True)
                    # if self.mode == Mode.DELETE_ROI_DUPLICATE:
                    #     self.delete_vertices(polygon, vertex_indices)
                    # elif self.mode == Mode.DELETE_ROI_MERGE:
                    #     self.delete_vertices(polygon, vertex_indices, merge=True)

                # self.set_mode('idle')
                # self.set_mode(Mode.IDLE)

            self.press_x_wrt_imageData_gsceneResol = None
            self.press_y_wrt_imageData_gsceneResol = None

        return False
        # return super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).eventFilter(obj, event)

    def compute_rotate_transform_vector(self, start, finish, center):
        """
        Compute the 3x4 transform matrix representing the rotation around
        an axis piercing through image plane at `axis`.

        Args:
            finish (3-tuple of float): 3-D coordinate of finish point
            start (3-tuple of float): 3-D coordinate of start point
            center (3-tuple of float): 3-D coordinate of the rotation center

        Returns:
            12-tuple of float: flattened 3x4 transform matrix. Frame and resolution are identical to input.
        """

        i = np.where(start - finish == 0)[0]
        assert len(i) == 1, 'Movement should be more curvy.'
        if i == 0: # around x axis
            print 'around x axis'
            vec2 = finish[[1,2]] - center[[1,2]]
            vec1 = start[[1,2]] - center[[1,2]]
            theta_ccwise = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            print np.rad2deg(theta_ccwise)
            tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_yz=theta_ccwise,c=center)
            print tf.reshape((3,4))
        elif i == 1: # around y axis
            print 'around y axis'
            vec2 = finish[[0,2]] - center[[0,2]]
            vec1 = start[[0,2]] - center[[0,2]]
            theta_ccwise = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            print np.rad2deg(theta_ccwise)
            tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xz=theta_ccwise,c=center)
            print tf.reshape((3,4))
        elif i == 2: # around z axis
            print 'around z axis'
            vec2 = finish[[0,1]] - center[[0,1]]
            vec1 = start[[0,1]] - center[[0,1]]
            theta_ccwise = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            print np.rad2deg(theta_ccwise)
            tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xy=theta_ccwise,c=center)
            print tf.reshape((3,4))
        else:
            raise

        return tf

    # @profile(precision=4)
    def transform_structure(self, name_s, set_name, tf=None):
        """
        Compute transform based on recorded mouse movements.
        Transform the given structure and save back into repository.
        Then send signal to update display.

        Args:
            name_s (str): Structure name, sided.
        """

        # structure_volumes = self.structure_volumes['aligned_atlas']
        structure_volume_resolution_um = self.structure_volumes_resolution_um

        assert name_s in self.structure_volumes[set_name], \
        "`structure_volumes` does not contain %s. Need to load this structure first." % name_s

        press_position_wrt_wholebrain_volResol = self.converter.convert_frame_and_resolution(p=[(self.press_x_wrt_imageData_gsceneResol, self.press_y_wrt_imageData_gsceneResol, 0)],
        in_wrt=('data', self.data_feeder.orientation), in_resolution='image', out_wrt='wholebrain', out_resolution='volume')[0]

        release_position_wrt_wholebrain_volResol = self.converter.convert_frame_and_resolution(p=[(self.gscene_x, self.gscene_y, 0)],
        in_wrt=('data', self.data_feeder.orientation), in_resolution='image', out_wrt='wholebrain', out_resolution='volume')[0]

        if self.structure_volumes[set_name][name_s]['volume'] is not None: # the volume is loaded

            vol = self.structure_volumes[set_name][name_s]['volume'].copy()
            vol_origin_wrt_wholebrain_volResol = np.array(self.structure_volumes[set_name][name_s]['origin'])
            print 'vol', vol.shape, 'vol_origin_wrt_wholebrain_volResol', vol_origin_wrt_wholebrain_volResol

        if tf is None:

            if self.mode == 'prob_rotate3d' or self.mode == 'rotate3d' or self.mode == 'global_rotate3d':

                if self.mode == 'global_rotate3d':

                    if self.rotation_center_wrt_wholebrain_volResol is None:
                        sys.stderr.write('Must specify rotation center.\n')
                        return

                    print 'press', press_position_wrt_wholebrain_volResol, 'release', release_position_wrt_wholebrain_volResol, 'center', self.rotation_center_wrt_wholebrain_volResol

                    tf = self.compute_rotate_transform_vector(start=press_position_wrt_wholebrain_volResol,
                    finish=release_position_wrt_wholebrain_volResol,
                    center=self.rotation_center_wrt_wholebrain_volResol)

                else:

                    if self.rotation_center_wrt_wholebrain_volResol is None:
                        sys.stderr.write('No rotation center is specified. Using contour center.\n')
                        rotation_center_wrt_wholebrain_volResol = self.converter.convert_frame_and_resolution(p=[np.r_[np.mean(vertices_from_polygon(polygon=self.active_polygon), axis=0), 0]],
                        in_wrt=('data', self.data_feeder.orientation), in_resolution='image', out_wrt='wholebrain', out_resolution='volume')[0]
                    else:
                        rotation_center_wrt_wholebrain_volResol = self.rotation_center_wrt_wholebrain_volResol

                    tf = self.compute_rotate_transform_vector(start=press_position_wrt_wholebrain_volResol,
                    finish=release_position_wrt_wholebrain_volResol,
                    center=rotation_center_wrt_wholebrain_volResol
                    )

            elif self.mode == 'prob_shift3d' or self.mode == 'shift3d' or self.mode == 'global_shift3d':

                shift = release_position_wrt_wholebrain_volResol - press_position_wrt_wholebrain_volResol
                print release_position_wrt_wholebrain_volResol, press_position_wrt_wholebrain_volResol, shift
                tf = affine_components_to_vector(tx=shift[0], ty=shift[1], tz=shift[2])

            else:
                raise

        # If this structure's volume has not been loaded, don't do the transform, just add edits.

        if self.structure_volumes[set_name][name_s]['volume'] is not None: # the volume is loaded

            del self.structure_volumes[set_name][name_s]['volume']

            self.structure_volumes[set_name][name_s]['volume'], \
            self.structure_volumes[set_name][name_s]['origin'] = \
            transform_volume_v4(volume=(vol, vol_origin_wrt_wholebrain_volResol), transform=tf, return_origin_instead_of_bbox=True)

        ###################### Append edits ###############################

        edit_entry = {'username': self.gui.get_username(),
        'timestamp': datetime.now().strftime("%m%d%Y%H%M%S"),
        'type': self.mode,
        'transform':tf}
        self.structure_volumes[set_name][name_s]['edits'].append(edit_entry)
        print "Edit entry %s added to %s, %s" % (self.mode, set_name, name_s)

        self.structure_volume_updated.emit(set_name, name_s, False, False)


    def get_structure_centroid3d(self, set_name, name_s):
        """
        Compute given structure's centroid in 3-D wrt wholebrain.

        Args:
            set_name (str):
            name_s (str): name, sided
        Return:
            ((3,)-array): structure centroid in 3d wrt wholebrain
        """

        assert 'volume' in self.structure_volumes[set_name][name_s] and \
        self.structure_volumes[set_name][name_s]['volume'] is not None and \
        'origin' in self.structure_volumes[set_name][name_s] and \
        self.structure_volumes[set_name][name_s]['origin'] is not None, \
        'Structure volume of set %s, %s has not been initialized.' % (set_name, name_s)

        vol = self.structure_volumes[set_name][name_s]['volume']

        yc_wrt_structureVolInBbox_volResol, \
        xc_wrt_structureVolInBbox_volResol, \
        zc_wrt_structureVolInBbox_volResol = np.mean(np.where(vol), axis=1)

        structure_centroid3d_wrt_structureVolInBbox_volResol = \
        np.array((xc_wrt_structureVolInBbox_volResol, \
        yc_wrt_structureVolInBbox_volResol, \
        zc_wrt_structureVolInBbox_volResol))

        vol_origin_wrt_wholebrain_volResol = np.array(self.structure_volumes[set_name][name_s]['origin'])

        # print vol_origin_wrt_wholebrain_volResol, structure_centroid3d_wrt_structureVolInBbox_volResol
        structure_centroid3d_wrt_wholebrain_volResol = structure_centroid3d_wrt_structureVolInBbox_volResol + vol_origin_wrt_wholebrain_volResol

        return np.array(structure_centroid3d_wrt_wholebrain_volResol)

    def move_atlas_structure_to(self, name_s, global_instead_of_local=False, where='handdrawn_centroid'):

        if where == 'handdrawn_centroid':
            target_centroid3d_wrt_wholebrain_volResol = self.get_structure_centroid3d(name_s=name_s, set_name='handdrawn')
        elif where == 'mouse_click':
            print (self.press_x_wrt_imageData_gsceneResol, self.press_y_wrt_imageData_gsceneResol, self.active_i)
            target_centroid3d_wrt_wholebrain_volResol = self.converter.convert_frame_and_resolution(p=[(self.press_x_wrt_imageData_gsceneResol, self.press_y_wrt_imageData_gsceneResol, self.active_i)],
            in_wrt=('data', self.data_feeder.orientation), in_resolution='image_image_index', out_wrt='wholebrain', out_resolution='volume')[0]
        else:
            raise Exception("where = %s is not recognized." % where)

        print "target =", target_centroid3d_wrt_wholebrain_volResol

        atlas_structure_centroid3d_wrt_wholebrain_volResol = self.get_structure_centroid3d(name_s=name_s, set_name='aligned_atlas')
        tf = np.column_stack([np.eye(3), - atlas_structure_centroid3d_wrt_wholebrain_volResol + target_centroid3d_wrt_wholebrain_volResol])

        if global_instead_of_local:
            for name in self.structure_volumes['aligned_atlas'].keys():
                self.transform_structure(name_s=name, set_name='aligned_atlas', tf=tf)
        else:
            self.transform_structure(name_s=name_s, set_name='aligned_atlas', tf=tf)

    # def align_atlas_structure_to_manual_structure(self, name_s, global_instead_of_local=False):
    #
    #     manual_structure_centroid3d_wrt_wholebrain_volResol = self.get_structure_centroid3d(name_s=name_s, set_name='handdrawn')
    #     print "handdrawn =", manual_structure_centroid3d_wrt_wholebrain_volResol
    #     prob_structure_centroid3d_wrt_wholebrain_volResol = self.get_structure_centroid3d(name_s=name_s, set_name='aligned_atlas')
    #     print "atlas =", prob_structure_centroid3d_wrt_wholebrain_volResol
    #
    #     tf = np.column_stack([np.eye(3), - prob_structure_centroid3d_wrt_wholebrain_volResol + manual_structure_centroid3d_wrt_wholebrain_volResol])
    #
    #     if global_instead_of_local:
    #         for name in self.structure_volumes['aligned_atlas'].keys():
    #             self.transform_structure(name_s=name, set_name='aligned_atlas', tf=tf)
    #     else:
    #         self.transform_structure(name_s=name_s, set_name='aligned_atlas', tf=tf)
