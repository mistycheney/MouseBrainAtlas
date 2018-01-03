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
    structure_volume_updated = pyqtSignal(str, str, bool, bool)
    prob_structure_volume_updated = pyqtSignal(str, str)

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

    def set_prob_structure_volumes(self, structure_volumes):
        """
        Args:
        """

        self.prob_structure_volumes = structure_volumes

    def set_structure_volumes_downscale_factor(self, downscale):
        sys.stderr.write('Set structure volumes downscale to %d\n' % downscale)
        self.structure_volumes_downscale_factor = downscale

    def set_prob_structure_volumes_downscale_factor(self, downscale):
        sys.stderr.write('Set probalistci structure volumes downscale to %d\n' % downscale)
        self.prob_structure_volumes_downscale_factor = downscale

    def update_drawings_from_structure_volume(self, name_u, side):
        """
        Update drawings based on `self.structure_volumes`, which is a reference to the GUI's `structure_volumes`.

        Args:
            name_u (str): structure name, unsided
            side (str): L, R or S
        """

        print "%s: Updating drawings based on structure volume of %s, %s" % (self.id, name_u, side)

        volume_volResol = self.structure_volumes[(name_u, side)]['volume_in_bbox']
        internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol = np.array(self.structure_volumes[(name_u, side)]['bbox'])
        internal_structure_origin_wrt_WholebrainAlignedPadded_volResol = internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol[[0,2,4]]

        data_origin_wrt_WholebrainAlignedPadded_dataResol = self.gui.image_origin_wrt_WholebrainAlignedPadded_tbResol[self.id] * 32. / self.data_feeder.downsample
        data_origin_wrt_WholebrainAlignedPadded_volResol = data_origin_wrt_WholebrainAlignedPadded_dataResol * self.data_feeder.downsample / float(self.structure_volumes_downscale_factor)
        internal_structure_origin_wrt_dataVolume_volResol = internal_structure_origin_wrt_WholebrainAlignedPadded_volResol - data_origin_wrt_WholebrainAlignedPadded_volResol
        internal_structure_origin_wrt_dataVolume_dataResol = internal_structure_origin_wrt_dataVolume_volResol * float(self.structure_volumes_downscale_factor) / self.data_feeder.downsample

        print 'internal_structure_origin_wrt_dataVolume_dataResol', internal_structure_origin_wrt_dataVolume_dataResol

        print 'volume (internal vol resol)', volume_volResol.shape, internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol

        volume_downsample_factor = self.structure_volumes_downscale_factor
        bbox_wrt_WholebrainAlignedPadded_losslessResol = np.array(internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol) * volume_downsample_factor

        data_vol_downsample_ratio = float(self.data_feeder.downsample) / volume_downsample_factor
        print 'Downsample from image data to internal volume representation =', data_vol_downsample_ratio

        if data_vol_downsample_ratio > 1:
            volume_dataResol = volume_volResol[::int(data_vol_downsample_ratio), ::int(data_vol_downsample_ratio), ::int(data_vol_downsample_ratio)]
            bbox_wrt_WholebrainAlignedPadded_dataResol = np.array(bbox_wrt_WholebrainAlignedPadded_losslessResol) / self.data_feeder.downsample
            # xmin_ds, xmax_ds, ymin_ds, ymax_ds, zmin_ds, zmax_ds = np.array(bbox_wrt_WholebrainAlignedPadded_losslessResol) / self.data_feeder.downsample
            print 'volume (data resol)', volume_dataResol.shape, bbox_wrt_WholebrainAlignedPadded_dataResol

        if hasattr(self.data_feeder, 'sections'):

            assert self.data_feeder.orientation == 'sagittal', "Current implementation only considers sagittal sections."

            print "Removing all unconfirmed polygons..."

            t = time.time()

            for i in range(len(self.data_feeder.sections)):
                for p in self.drawings[i]:
                    assert 'label' in p.properties, "ERROR! polygon has no label i=%d, sec=%d" % (i, self.data_feeder.sections[i])

            # Find all unconfirmed polygons. These are to be removed.
            matched_unconfirmed_polygons_to_remove = {i: [p for p in self.drawings[i] \
                                                        if p.properties['label'] == name_u and \
                                                        p.properties['side'] == side and \
                                                        p.properties['type'] != 'confirmed']
                                                    for i in range(len(self.data_feeder.sections))}
            sys.stderr.write("Find unconfirmed polygons: %.2f seconds\n" % (time.time()-t))

            # Remove all unconfirmed polygons from graphicscene.
            t = time.time()
            for i in range(len(self.data_feeder.sections)):
                for p in matched_unconfirmed_polygons_to_remove[i]:
                    self.drawings[i].remove(p)
                    if i == self.active_i:
                        self.removeItem(p)
            sys.stderr.write("Remove unconfirmed polygons from graphicscene: %.2f seconds\n" % (time.time()-t))

            # Identify sections affected by new structure.
            # t = time.time()
            # sections_used = []
            # positions_rel_volResol = []
            # for sec in self.data_feeder.sections:
            #     pos_gl_volResol = DataManager.convert_section_to_z(sec=sec, downsample=volume_downsample_factor, mid=True)
            #     pos_rel_volResol = int(np.round(pos_gl_volResol - internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[2]))
            #     if pos_rel_volResol >= 0 and pos_rel_volResol < volume_volResol.shape[2]:
            #         positions_rel_volResol.append(pos_rel_volResol)
            #         sections_used.append(sec)
            # sys.stderr.write("Identify sections affected by new structure: %.2f seconds\n" % (time.time()-t))

            t = time.time()
            sections_used = []
            positions_wrt_internalStructureVolume_volResol = []
            for sec in self.data_feeder.sections:
                pos_wrt_WholebrainAlignedPadded_volResol = DataManager.convert_section_to_z(sec=sec, downsample=volume_downsample_factor, mid=True)
                pos_wrt_internalStructureVolume_volResol = int(np.round(pos_wrt_WholebrainAlignedPadded_volResol - internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[2]))
                if pos_wrt_internalStructureVolume_volResol >= 0 and pos_wrt_internalStructureVolume_volResol < volume_volResol.shape[2]:
                    positions_wrt_internalStructureVolume_volResol.append(pos_wrt_internalStructureVolume_volResol)
                    sections_used.append(sec)
            sys.stderr.write("Identify sections affected by new structure: %.2f seconds\n" % (time.time()-t))

            # Compute contours of new structure on these sections
            t = time.time()
            sample_every = max(1, int(np.floor(20./self.data_feeder.downsample)))

            gscene_pts_wrt_internalStructureVolume_volResol_allpos = find_contour_points_3d(volume_volResol, along_direction='z', sample_every= sample_every,
                                                                    positions=positions_wrt_internalStructureVolume_volResol)

            sys.stderr.write("Compute contours of new structure on these sections: %.2f seconds\n" % (time.time()-t))
            t = time.time()
            m = dict(zip(positions_wrt_internalStructureVolume_volResol, sections_used))
            gscene_pts_wrt_internalStructureVolume_volResol_allsec = {m[pos_wrt_internalStructureVolume_volResol]: pts_wrt_internalStructureVolume_volResol \
                                                for pos_wrt_internalStructureVolume_volResol, pts_wrt_internalStructureVolume_volResol \
                                                in gscene_pts_wrt_internalStructureVolume_volResol_allpos.iteritems()}
            sys.stderr.write("Compute contours of new structure on these sections 2: %.2f seconds\n" % (time.time()-t))

            t = time.time()
            for sec, gscene_pts_wrt_internalStructureVolume_volResol in gscene_pts_wrt_internalStructureVolume_volResol_allsec.iteritems():

                print sec, gscene_pts_wrt_internalStructureVolume_volResol[0]

                # if this section already has a confirmed contour, do not add a new one.
                if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                p.properties['type'] == 'confirmed' for p in self.drawings[self.data_feeder.sections.index(sec)]]):
                    continue

                try:
                    # gscene_xs_gl_vol_resol = gscene_pts_wrt_internalStructureVolume_volResol[:,0] + internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[0]
                    # gscene_ys_gl_vol_resol = gscene_pts_wrt_internalStructureVolume_volResol[:,1] + internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[1]
                    # gscene_pts_gl_vol_resol = np.c_[gscene_xs_gl_vol_resol, gscene_ys_gl_vol_resol]
                    # gscene_pts_gl_data_resol = gscene_pts_gl_vol_resol / data_vol_downsample_ratio

                    gscene_pts_wrt_internalStructureVolume_dataResol = gscene_pts_wrt_internalStructureVolume_volResol *  self.structure_volumes_downscale_factor / self.data_feeder.downsample
                    gscene_xs_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[0]
                    gscene_ys_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[1]
                    gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]

                    # t = time.time()
                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                            label=name_u, linecolor='r', vertex_radius=8, linewidth=5, section=sec,
                                                            type='intersected',
                                                            side=side,
                                                            side_manually_assigned=False)
                except Exception as e:
                    sys.stderr.write("Error adding polygon, sec %d: %s\n" % (sec, e))

                # sys.stderr.write("Add polygon and vertices: %.2f seconds.\n" % (time.time()-t))
            sys.stderr.write("Add polygons: %.2f seconds\n" % (time.time()-t))

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
                gscene_pts_wrt_internalStructureVolume_allpos_dataResol = find_contour_points_3d(volume_dataResol, along_direction='x', sample_every=1)
                # Note that find_contour_points_3d returns (z, y) where z is counted from bottom up.

                for pos_wrt_internalStructureVolume_dataResol, gscene_pts_wrt_internalStructureVolume_dataResol in gscene_pts_wrt_internalStructureVolume_allpos_dataResol.iteritems():
                    gscene_xs_wrt_dataVolume_dataResol = self.data_feeder.z_dim - 1 - (gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[2])
                    gscene_ys_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[1]
                    gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]
                    pos_wrt_dataVolume_dataResol = pos_wrt_internalStructureVolume_dataResol + internal_structure_origin_wrt_dataVolume_dataResol[0]

                    # if this position already has a confirmed contour, do not add a new one.
                    if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                    p.properties['type'] == 'confirmed' for p in self.drawings[pos_wrt_dataVolume_dataResol]]):
                        continue

                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                        label=name_u,
                                                        linecolor='g', vertex_radius=1, linewidth=2,
                                                        index=int(np.round(pos_wrt_dataVolume_dataResol)),
                                                        type='intersected',
                                                        side=side,
                                                        side_manually_assigned=False)

            elif self.data_feeder.orientation == 'horizontal':

                gscene_pts_wrt_internalStructureVolume_allpos_dataResol = find_contour_points_3d(volume_dataResol, along_direction='y', sample_every=1)
                # Note that find_contour_points_3d returns (z, x) where z is counted from bottom up.

                for pos_wrt_internalStructureVolume_dataResol, gscene_pts_wrt_internalStructureVolume_dataResol in gscene_pts_wrt_internalStructureVolume_allpos_dataResol.iteritems():
                    gscene_xs_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[0]
                    gscene_ys_wrt_dataVolume_dataResol = self.data_feeder.z_dim - 1 - (gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[2])
                    gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]
                    pos_wrt_dataVolume_dataResol = pos_wrt_internalStructureVolume_dataResol + internal_structure_origin_wrt_dataVolume_dataResol[1]

                    # if this position already has a confirmed contour, do not add a new one.
                    if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                    p.properties['type'] == 'confirmed' for p in self.drawings[pos_wrt_dataVolume_dataResol]]):
                        continue

                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                        label=name_u,
                                                        linecolor='g', vertex_radius=1, linewidth=2,
                                                        index=int(np.round(pos_wrt_dataVolume_dataResol)),
                                                        type='intersected',
                                                        side=side,
                                                        side_manually_assigned=False)

            elif self.data_feeder.orientation == 'sagittal':
                # pos means z-voxel index for sagittal

                gscene_pts_wrt_internalStructureVolume_allpos_dataResol = find_contour_points_3d(volume_dataResol, along_direction='z', sample_every=1)

                for pos_wrt_internalStructureVolume_dataResol, gscene_pts_wrt_internalStructureVolume_dataResol in gscene_pts_wrt_internalStructureVolume_allpos_dataResol.iteritems():

                    gscene_xs_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[0]
                    gscene_ys_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[1]
                    gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]
                    pos_wrt_dataVolume_dataResol = pos_wrt_internalStructureVolume_dataResol + internal_structure_origin_wrt_dataVolume_dataResol[2]

                    # if this position already has a confirmed contour, do not add a new one.
                    if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                    p.properties['type'] == 'confirmed' for p in self.drawings[pos_wrt_dataVolume_dataResol]]):
                        continue

                    self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                        label=name_u,
                                                        linecolor='g', vertex_radius=1, linewidth=2,
                                                        index=int(np.round(pos_wrt_dataVolume_dataResol)),
                                                        type='intersected',
                                                        side=side,
                                                        side_manually_assigned=False)
                    # except Exception as e:
                    #     raise e
                        # sys.stderr.write("Error adding polygon, pos %d (wrt dataVolume, dataResol): %s\n" % (pos_wrt_dataVolume_dataResol, e))


    def update_drawings_from_prob_structure_volume(self, name_u, side, levels=[0.1, 0.25, 0.5, 0.75, 0.99]):
        """
        Update drawings based on `self.prob_structure_volumes`, which is a reference to the GUI's `prob_structure_volumes`.

        Args:
            name_u (str): structure name, unsided
            side (str): L, R or S
        """

        # self.drawings = defaultdict(list) # Clear the internal variable `drawings`, and let `load_drawings` append to an empty set.

        level_to_color = {0.1: (125,0,125), 0.25: (0,255,0), 0.5: (255,0,0), 0.75: (0,125,0), 0.99: (0,0,255)}

        print "%s: Updating drawings based on prob. structure volume of %s, %s" % (self.id, name_u, side)

        volume_volResol = self.prob_structure_volumes[(name_u, side)]['volume_in_bbox']
        internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol = np.array(self.prob_structure_volumes[(name_u, side)]['bbox'])
        # internal_structure_bbox_wrt_WholebrainAlignedPaddedXYCropped_volResol = np.array(self.prob_structure_volumes[(name_u, side)]['bbox'])
        # print 'internal_structure_bbox_wrt_WholebrainAlignedPaddedXYCropped_volResol=', internal_structure_bbox_wrt_WholebrainAlignedPaddedXYCropped_volResol
        # wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_tbResol = DataManager.load_cropbox(self.gui.stack, convert_section_to_z=True)[[0,2,4]]
        # print 'wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_tbResol=', wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_tbResol
        # wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_volResol = wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_tbResol * 32. / self.prob_structure_volumes_downscale_factor
        # print 'wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_volResol=', wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_volResol
        # internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol = internal_structure_bbox_wrt_WholebrainAlignedPaddedXYCropped_volResol + wholeBrainAlignedPaddedXYCropped_origin_wrt_wholeBrainAlignedPadded_volResol[[0,0,1,1,2,2]]
        # print 'internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol=', internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol
        # # internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol = np.array(self.prob_structure_volumes[(name_u, side)]['bbox'])
        internal_structure_origin_wrt_WholebrainAlignedPadded_volResol = internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol[[0,2,4]]

        data_origin_wrt_WholebrainAlignedPadded_dataResol = self.gui.image_origin_wrt_WholebrainAlignedPadded_tbResol[self.id] * 32. / self.data_feeder.downsample
        data_origin_wrt_WholebrainAlignedPadded_volResol = data_origin_wrt_WholebrainAlignedPadded_dataResol * self.data_feeder.downsample / float(self.prob_structure_volumes_downscale_factor)
        internal_structure_origin_wrt_dataVolume_volResol = internal_structure_origin_wrt_WholebrainAlignedPadded_volResol - data_origin_wrt_WholebrainAlignedPadded_volResol
        internal_structure_origin_wrt_dataVolume_dataResol = internal_structure_origin_wrt_dataVolume_volResol * float(self.prob_structure_volumes_downscale_factor) / self.data_feeder.downsample

        print 'internal_structure_origin_wrt_dataVolume_dataResol', internal_structure_origin_wrt_dataVolume_dataResol
        print 'volume (internal vol resol)', volume_volResol.shape, internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol

        print 'self.prob_structure_volumes_downscale_factor=', self.prob_structure_volumes_downscale_factor

        volume_downsample_factor = self.prob_structure_volumes_downscale_factor
        bbox_wrt_WholebrainAlignedPadded_losslessResol = np.array(internal_structure_bbox_wrt_WholebrainAlignedPadded_volResol) * volume_downsample_factor

        data_vol_downsample_ratio = float(self.data_feeder.downsample) / volume_downsample_factor
        print 'Downsample from image data to internal volume representation =', data_vol_downsample_ratio

        if data_vol_downsample_ratio >= 1:
            volume_dataResol = volume_volResol[::int(data_vol_downsample_ratio), ::int(data_vol_downsample_ratio), ::int(data_vol_downsample_ratio)]
            bbox_wrt_WholebrainAlignedPadded_dataResol = np.array(bbox_wrt_WholebrainAlignedPadded_losslessResol) / self.data_feeder.downsample
            # xmin_ds, xmax_ds, ymin_ds, ymax_ds, zmin_ds, zmax_ds = np.array(bbox_wrt_WholebrainAlignedPadded_losslessResol) / self.data_feeder.downsample
            print 'volume (data resol)', volume_dataResol.shape, bbox_wrt_WholebrainAlignedPadded_dataResol

        if hasattr(self.data_feeder, 'sections'):

            assert self.data_feeder.orientation == 'sagittal', "Current implementation only considers sagittal sections."

            print "Removing all unconfirmed polygons..."

            t = time.time()

            for i in range(len(self.data_feeder.sections)):
                for p in self.drawings[i]:
                    assert 'label' in p.properties, "ERROR! polygon has no label i=%d, sec=%d" % (i, self.data_feeder.sections[i])

            # Find all unconfirmed polygons. These are to be removed.
            matched_unconfirmed_polygons_to_remove = {i: [p for p in self.drawings[i] \
                                                        if p.properties['label'] == name_u and \
                                                        p.properties['side'] == side and \
                                                        p.properties['type'] != 'confirmed']
                                                    for i in range(len(self.data_feeder.sections))}
            sys.stderr.write("Find unconfirmed polygons: %.2f seconds\n" % (time.time()-t))

            # Remove all unconfirmed polygons from graphicscene.
            t = time.time()
            for i in range(len(self.data_feeder.sections)):
                for p in matched_unconfirmed_polygons_to_remove[i]:
                    self.drawings[i].remove(p)
                    if i == self.active_i:
                        self.removeItem(p)
            sys.stderr.write("Remove unconfirmed polygons from graphicscene: %.2f seconds\n" % (time.time()-t))

            # Identify sections affected by new structure.
            # t = time.time()
            # sections_used = []
            # positions_rel_volResol = []
            # for sec in self.data_feeder.sections:
            #     pos_gl_volResol = DataManager.convert_section_to_z(sec=sec, downsample=volume_downsample_factor, mid=True)
            #     pos_rel_volResol = int(np.round(pos_gl_volResol - internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[2]))
            #     if pos_rel_volResol >= 0 and pos_rel_volResol < volume_volResol.shape[2]:
            #         positions_rel_volResol.append(pos_rel_volResol)
            #         sections_used.append(sec)
            # sys.stderr.write("Identify sections affected by new structure: %.2f seconds\n" % (time.time()-t))

            t = time.time()
            sections_used = []
            positions_wrt_internalStructureVolume_volResol = []
            for sec in self.data_feeder.sections:
                pos_wrt_WholebrainAlignedPadded_volResol = DataManager.convert_section_to_z(sec=sec, downsample=volume_downsample_factor, mid=True)
                pos_wrt_internalStructureVolume_volResol = int(np.round(pos_wrt_WholebrainAlignedPadded_volResol - internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[2]))
                if pos_wrt_internalStructureVolume_volResol >= 0 and pos_wrt_internalStructureVolume_volResol < volume_volResol.shape[2]:
                    positions_wrt_internalStructureVolume_volResol.append(pos_wrt_internalStructureVolume_volResol)
                    sections_used.append(sec)
            sys.stderr.write("Identify sections affected by new structure: %.2f seconds\n" % (time.time()-t))

            # Compute contours of new structure on these sections
            t = time.time()
            sample_every = max(1, int(np.floor(20./self.data_feeder.downsample)))

            for level in levels:
            # for level in [0.5]:

                gscene_pts_wrt_internalStructureVolume_volResol_allpos = find_contour_points_3d(volume_volResol >= level, along_direction='z', sample_every= sample_every,
                                                                        positions=positions_wrt_internalStructureVolume_volResol)

                sys.stderr.write("Compute contours of new structure on these sections: %.2f seconds\n" % (time.time()-t))
                t = time.time()
                m = dict(zip(positions_wrt_internalStructureVolume_volResol, sections_used))
                gscene_pts_wrt_internalStructureVolume_volResol_allsec = {m[pos_wrt_internalStructureVolume_volResol]: pts_wrt_internalStructureVolume_volResol \
                                                    for pos_wrt_internalStructureVolume_volResol, pts_wrt_internalStructureVolume_volResol \
                                                    in gscene_pts_wrt_internalStructureVolume_volResol_allpos.iteritems()}
                sys.stderr.write("Compute contours of new structure on these sections 2: %.2f seconds\n" % (time.time()-t))

                t = time.time()
                for sec, gscene_pts_wrt_internalStructureVolume_volResol in gscene_pts_wrt_internalStructureVolume_volResol_allsec.iteritems():

                    print sec, gscene_pts_wrt_internalStructureVolume_volResol[0]

                    # if this section already has a confirmed contour, do not add a new one.
                    if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                    p.properties['type'] == 'confirmed' for p in self.drawings[self.data_feeder.sections.index(sec)]]):
                        continue

                    try:
                        # gscene_xs_gl_vol_resol = gscene_pts_wrt_internalStructureVolume_volResol[:,0] + internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[0]
                        # gscene_ys_gl_vol_resol = gscene_pts_wrt_internalStructureVolume_volResol[:,1] + internal_structure_origin_wrt_WholebrainAlignedPadded_volResol[1]
                        # gscene_pts_gl_vol_resol = np.c_[gscene_xs_gl_vol_resol, gscene_ys_gl_vol_resol]
                        # gscene_pts_gl_data_resol = gscene_pts_gl_vol_resol / data_vol_downsample_ratio

                        gscene_pts_wrt_internalStructureVolume_dataResol = gscene_pts_wrt_internalStructureVolume_volResol *  self.prob_structure_volumes_downscale_factor / self.data_feeder.downsample
                        gscene_xs_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[0]
                        gscene_ys_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[1]
                        gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]

                        # t = time.time()
                        self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                                label=name_u, linecolor=level_to_color[level], vertex_radius=8,
                                                                linewidth=5, section=sec,
                                                                type='intersected',
                                                                side=side,
                                                                side_manually_assigned=False)
                    except Exception as e:
                        sys.stderr.write("Error adding polygon, sec %d: %s\n" % (sec, e))

                # sys.stderr.write("Add polygon and vertices: %.2f seconds.\n" % (time.time()-t))
            sys.stderr.write("Add polygons: %.2f seconds\n" % (time.time()-t))

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

                for level in levels:

                    gscene_pts_wrt_internalStructureVolume_allpos_dataResol = find_contour_points_3d(volume_dataResol >= level, along_direction='x', sample_every=1)
                    # Note that find_contour_points_3d returns (z, y) where z is counted from bottom up.

                    for pos_wrt_internalStructureVolume_dataResol, gscene_pts_wrt_internalStructureVolume_dataResol in gscene_pts_wrt_internalStructureVolume_allpos_dataResol.iteritems():
                        gscene_xs_wrt_dataVolume_dataResol = self.data_feeder.z_dim - 1 - (gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[2])
                        gscene_ys_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[1]
                        gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]
                        pos_wrt_dataVolume_dataResol = pos_wrt_internalStructureVolume_dataResol + internal_structure_origin_wrt_dataVolume_dataResol[0]

                        # if this position already has a confirmed contour, do not add a new one.
                        if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                        p.properties['type'] == 'confirmed' for p in self.drawings[pos_wrt_dataVolume_dataResol]]):
                            continue

                        self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                            label=name_u,
                                                            linecolor=level_to_color[level], vertex_radius=.2, vertex_color=level_to_color[level], linewidth=1,
                                                            index=int(np.round(pos_wrt_dataVolume_dataResol)),
                                                            type='intersected',
                                                            side=side,
                                                            side_manually_assigned=False)

            elif self.data_feeder.orientation == 'horizontal':

                for level in levels:
                    gscene_pts_wrt_internalStructureVolume_allpos_dataResol = find_contour_points_3d(volume_dataResol >= level, along_direction='y', sample_every=1)
                    # Note that find_contour_points_3d returns (z, x) where z is counted from bottom up.

                    for pos_wrt_internalStructureVolume_dataResol, gscene_pts_wrt_internalStructureVolume_dataResol in gscene_pts_wrt_internalStructureVolume_allpos_dataResol.iteritems():
                        gscene_xs_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[0]
                        gscene_ys_wrt_dataVolume_dataResol = self.data_feeder.z_dim - 1 - (gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[2])
                        gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]
                        pos_wrt_dataVolume_dataResol = pos_wrt_internalStructureVolume_dataResol + internal_structure_origin_wrt_dataVolume_dataResol[1]

                        # if this position already has a confirmed contour, do not add a new one.
                        if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                        p.properties['type'] == 'confirmed' for p in self.drawings[pos_wrt_dataVolume_dataResol]]):
                            continue

                        self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                            label=name_u,
                                                            linecolor=level_to_color[level], vertex_radius=.2, vertex_color=level_to_color[level], linewidth=1,
                                                            index=int(np.round(pos_wrt_dataVolume_dataResol)),
                                                            type='intersected',
                                                            side=side,
                                                            side_manually_assigned=False)

            elif self.data_feeder.orientation == 'sagittal':
                # pos means z-voxel index for sagittal

                for level in levels:

                    gscene_pts_wrt_internalStructureVolume_allpos_dataResol = find_contour_points_3d(volume_dataResol >= level, along_direction='z', sample_every=1)

                    for pos_wrt_internalStructureVolume_dataResol, gscene_pts_wrt_internalStructureVolume_dataResol in gscene_pts_wrt_internalStructureVolume_allpos_dataResol.iteritems():

                        gscene_xs_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,0] + internal_structure_origin_wrt_dataVolume_dataResol[0]
                        gscene_ys_wrt_dataVolume_dataResol = gscene_pts_wrt_internalStructureVolume_dataResol[:,1] + internal_structure_origin_wrt_dataVolume_dataResol[1]
                        gscene_pts_wrt_dataVolume_dataResol = np.c_[gscene_xs_wrt_dataVolume_dataResol, gscene_ys_wrt_dataVolume_dataResol]
                        pos_wrt_dataVolume_dataResol = pos_wrt_internalStructureVolume_dataResol + internal_structure_origin_wrt_dataVolume_dataResol[2]

                        # if this position already has a confirmed contour, do not add a new one.
                        if any([p.properties['label'] == name_u and p.properties['side'] == side and \
                        p.properties['type'] == 'confirmed' for p in self.drawings[pos_wrt_dataVolume_dataResol]]):
                            continue

                        self.add_polygon_with_circles_and_label(path=vertices_to_path(gscene_pts_wrt_dataVolume_dataResol),
                                                            label=name_u,
                                                            linecolor=level_to_color[level], vertex_radius=.2, vertex_color=level_to_color[level], linewidth=1,
                                                            index=int(np.round(pos_wrt_dataVolume_dataResol)),
                                                            type='intersected',
                                                            side=side,
                                                            side_manually_assigned=False)
                    # except Exception as e:
                    #     raise e
                        # sys.stderr.write("Error adding polygon, pos %d (wrt dataVolume, dataResol): %s\n" % (pos_wrt_dataVolume_dataResol, e))


    def update_image(self, i=None, sec=None):
        i, sec = self.get_requested_index_and_section(i=i, sec=sec)

        if self.showing_which == 'histology':
            qimage = self.data_feeder.retrieve_i(i=i)
            # histology_pixmap = QPixmap.fromImage(qimage)
            self.histology_pixmap.convertFromImage(qimage) # Keeping a global pixmap avoids creating a new pixmap every time which is tha case if using the static method QPixmap.fromImage(image)
            self.pixmapItem.setPixmap(self.histology_pixmap)
            self.pixmapItem.setVisible(True)

        elif self.showing_which == 'scoremap':
            assert self.active_polygon is not None, 'Must have an active polygon first.'
            name_u = self.active_polygon.properties['label']
            scoremap_viz_fp = DataManager.get_scoremap_viz_filepath(stack=self.gui.stack, downscale=8, section=sec, structure=name_u, detector_id=15, prep_id=2)
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

                blue_fp = DataManager.get_image_filepath_v2(stack=self.gui.stack, section=self.active_section, prep_id=2, resol='lossless', version='ChatJpeg', ext='jpg')
                qimage_blue = QImage(blue_fp)
                if self.data_feeder.downsample != 1:
                    # Downsample the image for CryoJane data, which is too large and exceeds QPixmap size limit.
                    raw_width, raw_height = (qimage_blue.width(), qimage_blue.height())
                    new_width, new_height = (raw_width / self.data_feeder.downsample, raw_height / self.data_feeder.downsample)
                    qimage_blue = qimage_blue.scaled(new_width, new_height)
                    sys.stderr.write("Downsampling image by %.2f from size (w=%d,h=%d) to (w=%d,h=%d)\n" % (self.data_feeder.downsample, raw_width, raw_height, new_width, new_height))

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


    # def set_conversion_func_section_to_z(self, func):
    #     """
    #     Set the conversion function that converts section index to voxel position.
    #     """
    #     self.convert_section_to_z = func
    #
    # def set_conversion_func_z_to_section(self, func):
    #     """
    #     Set the conversion function that converts voxel position to section index.
    #     """
    #     self.convert_z_to_section = func

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
        Load annotation contours and place drawings.

        Args:
            contours (DataFrame): row indices are random polygon_ids, and columns include vertices, edits, creator, section, type, name, side, class, label_position, time_created
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

                # type = confirmed or intersected
                # if sec >= 170 and sec < 180 and contour['type'] == 'confirmed':
                #     print sec, contour['name'], 'type', contour['type']
                if 'type' in contour and contour['type'] is not None:
                    contour_type = contour['type']
                else:
                    contour_type = None

                # class = neuron or contour
                if 'class' in contour and contour['class'] is not None:
                    contour_class = contour['class']
                else:
                    contour_class = 'contour'
                    # contour_class = None

                # p = vertices_from_polygon(path=vertices_to_path(vertices))
                # if len(p) != len(vertices):
                #     print vertices
                #     print p
                #     raise Exception("")

                self.add_polygon_with_circles_and_label(path=vertices_to_path(vertices),
                                                        label=contour['name'], label_pos=contour['label_position'],
                                                        linecolor=linecolor, vertex_color=vertex_color,
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
                        contour_entry['filename'] = metadata_cache['sections_to_filenames'][self.gui.stack][contour_entry['section']]
                    else:
                        contour_entry['voxel_position'] = idx

                contour_entries[polygon_id] = contour_entry

        return contour_entries

    def get_label_section_lookup(self):

        label_section_lookup = defaultdict(list)

        for section_index, polygons in self.drawings.iteritems():
            for p in polygons:
                if p.properties['side_manually_assigned']:
                    if p.properties['side'] is None:
                        label = p.properties['label']
                    elif p.properties['side'] == 'S':
                        label = p.properties['label']
                    elif p.properties['side'] == 'L':
                        label = convert_to_left_name(p.properties['label'])
                    elif p.properties['side'] == 'R':
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
        action_reconstructUsingUnconfirmed = myMenu.addAction("Update 3D structure (using also unconfirmed contours)")
        action_showInfo = myMenu.addAction("Show contour information")
        action_showReferences = myMenu.addAction("Show reference resources")

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

        elif selected_action == action_reconstructUsingUnconfirmed:
            assert 'side' in self.active_polygon.properties and self.active_polygon.properties['side'] is not None, 'Must specify side first.'
            self.structure_volume_updated.emit(self.active_polygon.properties['label'], self.active_polygon.properties['side'], False, True)

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
            d_voxel = DataManager.convert_section_to_z(sec=self.active_section, downsample=self.data_feeder.downsample, mid=True)
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
        contour_info_text += "Class: %(class)s\n" % {'class': self.active_polygon.properties['class']}
        contour_info_text += "Level: %(position_um).2f microns (from origin of whole brain aligned and padded volume)\n" % {'position_um': self.active_polygon.properties['position_um']}

        QMessageBox.information(self.gview, "Information", contour_info_text)

    def update_cross(self, cross_x_lossless, cross_y_lossless, cross_z_lossless, origin):
        """
        Update position of the two cross lines.
        Input coordinates are wrt whole brain aligned and padded, in lossless resolution.

        Args:
            origin (3-tuple): origin of the image data wrt whole brain aligned and padded, in lossless resolution.
        """

        print self.id, ': cross_lossless', cross_x_lossless, cross_y_lossless, cross_z_lossless

        # self.hline.setVisible(True)
        # self.vline.setVisible(True)

        self.cross_x_lossless = cross_x_lossless
        self.cross_y_lossless = cross_y_lossless
        self.cross_z_lossless = cross_z_lossless

        print 'cross_lossless', cross_x_lossless, cross_y_lossless, cross_z_lossless
        print 'origin', origin
        # print 'cross_ds', cross_x_ds, cross_y_ds, cross_z_ds

        if self.data_feeder.orientation == 'sagittal':

            cross_x_ds = (cross_x_lossless - int(origin[0])) / self.data_feeder.downsample
            cross_y_ds = (cross_y_lossless - int(origin[1])) / self.data_feeder.downsample
            cross_z_ds = (cross_z_lossless - int(origin[2])) / self.data_feeder.downsample

            self.hline.setLine(0, cross_y_ds, self.data_feeder.x_dim-1, cross_y_ds)
            self.vline.setLine(cross_x_ds, 0, cross_x_ds, self.data_feeder.y_dim-1)

            if hasattr(self.data_feeder, 'sections'):
                # sec = DataManager.convert_z_to_section(z=cross_z_ds, downsample=downsample)
                # print 'cross_z', cross_z_ds, 'sec', sec, 'reverse z', DataManager.convert_section_to_z(sec=sec, downsample=downsample)
                section_thickness_in_lossless_z = SECTION_THICKNESS / XY_PIXEL_DISTANCE_LOSSLESS
                sec = int(np.ceil(cross_z_lossless / section_thickness_in_lossless_z))
                print 'crossline has been updated to cross_z_lossless =', cross_z_lossless, ', so set section to', sec
                self.set_active_section(sec, update_crossline=False)
            else:
                self.set_active_i(cross_z_ds, update_crossline=False)

        elif self.data_feeder.orientation == 'coronal':

            cross_x_ds = (cross_x_lossless - int(origin[0])) / self.data_feeder.downsample
            cross_y_ds = (cross_y_lossless - int(origin[1])) / self.data_feeder.downsample
            cross_z_ds = (cross_z_lossless - int(origin[2])) / self.data_feeder.downsample

            self.hline.setLine(0, cross_y_ds, self.data_feeder.z_dim-1, cross_y_ds)
            self.vline.setLine(self.data_feeder.z_dim-1-cross_z_ds, 0, self.data_feeder.z_dim-1-cross_z_ds, self.data_feeder.y_dim-1)

            self.set_active_i(cross_x_ds, update_crossline=False)

        elif self.data_feeder.orientation == 'horizontal':

            cross_x_ds = (cross_x_lossless - int(origin[0])) / self.data_feeder.downsample
            cross_y_ds = (cross_y_lossless - int(origin[1])) / self.data_feeder.downsample
            cross_z_ds = (cross_z_lossless - int(origin[2])) / self.data_feeder.downsample

            self.hline.setLine(0, self.data_feeder.z_dim-1-cross_z_ds, self.data_feeder.x_dim-1, self.data_feeder.z_dim-1-cross_z_ds)
            self.vline.setLine(cross_x_ds, 0, cross_x_ds, self.data_feeder.z_dim-1)

            self.set_active_i(cross_y_ds, update_crossline=False)

    def set_active_i(self, index, emit_changed_signal=True, update_crossline=True):
        super(DrawableZoomableBrowsableGraphicsScene_ForLabeling, self).set_active_i(i=index, emit_changed_signal=emit_changed_signal)

        if update_crossline and hasattr(self, 'cross_x_lossless'):

            origin_wrt_WholebrainAlignedPadded_losslessResol = self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol() * 32.

            print 'update_crossline', update_crossline
            if hasattr(self.data_feeder, 'sections'):
                cross_depth_lossless = DataManager.convert_section_to_z(sec=self.active_section, downsample=1, mid=True)
            else:
                print 'active_i =', self.active_i, 'downsample =', self.data_feeder.downsample
                if self.data_feeder.orientation == 'sagittal':
                    cross_depth_lossless = self.active_i * self.data_feeder.downsample + origin_wrt_WholebrainAlignedPadded_losslessResol[2]
                elif self.data_feeder.orientation == 'coronal':
                    cross_depth_lossless = self.active_i * self.data_feeder.downsample + origin_wrt_WholebrainAlignedPadded_losslessResol[0]
                elif self.data_feeder.orientation == 'horizontal':
                    cross_depth_lossless = self.active_i * self.data_feeder.downsample + origin_wrt_WholebrainAlignedPadded_losslessResol[1]

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

    def get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol(self):
        """
        Get the appropriate coordinate origin for this gscene.
        The coordinate is wrt to whole brain aligned and padded, in thumbnail resolution (1/32 of raw).
        """
        return self.gui.image_origin_wrt_WholebrainAlignedPadded_tbResol[self.id]

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
                if modifiers & Qt.AltModifier:
                    self.set_mode('global_rotate3d')
                else:
                    self.set_mode('prob_rotate3d')
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

            if self.mode == 'rotate2d' or self.mode == 'rotate3d' or self.mode == 'prob_rotate3d' or self.mode == 'global_rotate3d':
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

            # if self.id == 'sagittal' or self.id == 'sagittal_tb':
            #     active_structure_center_2d_wrt_WholebrainAlignedPadded_volResol = np.array((cx_wrt_WholebrainAlignedPadded_volResol, cy_wrt_WholebrainAlignedPadded_volResol))
            #     active_structure_center_2d_wrt_WholebrainAlignedPadded_gsceneResol = active_structure_center_2d_wrt_WholebrainAlignedPadded_volResol * self.structure_volumes_downscale_factor / self.data_feeder.downsample
            #     # print 'active_structure_center_2d_wrt_WholebrainAlignedPadded_gsceneResol', active_structure_center_2d_wrt_WholebrainAlignedPadded_gsceneResol, 'self.data_feeder.downsample', self.data_feeder.downsample
            #     active_structure_center_2d_wrt_imagedata_gsceneResol = active_structure_center_2d_wrt_WholebrainAlignedPadded_gsceneResol - self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[[0,1]] * 32. / self.data_feeder.downsample
            # elif self.id == 'coronal':
            #     active_structure_center_2d_wrt_imagedata_gsceneResol = \
            #     np.array((self.data_feeder.z_dim - 1 - (cz_wrt_WholebrainAlignedPadded_volResol * self.structure_volumes_downscale_factor - self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[2] * 32.) / self.data_feeder.downsample,
            #             (cy_wrt_WholebrainAlignedPadded_volResol * self.structure_volumes_downscale_factor - self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[1] * 32.) / self.data_feeder.downsample))
            # elif self.id == 'horizontal':
            #     active_structure_center_2d_wrt_imagedata_gsceneResol = \
            #     np.array([(cx_wrt_WholebrainAlignedPadded_volResol * self.structure_volumes_downscale_factor - self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[0] * 32.)  / self.data_feeder.downsample,
            #             self.data_feeder.z_dim - 1 - (cz_wrt_WholebrainAlignedPadded_volResol * self.structure_volumes_downscale_factor - self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[2] * 32.) / self.data_feeder.downsample])
            #
            # print theta_ccwise, np.rad2deg(theta_ccwise)
            # if self.id == 'sagittal' or self.id == 'sagittal_tb':
            #     tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xy=theta_ccwise)
            # elif self.id == 'coronal':
            #     tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_yz=theta_ccwise)
            # elif self.id == 'horizontal':
            #     tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xz=-theta_ccwise)

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

            # elif self.mode == 'shift3d':

            elif self.mode == 'crossline':
                # user clicks, while in crossline mode (holding down space bar).

                gscene_y_lossless = gscene_y * self.data_feeder.downsample
                gscene_x_lossless = gscene_x * self.data_feeder.downsample

                if hasattr(self.data_feeder, 'sections'):
                    gscene_z_lossless = DataManager.convert_section_to_z(sec=self.active_section, downsample=1, mid=True)
                    # print 'section', self.active_section, 'gscene_z_lossless', gscene_z_lossless
                else:
                    gscene_z_lossless = self.active_i * self.data_feeder.downsample

                origin_wrt_WholebrainAlignedPadded_losslessResol = self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol() * 32.

                if self.data_feeder.orientation == 'sagittal':
                    cross_x_lossless = gscene_x_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[0]
                    cross_y_lossless = gscene_y_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[1]
                    cross_z_lossless = gscene_z_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[2]

                elif self.data_feeder.orientation == 'coronal':
                    cross_z_lossless = self.data_feeder.z_dim * self.data_feeder.downsample - 1 - gscene_x_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[2]
                    cross_y_lossless = gscene_y_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[1]
                    cross_x_lossless = gscene_z_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[0]

                elif self.data_feeder.orientation == 'horizontal':
                    cross_x_lossless = gscene_x_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[0]
                    cross_z_lossless = self.data_feeder.z_dim * self.data_feeder.downsample - 1 - gscene_y_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[2]
                    cross_y_lossless = gscene_z_lossless + origin_wrt_WholebrainAlignedPadded_losslessResol[1]

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
                                                        label=polygon_to_delete.properties['label'], linecolor='r', vertex_radius=8, linewidth=5,
                                                        section=polygon_to_delete.properties['section'],
                                                        type=polygon_to_delete.properties['type'],
                                                        side=polygon_to_delete.properties['side'],
                                                        side_manually_assigned=polygon_to_delete.properties['side_manually_assigned'])
                self.delete_polygon(polygon=polygon_to_delete)

            # Transform the current structure volume.
            # Notify GUI to use the new volume to update contours on all gscenes.
            elif self.mode == 'rotate3d' or self.mode == 'shift3d':

                # name_side_tuple = (self.active_polygon.properties['label'], self.active_polygon.properties['side'])
                self.transform_structure(name=self.active_polygon.properties['label'], side=self.active_polygon.properties['side'])
                self.set_mode('idle')

            elif self.mode == 'global_shift3d' or self.mode == 'global_rotate3d':

                for name, side in self.prob_structure_volumes.iterkeys():
                    self.transform_structure(name=name, side=side, prob=True)
                self.set_mode('idle')

            elif self.mode == 'prob_shift3d' or self.mode == 'prob_rotate3d':

                self.transform_structure(name=self.active_polygon.properties['label'], side=self.active_polygon.properties['side'], prob=True)
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

    def compute_rotation_center_in_2d(self):
        return np.mean(vertices_from_polygon(polygon=self.active_polygon), axis=0)

    def compute_rotation_center_in_3d(self, plane, vol, vol_origin_wrt_WholebrainAlignedPadded_volResol, vol_downscale_factor, rotation_center_2d=None):
        """
        Compute 3d coordinates of rotation center.
        For sagittal, this is the point (x of polygon centroid, y of polygon centroid, z of structure centroid)
        """

        if rotation_center_2d is None:
            rotation_center_2d = self.compute_rotation_center_in_2d()

        polygon_cx_wrt_imageData_gsceneResol, polygon_cy_wrt_imageData_gsceneResol = rotation_center_2d
        ys_wrt_structureVol_volResol, xs_wrt_structureVol_volResol, zs_wrt_structureVol_volResol = np.where(vol)

        if plane == 'sagittal':

            cx_wrt_WholebrainAlignedPadded_volResol = \
            (polygon_cx_wrt_imageData_gsceneResol * self.data_feeder.downsample +
            self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[0] * 32.) / vol_downscale_factor
            cx_wrt_structureVol_volResol = cx_wrt_WholebrainAlignedPadded_volResol - vol_origin_wrt_WholebrainAlignedPadded_volResol[0]
            print 'cx_wrt_structureVol_volResol', cx_wrt_structureVol_volResol

            cy_wrt_WholebrainAlignedPadded_volResol = \
            (polygon_cy_wrt_imageData_gsceneResol * self.data_feeder.downsample +
            self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[1] * 32.) / vol_downscale_factor
            cy_wrt_structureVol_volResol = cy_wrt_WholebrainAlignedPadded_volResol - vol_origin_wrt_WholebrainAlignedPadded_volResol[1]

            cz_wrt_structureVol_volResol = np.mean(zs_wrt_structureVol_volResol)

        elif plane == 'coronal':

            cx_wrt_structureVol_volResol = np.mean(xs_wrt_structureVol_volResol)

            cy_wrt_WholebrainAlignedPadded_volResol = \
            (polygon_cy_wrt_imageData_gsceneResol * self.data_feeder.downsample +
            self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[1] * 32.) / vol_downscale_factor
            cy_wrt_structureVol_volResol = cy_wrt_WholebrainAlignedPadded_volResol - vol_origin_wrt_WholebrainAlignedPadded_volResol[1]

            cz_wrt_WholebrainAlignedPadded_volResol = \
            ((self.data_feeder.z_dim - 1 - polygon_cx_wrt_imageData_gsceneResol) * self.data_feeder.downsample +
            self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[2] * 32.) / vol_downscale_factor
            cz_wrt_structureVol_volResol = cz_wrt_WholebrainAlignedPadded_volResol - vol_origin_wrt_WholebrainAlignedPadded_volResol[2]

        elif plane == 'horizontal':
            cx_wrt_WholebrainAlignedPadded_volResol = \
            (polygon_cx_wrt_imageData_gsceneResol * self.data_feeder.downsample +
            self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[0] * 32.) / vol_downscale_factor
            cx_wrt_structureVol_volResol = cx_wrt_WholebrainAlignedPadded_volResol - vol_origin_wrt_WholebrainAlignedPadded_volResol[0]

            cy_wrt_structureVol_volResol = np.mean(ys_wrt_structureVol_volResol)

            cz_wrt_WholebrainAlignedPadded_volResol = \
            ((self.data_feeder.z_dim - 1 - polygon_cy_wrt_imageData_gsceneResol) * self.data_feeder.downsample +
            self.get_imageData_origin_wrt_WholebrainAlignedPadded_tbResol()[2] * 32.) / vol_downscale_factor
            cz_wrt_structureVol_volResol = cz_wrt_WholebrainAlignedPadded_volResol - vol_origin_wrt_WholebrainAlignedPadded_volResol[2]

        else:
            raise

        active_structure_center_3d_wrt_structureVol_volResol = np.array([cx_wrt_structureVol_volResol, cy_wrt_structureVol_volResol, cz_wrt_structureVol_volResol])
        return active_structure_center_3d_wrt_structureVol_volResol

    def compute_translate_transform_vector(self, curr_gscene_coords, start_gscene_coords, plane, vol_downscale_factor):
        """
        Args:
            curr_gscene_coords (2-tuple of float): 2D gscene coordinate of current point
            start_gscene_coords (2-tuple of float): 2D gscene coordinate of starting point
            plane (str): sagittal, coronal or horizontal

        Returns:
            12-tuple of float: flattened 3x4 transform matrix, in internal structure volume resolution.
        """
        # shift_2d_gsceneResol = np.array((gscene_x - self.press_x_wrt_imageData_gsceneResol, gscene_y - self.press_y_wrt_imageData_gsceneResol))
        shift_2d_gsceneResol = np.array(curr_gscene_coords) - np.array(start_gscene_coords)
        shift_2d_fullResol = shift_2d_gsceneResol * self.data_feeder.downsample
        shift_2d_volResol = shift_2d_fullResol / float(vol_downscale_factor)
        print 'shift_2d_volResol', shift_2d_volResol
        if plane == 'sagittal':
            tf = affine_components_to_vector(tx=shift_2d_volResol[0],ty=shift_2d_volResol[1],tz=0)
        elif plane == 'coronal':
            tf = affine_components_to_vector(tx=0,ty=shift_2d_volResol[1],tz=-shift_2d_volResol[0])
        elif plane == 'horizontal':
            tf = affine_components_to_vector(tx=shift_2d_volResol[0],ty=0,tz=-shift_2d_volResol[1])
        else:
            raise
        return tf


    def compute_rotate_transform_vector(self, curr_gscene_coords, start_gscene_coords, center_gscene_coords, plane):
        """
        Args:
            curr_gscene_coords (2-tuple of float): 2D gscene coordinate of current point
            start_gscene_coords (2-tuple of float): 2D gscene coordinate of starting point
            center_gscene_coords (2-tuple of float): 2D gscene coordinate of the rotation center
            plane (str): sagittal, coronal or horizontal

        Returns:
            12-tuple of float: flattened 3x4 transform matrix, in internal structure volume resolution.
        """

        vec2 = np.array(curr_gscene_coords) - np.array(center_gscene_coords)
        vec1 = np.array(start_gscene_coords) - np.array(center_gscene_coords)
        theta_ccwise = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
        # print theta_ccwise, np.rad2deg(theta_ccwise)
        if plane == 'sagittal':
            tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xy=theta_ccwise)
        elif plane == 'coronal':
            tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_yz=theta_ccwise)
        elif plane == 'horizontal':
            tf = affine_components_to_vector(tx=0,ty=0,tz=0,theta_xz=-theta_ccwise)
        else:
            raise
        return tf

    def transform_structure(self, name, side, prob=False):
        """
        Args:
            name (str): Structure name, without sides.
            side (str): L, R, S
            prob (bool): If true, transform probalistic structures. Otherwise, transform regular structures.
        """

        name_side_tuple = (name, side)

        if prob:
            structure_volumes = self.prob_structure_volumes
            structure_volume_downscale_factor = self.prob_structure_volumes_downscale_factor
        else:
            structure_volumes = self.structure_volumes
            structure_volume_downscale_factor = self.structure_volumes_downscale_factor

        assert name_side_tuple in structure_volumes, \
        "`structure_volumes` does not contain %s. Need to load this structure first." % str(name_side_tuple)
        vol = structure_volumes[name_side_tuple]['volume_in_bbox']
        # bbox_wrt_WholebrainAlignedPadded_volResol = np.array(structure_volumes[name_side_tuple]['bbox'])
        vol_origin_wrt_WholebrainAlignedPadded_volResol = np.array(structure_volumes[name_side_tuple]['bbox'])[[0,2,4]]

        print 'vol', vol.shape, 'vol_origin_wrt_WholebrainAlignedPadded_volResol', vol_origin_wrt_WholebrainAlignedPadded_volResol

        if self.id == 'sagittal' or self.id == 'sagittal_tb':
            plane = 'sagittal'
        elif self.id == 'coronal':
            plane = 'coronal'
        elif self.id == 'horizontal':
            plane = 'horizontal'
        else:
            raise

        if self.mode == 'prob_rotate3d' or self.mode == 'global_rotate3d' or self.mode == 'rotate3d':

            active_structure_center_2d_wrt_imagedata_gsceneResol = self.compute_rotation_center_in_2d()

            tf = self.compute_rotate_transform_vector(curr_gscene_coords=(self.gscene_x, self.gscene_y),
            start_gscene_coords=(self.press_x_wrt_imageData_gsceneResol, self.press_y_wrt_imageData_gsceneResol),
            center_gscene_coords=active_structure_center_2d_wrt_imagedata_gsceneResol,
            plane=plane)

            center_wrt_structureVol_volResol = self.compute_rotation_center_in_3d(plane=plane, vol=vol,
                    vol_origin_wrt_WholebrainAlignedPadded_volResol=vol_origin_wrt_WholebrainAlignedPadded_volResol,
                    vol_downscale_factor=structure_volume_downscale_factor,
                    rotation_center_2d=active_structure_center_2d_wrt_imagedata_gsceneResol)

        elif self.mode == 'prob_shift3d' or self.mode == 'global_shift3d' or self.mode == 'shift3d':

            tf = self.compute_translate_transform_vector(curr_gscene_coords=(self.gscene_x, self.gscene_y),
            start_gscene_coords=(self.press_x_wrt_imageData_gsceneResol, self.press_y_wrt_imageData_gsceneResol),
            plane=plane,
            vol_downscale_factor=structure_volume_downscale_factor)

            center_wrt_structureVol_volResol = np.zeros((3,)) # This does not matter since translation does not depend on a center.

        else:
            raise

        tfed_structure_volume, tfed_structure_volume_bbox_wrt_structureVol_volResol = transform_volume_v2(vol, tf,
        centroid_m=center_wrt_structureVol_volResol,
        centroid_f=center_wrt_structureVol_volResol,
        fill_sparse=True)


        t = time.time()
        structure_volumes[name_side_tuple]['volume_in_bbox'] = tfed_structure_volume
        sys.stderr.write('transform volume: %.2f seconds.\n' % (time.time() - t))
        # print 'tfed_structure_volume_bbox_wrt_structureVol_volResol', tfed_structure_volume_bbox_wrt_structureVol_volResol

        tfed_structure_volume_bbox_wrt_WholebrainAlignedPadded_volResol = \
        np.array(tfed_structure_volume_bbox_wrt_structureVol_volResol) + vol_origin_wrt_WholebrainAlignedPadded_volResol[[0,0,1,1,2,2]]
        # print 'tfed_structure_volume.shape', tfed_structure_volume.shape, 'tfed_structure_volume_bbox_wrt_WholebrainAlignedPadded_volResol', tfed_structure_volume_bbox_wrt_WholebrainAlignedPadded_volResol
        # print 'AFTER', np.count_nonzero(tfed_structure_volume.astype(np.bool))
        structure_volumes[name_side_tuple]['bbox'] = tfed_structure_volume_bbox_wrt_WholebrainAlignedPadded_volResol

        # Append edits
        if self.mode == 'shift3d' or self.mode == 'prob_shift3d' or self.mode == 'global_shift3d':
            if 'edits' not in structure_volumes[name_side_tuple]:
                structure_volumes[name_side_tuple]['edits'] = []
            # self.structure_volumes[name_side_tuple]['edits'].append(('shift3d', tf, (cx_wrt_structureVol_volResol, cy_wrt_structureVol_volResol, cz_wrt_structureVol_volResol), (cx_wrt_structureVol_volResol, cy_wrt_structureVol_volResol, cz_wrt_structureVol_volResol)))
            edit_entry = {'username': self.gui.get_username(),
            'timestamp': datetime.now().strftime("%m%d%Y%H%M%S"),
            'type': 'shift3d',
            'transform':tf,
            'centroid_m':center_wrt_structureVol_volResol,
            'centroid_f':center_wrt_structureVol_volResol}
            structure_volumes[name_side_tuple]['edits'].append(edit_entry)

        elif self.mode == 'rotate3d' or self.mode == 'prob_rotate3d' or self.mode == 'global_rotate3d':
            if 'edits' not in structure_volumes[name_side_tuple]:
                structure_volumes[name_side_tuple]['edits'] = []
            # self.structure_volumes[name_side_tuple]['edits'].append(('rotate3d', tf, (cx_wrt_structureVol_volResol, cy_wrt_structureVol_volResol, cz_wrt_structureVol_volResol), (cx_wrt_structureVol_volResol, cy_wrt_structureVol_volResol, cz_wrt_structureVol_volResol)))
            edit_entry = {'username': self.gui.get_username(),
            'timestamp': datetime.now().strftime("%m%d%Y%H%M%S"),
            'type': 'rotate3d',
            'transform':tf,  # Note that this transform is centered at centroid_m which is equal to centroid_f.
            'centroid_m':center_wrt_structureVol_volResol,
            'centroid_f':center_wrt_structureVol_volResol}
            structure_volumes[name_side_tuple]['edits'].append(edit_entry)
        else:
            raise

        if prob:
            self.prob_structure_volume_updated.emit(name, side)
        else:
            self.structure_volume_updated.emit(name, side, False, False)
