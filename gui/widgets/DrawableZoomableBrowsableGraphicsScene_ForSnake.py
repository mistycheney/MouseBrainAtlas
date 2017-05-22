import sys, os

from PyQt4.QtCore import *
from PyQt4.QtGui import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from data_manager import DataManager
from metadata import *
from gui_utilities import *
from registration_utilities import find_contour_points
from annotation_utilities import contours_to_mask, get_interpolated_contours

# from DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene
from widgets.DrawableZoomableBrowsableGraphicsScene_ForMasking import DrawableZoomableBrowsableGraphicsScene_ForMasking

class DrawableZoomableBrowsableGraphicsScene_ForSnake(DrawableZoomableBrowsableGraphicsScene_ForMasking):
    """
    Extends base class by:
    - define a dict of special polygon called init_snake_contour
    - define a flag is_adding_snake_contour, so we know a polygon_completed signal is for
    """

    # submask_decision_updated = pyqtSignal(int)
    submask_decision_updated = pyqtSignal(int, int, bool)

    def __init__(self, id, gview=None, parent=None):

        super(DrawableZoomableBrowsableGraphicsScene_ForSnake, self).__init__(id=id, gview=gview, parent=parent)

        # self.polygon_completed.connect(self._submask_added)
        # self.polygon_deleted.connect(self._submask_deleted)

        self.init_snake_contour_polygons = {}
        self.is_adding_snake_contour = False
        self.anchor_sections = set([])

    def _submask_added(self, polygon):
        if self.is_adding_snake_contour:
            self.init_snake_contour_polygons[self.active_section] = polygon
            self.set_current_section_as_anchor()
            self.is_adding_snake_contour = False
        else:
            super(DrawableZoomableBrowsableGraphicsScene_ForSnake, self)._submask_added(polygon)

    def _submask_deleted(self, polygon, index, polygon_index):
        if self.is_adding_snake_contour:
            pass
        else:
            super(DrawableZoomableBrowsableGraphicsScene_ForSnake, self)._submask_deleted(polygon, index, polygon_index)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Space:
                self.accept_all_submasks()
                return True
            elif key == Qt.Key_Comma:
                self.copy_init_snake_contour_from_previous_section()
                return True
            elif key == Qt.Key_Period:
                self.copy_init_snake_contour_from_next_section()
                return True

        return super(DrawableZoomableBrowsableGraphicsScene_ForMasking, self).eventFilter(obj, event)

    def copy_init_snake_contour_from_previous_section(self):
        self.copy_init_snake_contour(from_index=self.active_i-1, to_index=self.active_i)

    def copy_init_snake_contour_from_next_section(self):
        self.copy_init_snake_contour(from_index=self.active_i+1, to_index=self.active_i)

    def unset_init_snake_contour(self, section=None, index=None):
        try:
            index, section = self.get_requested_index_and_section(sec=section, i=index)
        except:
            return
        # Remove from gscene's record
        if section in self.init_snake_contour_polygons:
            init_snake_contour_polygon_to_delete = self.init_snake_contour_polygons[section]
            self.delete_polygon(polygon=init_snake_contour_polygon_to_delete)
            self.init_snake_contour_polygons.pop(section, None)

    def set_init_snake_contour(self, vertices, section=None, index=None):
        try:
            index, section = self.get_requested_index_and_section(sec=section, i=index)
        except:
            return
        self.init_snake_contour_polygons[section] = \
        self.add_polygon_with_circles(path=vertices_to_path(vertices), index=index, linewidth=1, linecolor='b', vertex_radius=5)

    def copy_init_snake_contour(self, to_section=None, to_index=None, from_section=None, from_index=None):
        try:
            from_index, from_section = self.get_requested_index_and_section(sec=from_section, i=from_index)
        except:
            return

        assert from_section in self.init_snake_contour_polygons and self.init_snake_contour_polygons[from_section] is not None
        vertices = vertices_from_polygon(self.init_snake_contour_polygons[from_section])
        self.unset_init_snake_contour(section=to_section, index=to_index)
        self.set_init_snake_contour(vertices=vertices, section=to_section, index=to_index)

    def set_section_as_anchor(self, section):
        self.anchor_sections.add(section)
        print "Anchor sections:", sorted(list(self.anchor_sections))

    def set_current_section_as_anchor(self):
        self.set_section_as_anchor(section=self.active_section)

    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)

        action_newPolygon = myMenu.addAction("New polygon")
        action_deletePolygon = myMenu.addAction("Delete polygon")
        action_insertVertex = myMenu.addAction("Insert vertex")
        action_deleteVertices = myMenu.addAction("Delete vertices")
        action_addInitSnakeContour = myMenu.addAction("Create initial snake contour")
        action_deleteInitSnakeContour = myMenu.addAction("Delete initial snake contour")
        myMenu.addSeparator()
        action_setAsAnchorContour = myMenu.addAction("Set this section as anchor contour")
        action_autoEstimateAllContours = myMenu.addAction("Automatically estimate all contours")

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))

        if selected_action == action_newPolygon:
            self.close_curr_polygon = False
            self.active_polygon = self.add_polygon(QPainterPath(), index=self.active_i, linewidth=10)
            self.active_polygon.set_closed(False)
            self.set_mode('add vertices consecutively')

        elif selected_action == action_deletePolygon:
            self.polygon_deleted.emit(self.active_polygon)
            sys.stderr.write('%s: polygon_deleted signal emitted.\n' % (self.id))
            self.drawings[self.active_i].remove(self.active_polygon)
            self.removeItem(self.active_polygon)

        elif selected_action == action_insertVertex:
            self.set_mode('add vertices randomly')

        elif selected_action == action_deleteVertices:
            self.set_mode('delete vertices')

        elif selected_action == action_addInitSnakeContour:
            self.is_adding_snake_contour = True
            self.close_curr_polygon = False
            self.active_polygon = self.add_polygon(QPainterPath(), index=self.active_i, linewidth=1, color=(255,0,255), vertex_radius=5)
            self.active_polygon.set_closed(False)
            self.set_mode('add vertices consecutively')

        elif selected_action == action_deleteInitSnakeContour:
            self.polygon_deleted.emit(self.active_polygon)
            sys.stderr.write('%s: polygon_deleted signal emitted.\n' % (self.id))
            self.drawings[self.active_i].remove(self.active_polygon)
            self.removeItem(self.active_polygon)

        elif selected_action == action_setAsAnchorContour:
            self.set_current_section_as_anchor()

        elif selected_action == action_autoEstimateAllContours:
            self.anchor_sections_sorted = sorted(list(self.anchor_sections))
            for sec in range(self.data_feeder.sections[0], self.anchor_sections_sorted[0]):
                self.copy_init_snake_contour(from_section=self.anchor_sections_sorted[0], to_section=sec)
            for sec in range(self.anchor_sections_sorted[-1], self.data_feeder.sections[-1]+1):
                self.copy_init_snake_contour(from_section=self.anchor_sections_sorted[-1], to_section=sec)
            contours_grouped_by_pos = {anchor_sec: vertices_from_polygon(self.init_snake_contour_polygons[anchor_sec])
                                        for anchor_sec in self.anchor_sections}

            contours_all_sections = get_interpolated_contours(contours_grouped_by_pos, len_interval=10)

            for sec, contours in contours_all_sections.iteritems():
                self.unset_init_snake_contour(section=sec)
                self.set_init_snake_contour(vertices=contours, section=sec)
