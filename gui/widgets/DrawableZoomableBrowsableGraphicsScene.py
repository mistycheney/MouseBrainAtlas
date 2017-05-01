from matplotlib.backends import qt4_compat
if qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE:
    #print 'Using PySide'
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    #print 'Using PyQt4'
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

from SignalEmittingItems import *
from ZoomableBrowsableGraphicsScene import ZoomableBrowsableGraphicsScene
from ZoomableBrowsableGraphicsSceneWithReadonlyPolygon import ZoomableBrowsableGraphicsSceneWithReadonlyPolygon
from SignalEmittingGraphicsPathItemWithVertexCircles import SignalEmittingGraphicsPathItemWithVertexCircles

from collections import defaultdict

class DrawableZoomableBrowsableGraphicsScene(ZoomableBrowsableGraphicsSceneWithReadonlyPolygon):
    """
    Extend base class by:
    - Allow user to draw polygons.
    """

    drawings_updated = pyqtSignal(object)
    polygon_completed = drawings_updated
    polygon_deleted = pyqtSignal(object, int, int)
    polygon_changed = pyqtSignal(int, int) # Notify upper parties to re-compute data for the changed polygon (e.g. binary masks)

    def __init__(self, id, gview=None, parent=None):
        super(DrawableZoomableBrowsableGraphicsScene, self).__init__(id=id, gview=gview, parent=parent)

        # self.drawings = defaultdict(list)
        self.mode = 'idle'

    def get_polygon_vertices(self, polygon_ind, section=None, index=None):
        index, _ = self.get_requested_index_and_section(i=index, sec=section)
        vertices = vertices_from_polygon(self.drawings[index][polygon_ind])
        return vertices

    def set_default_vertex_color(self, color):
        """
        color is one of r,g,b
        """
        self.default_vertex_color = color

    def set_default_vertex_radius(self, size):
        self.default_vertex_radius = size

    def set_mode(self, mode):
        if hasattr(self, 'mode'):
            print 'Mode change:', self.mode, '=>', mode

        if mode == 'add vertices consecutively':
            self.gview.setDragMode(QGraphicsView.NoDrag)
        elif mode == 'idle':
            self.gview.setDragMode(QGraphicsView.ScrollHandDrag)
        elif mode == 'delete vertices':
            self.gview.setDragMode(QGraphicsView.RubberBandDrag)

        self.mode = mode

    def add_polygon_with_circles_and_label(self, path, linecolor=None, linewidth=None, vertex_color=None, vertex_radius=None,
                                            label='unknown', section=None, label_pos=None, index=None, type=None,
                                            edit_history=[], side=None, side_manually_assigned=None,
                                            contour_id=None):

        polygon = self.add_polygon(path, color=linecolor, linewidth=linewidth, index=index, section=section)

        if vertex_color is None:
            vertex_color = self.default_vertex_color

        if vertex_radius is None:
            vertex_radius = self.default_vertex_radius

        polygon.add_circles_for_all_vertices(radius=vertex_radius, color=vertex_color)
        polygon.set_closed(True)
        return polygon


    def add_polygon(self, path=QPainterPath(), color=None, linewidth=None, section=None, index=None, z_value=50, vertex_color=None, vertex_radius=None):
        '''
        Add a polygon to a specified section.

        Args:
            path (QPainterPath): path of the polygon
            pen (QPen): pen used to draw polygon

        Returns:
            QGraphicsPathItemModified: added polygon
        '''

        if color is None:
            color = self.default_line_color

        if color == 'r':
            pen = QPen(Qt.red)
        elif color == 'g':
            pen = QPen(Qt.green)
        elif color == 'b':
            pen = QPen(Qt.blue)
        elif isinstance(color, tuple) or  isinstance(color, list):
            pen = QPen(QColor(color[0], color[1], color[2]))
        else:
            raise Exception('color argument to polygon must be r,g or b')

        if linewidth is None:
            linewidth = self.default_line_width
        pen.setWidth(linewidth)

        if vertex_radius is None:
            vertex_radius = self.default_vertex_radius

        index, section = self.get_requested_index_and_section(i=index, sec=section)

        polygon = SignalEmittingGraphicsPathItemWithVertexCircles(path, gscene=self, vertex_radius=vertex_radius)

        polygon.setPen(pen)
        polygon.setZValue(z_value)
        polygon.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
        polygon.setFlag(QGraphicsItem.ItemIsMovable, False)

        polygon.signal_emitter.press.connect(self._polygon_pressed)
        # polygon.signal_emitter.release.connect(self.polygon_release)
        polygon.signal_emitter.vertex_added.connect(self.vertex_added)
        polygon.signal_emitter.polygon_changed.connect(self.polygon_changed_callback)
        polygon.signal_emitter.polygon_completed.connect(self.polygon_completed_callbak)

        self.drawings[index].append(polygon)
        if self.id == 'user_submasks':
            print "self.drawings[index] appended at:" , polygon, index
        self.drawings_mapping[polygon] = index

        # if adding polygon to current section
        if index == self.active_i:
            print 'polygon added.'
            self.addItem(polygon)

        return polygon

    @pyqtSlot()
    def polygon_changed_callback(self):
        sending_polygon = self.sender().parent # Must get parent because sender is the signal_emitter object.
        sending_index = self.drawings_mapping[sending_polygon]
        polygon_ind = self.drawings[sending_index].index(sending_polygon)
        _, sending_sec = self.get_requested_index_and_section(i=sending_index)
        self.polygon_changed.emit(sending_sec, polygon_ind)

    @pyqtSlot(QGraphicsEllipseItemModified)
    def vertex_added(self, circle):
        pass
        # polygon = self.sender().parent
        # if polygon.index == self.active_i:
        #     pass

    @pyqtSlot()
    def polygon_completed_callbak(self):
        polygon = self.sender().parent
        self.set_mode('idle')
        self.drawings_updated.emit(polygon)

    @pyqtSlot(object)
    def _polygon_pressed(self, polygon):

        print 'polygon pressed'

        if self.mode == 'add vertices consecutively':
            # if we are adding vertices, do nothing when the click triggers a polygon.
            pass
        else:
            self.active_polygon = polygon
            print 'active polygon selected', self.active_polygon

            self.polygon_pressed.emit(polygon)

    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)

        action_newPolygon = myMenu.addAction("New polygon")
        action_deletePolygon = myMenu.addAction("Delete polygon")
        action_insertVertex = myMenu.addAction("Insert vertex")
        action_deleteVertices = myMenu.addAction("Delete vertices")

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))

        if selected_action == action_newPolygon:
            self.close_curr_polygon = False
            self.active_polygon = self.add_polygon(QPainterPath(), index=self.active_i, linewidth=10)
            self.active_polygon.set_closed(False)
            self.set_mode('add vertices consecutively')

        # elif selected_action == action_deletePolygon:
        #     self.polygon_deleted.emit(self.active_polygon, self.active_i, self.drawings[self.active_i].index(self.active_polygon))
        #     sys.stderr.write('%s: polygon_deleted signal emitted.\n' % (self.id))
        #     self.drawings[self.active_i].remove(self.active_polygon)
        #     self.removeItem(self.active_polygon)

        elif selected_action == action_insertVertex:
            self.set_mode('add vertices randomly')

        elif selected_action == action_deleteVertices:
            self.set_mode('delete vertices')

    def delete_all_polygons_one_section(self, section):
        index, _ = self.get_requested_index_and_section(sec=section)

        print "before delete", self.drawings[index]

        for polygon_index, polygon in enumerate(self.drawings[index]):
            self.drawings[index].remove(polygon)
            self.drawings_mapping.pop(polygon)
            self.removeItem(polygon)
            self.polygon_deleted.emit(polygon, index, polygon_index)
            sys.stderr.write('%s: polygon_deleted signal emitted.\n' % (self.id))
        print "after delete", self.drawings[index]

    @pyqtSlot()
    def delete_polygon(self, section=None, polygon_ind=None, index=None, polygon=None):
        if polygon is None:
            assert section is not None or index is not None
            index, section = self.get_requested_index_and_section(i=index, sec=section)
            assert polygon_ind is not None
            polygon = self.drawings[index][polygon_ind]
        else:
            index = self.drawings_mapping[polygon]
            polygon_ind = self.drawings[index].index(polygon)

        self.drawings[index].remove(polygon)
        self.drawings_mapping.pop(polygon)
        self.removeItem(polygon)

        self.polygon_deleted.emit(polygon, index, polygon_ind)
        sys.stderr.write('%s: polygon_deleted signal emitted.\n' % (self.id))

    @pyqtSlot()
    def vertex_clicked(self):
        # pass
        circle = self.sender().parent
        print 'vertex clicked:', circle

    @pyqtSlot()
    def vertex_released(self):
        # print self.sender().parent, 'released'

        clicked_vertex = self.sender().parent

        if self.mode == 'moving vertex' and self.vertex_is_moved:
            self.vertex_is_moved = False

    def eventFilter(self, obj, event):

        if event.type() == QEvent.KeyPress:
            key = event.key()

            if key == Qt.Key_Escape:
                self.set_mode('idle')
                return True

            elif (key == Qt.Key_Enter or key == Qt.Key_Return) and self.mode == 'add vertices consecutively': # CLose polygon
                first_circ = self.active_polygon.vertex_circles[0]
                first_circ.signal_emitter.press.emit(first_circ)
                return False

        elif event.type() == QEvent.GraphicsSceneMousePress:

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

            elif self.mode == 'add vertices consecutively':
                # if in add vertices mode, left mouse press means:
                # - closing a polygon, or
                # - adding a vertex

                if event.button() == Qt.LeftButton:
                    obj.mousePressEvent(event)
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

        elif event.type() == QEvent.GraphicsSceneMouseRelease:

            if self.mode == 'delete vertices':
                items_in_rubberband = self.analyze_rubberband_selection()
                print "items_in_rubberband", items_in_rubberband
                for polygon, vertex_indices in items_in_rubberband.iteritems():
                    polygon.delete_vertices(vertex_indices, merge=True)
                self.set_mode('idle')

                return True

        return super(DrawableZoomableBrowsableGraphicsScene, self).eventFilter(obj, event)

    def analyze_rubberband_selection(self):
        """
        Returns:
            dict {polygon: vertex_indices}
        """

        items_in_rubberband = self.selectedItems()

        items_involved = defaultdict(list)
        for item in items_in_rubberband:
            if isinstance(item, QGraphicsEllipseItemModified):
                items_involved[item.polygon].append(item.polygon.vertex_circles.index(item))
        items_involved.default_factory = None
        return items_involved

    def set_active_i(self, i, emit_changed_signal=True):

        for polygon in self.drawings[self.active_i]:
            self.removeItem(polygon)

        super(DrawableZoomableBrowsableGraphicsScene, self).set_active_i(i, emit_changed_signal=emit_changed_signal)

        for polygon in self.drawings[i]:
            self.addItem(polygon)
