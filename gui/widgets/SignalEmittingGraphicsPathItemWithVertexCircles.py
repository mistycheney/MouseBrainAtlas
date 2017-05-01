#! /usr/bin/env python

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

from SignalEmittingGraphicsPathItem import SignalEmittingGraphicsPathItem
from SignalEmittingItems import PolygonSignalEmitter, QGraphicsEllipseItemModified

from gui_utilities import *

class SignalEmittingGraphicsPathItemWithVertexCircles(SignalEmittingGraphicsPathItem):
    """
    Extends base class by:
    - Define a list of `QGraphicsEllipseItemModified` called vertex_circles.
    """

    def __init__(self, path, parent=None, gscene=None, vertex_radius=None):
        super(SignalEmittingGraphicsPathItemWithVertexCircles, self).__init__(path, parent=parent, gscene=gscene)

        self.vertex_circles = []

        if vertex_radius is None:
            self.vertex_radius = 20
        else:
            self.vertex_radius = vertex_radius

        self.closed = False

    def add_circles_for_all_vertices(self, radius=None, color='b'):
        '''
        Add vertex circles for all vertices in a polygon with existing path.

        Args:
            polygon (QGraphicsPathItemModified): the polygon
        '''

        path = self.path()
        is_closed = polygon_is_closed(path=path)

        n = polygon_num_vertices(path=path, closed=is_closed)

        for i in range(n):
            self.add_circle_for_vertex(index=i, radius=radius, color=color)

    def add_circle_for_vertex(self, index, radius=None, color='b'):
        """
        Add a circle for an existing vertex.
        """

        path = self.path()
        if index == -1:
            is_closed = polygon_is_closed(path=path)
            n = polygon_num_vertices(path=path, closed=is_closed)
            elem = path.elementAt(n-1)
        else:
            elem = path.elementAt(index)

        if radius is None:
            radius = self.vertex_radius

        ellipse = QGraphicsEllipseItemModified(-radius, -radius, 2*radius, 2*radius, polygon=self, parent=self) # set polygon as parent, so that moving polygon moves the children vertices as well
        ellipse.setPos(elem.x, elem.y)

        if color == 'r':
            ellipse.setPen(Qt.red)
            ellipse.setBrush(Qt.red)
        elif color == 'g':
            ellipse.setPen(Qt.green)
            ellipse.setBrush(Qt.green)
        elif color == 'b':
            ellipse.setPen(Qt.blue)
            ellipse.setBrush(Qt.blue)
        elif isinstance(color, tuple) or isinstance(color, list):
            ellipse.setPen(QColor(color[0], color[1], color[2]))
            ellipse.setBrush(QColor(color[0], color[1], color[2]))
        else:
            raise Exception('Input color is not recognized.')

        ellipse.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
        ellipse.setZValue(99)

        if index == -1:
            self.vertex_circles.append(ellipse)
        else:
            self.vertex_circles.insert(index, ellipse)

        self.signal_emitter.vertex_added.emit(ellipse)
        ellipse.signal_emitter.press.connect(self.vertex_press)
        return ellipse

    def add_vertex(self, x, y, new_index=-1):
        if new_index == -1:
            polygon_goto(self, x, y)
        else:
            new_path = insert_vertex(self.path())
            self.setPath(new_path)

        self.add_circle_for_vertex(new_index)

    def delete_vertices(self, indices_to_remove, merge=False):
        if merge:
            for i in indices_to_remove:
                self.gscene.removeItem(self.vertex_circles[i])
            self.vertex_circles = [c for c in self.vertex_circles if c not in indices_to_remove]

            new_path = delete_vertices_merge(self.path(), indices_to_remove)
            self.setPath(new_path)

            self.signal_emitter.polygon_changed.emit()
        else:
            paths_to_remove, paths_to_keep = split_path(polygon.path(), indices_to_remove)

    def set_closed(self, closed):
        self.closed = closed

    @pyqtSlot(object)
    def vertex_press(self, circle):

        if self.vertex_circles.index(circle) == 0 and len(self.vertex_circles) > 2 and not self.closed:
            # (self.mode == 'add vertices randomly' or self.mode == 'add vertices consecutively'):
            # the last condition is to prevent setting the flag when one clicks vertex 0 in idle mode.
            # print 'close polygon'
            self.closed = True
            self.close()

            self.signal_emitter.polygon_completed.emit()

    def close(self):
        # def close_polygon(self, polygon=None):
        path = self.path()
        path.closeSubpath()
        self.setPath(path)
