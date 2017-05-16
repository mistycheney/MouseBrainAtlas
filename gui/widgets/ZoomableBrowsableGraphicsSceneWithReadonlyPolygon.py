from matplotlib.backends import qt4_compat
if qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE:
    #print 'Using PySide'
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    #print 'Using PyQt4'
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

from SignalEmittingGraphicsPathItem import SignalEmittingGraphicsPathItem
from ZoomableBrowsableGraphicsScene import ZoomableBrowsableGraphicsScene

from collections import defaultdict

class ZoomableBrowsableGraphicsSceneWithReadonlyPolygon(ZoomableBrowsableGraphicsScene):
    """
    Read-only polygons.
    """

    polygon_pressed = pyqtSignal(object)

    def __init__(self, id, gview=None, parent=None):
        super(ZoomableBrowsableGraphicsSceneWithReadonlyPolygon, self).__init__(id=id, gview=gview, parent=parent)

        self.drawings = defaultdict(list)
        self.drawings_mapping = {}

    def set_default_line_width(self, width):
        self.default_line_width = width

    def set_default_line_color(self, color):
        """
        Args:
            color (str): "r", "g" or "b"
        """
        self.default_line_color = color

    def remove_polygon(self, polygon):
        self.polygon_deleted.emit(polygon)
        sys.stderr.write('%s: polygon_deleted signal emitted.\n' % (self.id))
        self.drawings[self.active_i].remove(polygon)
        self.removeItem(polygon)

    def remove_all_polygons(self, sec=None, i=None):
        index, _ = self.get_requested_index_and_section(sec=sec, i=i)
        for p in self.drawings[index]:
            self.remove_polygon(p)

    def add_polygon(self, path=QPainterPath(), color=None, linewidth=None, section=None, index=None, z_value=50):
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
            raise Exception('color not recognized.')

        if linewidth is None:
            linewidth = self.default_line_width

        pen.setWidth(linewidth)

        index, _ = self.get_requested_index_and_section(i=index, sec=section)

        polygon = SignalEmittingGraphicsPathItem(path, gscene=self)

        polygon.setPen(pen)
        polygon.setZValue(z_value)
        polygon.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
        polygon.setFlag(QGraphicsItem.ItemIsMovable, False)

        polygon.signal_emitter.press.connect(self._polygon_pressed)
        # polygon.signal_emitter.release.connect(self.polygon_release)

        self.drawings[index].append(polygon)
        self.drawings_mapping[polygon] = index

        # if adding polygon to current section
        if index == self.active_i:
            print 'polygon added.'
            self.addItem(polygon)

        return polygon

    @pyqtSlot(object)
    def _polygon_pressed(self, polygon):

        print 'polygon pressed'

        self.active_polygon = polygon
        print 'active polygon selected', self.active_polygon

        self.polygon_pressed.emit(polygon)

    def set_active_i(self, i, emit_changed_signal=True):

        for polygon in self.drawings[self.active_i]:
            self.removeItem(polygon)

        super(ZoomableBrowsableGraphicsSceneWithReadonlyPolygon, self).set_active_i(i, emit_changed_signal=emit_changed_signal)

        for polygon in self.drawings[i]:
            self.addItem(polygon)
