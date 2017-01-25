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
from zoomable_browsable_gscene import ZoomableBrowsableGraphicsScene

from collections import defaultdict

class DrawableZoomableBrowsableGraphicsScene(ZoomableBrowsableGraphicsScene):

    polygon_pressed = pyqtSignal(object)

    def __init__(self, id, gview=None, parent=None):
        super(DrawableZoomableBrowsableGraphicsScene, self).__init__(id=id, gview=gview, parent=parent)

        self.drawings = defaultdict(list)
        self.drawings_mapping = {}

    def add_polygon(self, path=QPainterPath(), color='r', linewidth=None, section=None, index=None, z_value=50):
        '''
        Add a polygon to a specified section.

        Args:
            path (QPainterPath): path of the polygon
            pen (QPen): pen used to draw polygon

        Returns:
            QGraphicsPathItemModified: added polygon
        '''

        # if path is None:
        #     path = QPainterPath()

        if color == 'r':
            pen = QPen(Qt.red)
        elif color == 'g':
            pen = QPen(Qt.green)
        elif color == 'b':
            pen = QPen(Qt.blue)

        pen.setWidth(linewidth)

        index, section = self.get_requested_index_and_section(i=index, sec=section)

        polygon = SignalEmittingGraphicsPathItem(path, gscene=self)

        polygon.setPen(pen)
        polygon.setZValue(z_value)
        polygon.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
        polygon.setFlag(QGraphicsItem.ItemIsMovable, False)

        polygon.signal_emitter.press.connect(self._polygon_pressed)
        # polygon.signal_emitter.release.connect(self.polygon_release)

        self.drawings[index].append(polygon)

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

        old_i = self.active_i

        for polygon in self.drawings[old_i]:
            self.removeItem(polygon)

        try:
            super(DrawableZoomableBrowsableGraphicsScene, self).set_active_i(i, emit_changed_signal=emit_changed_signal)

            for polygon in self.drawings[i]:
                self.addItem(polygon)
        except Exception as e:
            sys.stderr.write('%s: Error setting index to %d\n' % (self.id, i))
            raise e
