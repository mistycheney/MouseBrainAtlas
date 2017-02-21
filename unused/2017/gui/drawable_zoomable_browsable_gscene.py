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

class DrawableZoomableBrowsableGraphicsScene(SimpleGraphicsScene):

    def __init__(self, id, gview=None, parent=None):
        super(DrawableZoomableBrowsableGraphicsScene, self).__init__(parent=parent)

        self.drawings = defaultdict(list)

    def add_polygon(self, path=QPainterPath(), color='r', linewidth=None, z_value=50,
                section=None, index=None):
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

        if linewidth is None:
            linewidth = self.line_width

        pen.setWidth(linewidth)

        index, section = self.get_requested_index_and_section(i=index, sec=section)

        polygon = SignalEmittingGraphicsPathItem(path, gscene=self)

        polygon.setPen(pen)
        polygon.setZValue(z_value)
        polygon.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
        polygon.setFlag(QGraphicsItem.ItemIsMovable, False)

        polygon.signal_emitter.press.connect(self.polygon_press)
        # polygon.signal_emitter.release.connect(self.polygon_release)

        self.drawings[index].append(polygon)

        # if adding polygon to current section
        if index == self.active_i:
            print 'polygon added.'
            self.addItem(polygon)

        return polygon

    @pyqtSlot(object)
    def polygon_press(self, polygon):

        print 'polygon pressed'

        self.active_polygon = polygon
        self.polygon_is_moved = False
        print 'active polygon selected', self.active_polygon
