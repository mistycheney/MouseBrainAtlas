import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

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

from SignalEmittingItems import PolygonSignalEmitter

class SignalEmittingGraphicsPathItem(QGraphicsPathItem):
    def __init__(self, path, parent=None, gscene=None):
        super(SignalEmittingGraphicsPathItem, self).__init__(path, parent=parent)
        self.setPath(path)
        self.signal_emitter = PolygonSignalEmitter(parent=self)
        self.gscene = gscene

    def mousePressEvent(self, event):
        QGraphicsPathItem.mousePressEvent(self, event)
        self.signal_emitter.press.emit(self)
