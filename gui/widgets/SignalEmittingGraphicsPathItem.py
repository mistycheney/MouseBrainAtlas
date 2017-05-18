import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from SignalEmittingItems import PolygonSignalEmitter

class SignalEmittingGraphicsPathItem(QGraphicsPathItem):
    """
    Extend base class by
    - adding a member dict called `properties` to store application-specific data.
    """
    def __init__(self, path, parent=None, gscene=None):
        super(SignalEmittingGraphicsPathItem, self).__init__(path, parent=parent)
        self.setPath(path)
        self.signal_emitter = PolygonSignalEmitter(parent=self)
        self.gscene = gscene

        self.properties = {}

    def set_properties(self, property_name, property_value):
        if property_name not in self.properties or self.properties[property_name] != property_value:
            self.properties[property_name] = property_value
            sys.stderr.write(property_name + " is set.\n")
            self.signal_emitter.property_changed.emit(property_name, property_value)

    def mousePressEvent(self, event):
        QGraphicsPathItem.mousePressEvent(self, event)
        self.signal_emitter.press.emit(self)
