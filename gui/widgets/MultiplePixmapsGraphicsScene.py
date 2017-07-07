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

from custom_widgets import *
from SignalEmittingItems import *

from gui_utilities import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from data_manager import DataManager
from metadata import *

from annotation_utilities import *
from registration_utilities import *

PEN_WIDTH = 10

BLUE_PEN = QPen(Qt.blue)
BLUE_PEN.setWidth(PEN_WIDTH)
GREEN_PEN = QPen(Qt.green)
GREEN_PEN.setWidth(PEN_WIDTH)

CROSSLINE_PEN_WIDTH = 2
CROSSLINE_RED_PEN = QPen(Qt.red)
CROSSLINE_RED_PEN.setWidth(CROSSLINE_PEN_WIDTH)

class MultiplePixmapsGraphicsScene(QGraphicsScene):
    """
    Variant that supports overlaying multiple pixmaps and adjusting opacity of each.
    """

    active_image_updated = pyqtSignal()

    def __init__(self, id, pixmap_labels, gview=None, parent=None):
        super(QGraphicsScene, self).__init__(parent=parent)

        self.pixmapItems = {l: QGraphicsPixmapItem() for l in pixmap_labels}
        self.data_feeders = {}

        # self.pixmapItem = QGraphicsPixmapItem()
        for pm in self.pixmapItems.itervalues():
            self.addItem(pm)

        self.gview = gview
        self.id = id

        self.active_section = None
        self.active_i = None

        self.installEventFilter(self)

        # self.showing_which = 'histology'

        self.gview.setMouseTracking(False)
        self.gview.setVerticalScrollBarPolicy( Qt.ScrollBarAlwaysOff )
        self.gview.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOff )
        self.gview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.gview.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # Important! default is AnchorViewCenter.
        # self.gview.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.gview.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gview.setDragMode(QGraphicsView.ScrollHandDrag)
        # gview.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        if not hasattr(self, 'contextMenu_set') or (hasattr(self, 'contextMenu_set') and not self.contextMenu_set):
            self.gview.customContextMenuRequested.connect(self.show_context_menu)

        # self.gview.installEventFilter(self)
        self.gview.viewport().installEventFilter(self)

        # self.set_mode('idle')

    # def set_mode(self, mode):
    #     self.mode = mode

    def set_opacity(self, pixmap_label, opacity):
        self.pixmapItems[pixmap_label].setOpacity(opacity)

    def set_data_feeder(self, feeder, pixmap_label):
        if hasattr(self, 'data_feeder') and self.data_feeder == feeder:
            return

        self.data_feeders[pixmap_label] = feeder

        self.active_sections = None
        self.active_indices = None

        # if hasattr(self, 'active_i') and self.active_i is not None:
        #     self.update_image()
        #     self.active_image_updated.emit()

    def set_active_indices(self, indices, emit_changed_signal=True):

        if indices == self.active_indices:
            return

        old_indices = self.active_indices

        print self.id, ': Set active index to', indices, ', emit_changed_signal', emit_changed_signal

        self.active_indices = indices
        # if hasattr(self.data_feeder, 'sections'):
        self.active_sections = {label: self.data_feeders[label].sections[i] for label, i in self.active_indices.iteritems()}
        print self.id, ': Set active section to', self.active_sections

        try:
            self.update_image()
        except Exception as e: # if failed, do not change active_i or active_section
            raise e
            self.active_indices = old_indices
            self.active_sections = {label: self.data_feeders[label].sections[i] for label, i in old_indices.iteritems()}

        if emit_changed_signal:
            self.active_image_updated.emit()

    def set_active_section(self, sections, emit_changed_signal=True):
        raise Exception('Not implemented.')

        # if sections == self.active_sections:
        #     return
        #
        # print self.id, ': Set active section to', sections
        #
        # if hasattr(self.data_feeder, 'sections'):
        #     indices = {}
        #     for sec in sections:
        #         assert sec in self.data_feeder.all_sections, 'Section %s is not loaded.' % str(sec)
        #         i = self.data_feeder.all_sections.index(sec)
        #         indices.append(i)
        #     self.set_active_indices(indices, emit_changed_signal=emit_changed_signal)

    def update_image(self):

        indices = self.active_indices
        # sections = self.active_sections
        # indices = {label: self.data_feeders[label].all_sections.index(sec) for label, sec in sections.iteritems()}

        for label, idx in indices.iteritems():
            image = self.data_feeders[label].retrieve_i(i=idx)
            pixmap = QPixmap.fromImage(image)
            self.pixmapItems[label].setPixmap(pixmap)
            self.pixmapItems[label].setVisible(True)

        # self.set_active_indices(indices)

    def set_downsample_factor(self, downsample):
        for feeder in self.data_feeders.values():
            if feeder.downsample == downsample:
                continue
            feeder.set_downsample_factor(downsample)

        self.update_image()

    def show_next(self, cycle=False):
        indices = {}
        for pixmap_label, idx in self.active_indices.iteritems():
            if cycle:
                indices[pixmap_label] = (idx + 1) % self.data_feeders[pixmap_label].n
            else:
                indices[pixmap_label] = min(idx + 1, self.data_feeders[pixmap_label].n - 1)
        self.set_active_indices(indices)

    def show_previous(self, cycle=False):
        indices = {}
        for pixmap_label, idx in self.active_indices.iteritems():
            if cycle:
                indices[pixmap_label] = (idx - 1) % self.data_feeders[pixmap_label].n
            else:
                indices[pixmap_label] = max(idx - 1, 0)
        self.set_active_indices(indices)


    def show_context_menu(self, pos):
        pass

    def eventFilter(self, obj, event):
        # print obj.metaObject().className(), event.type()
        # http://doc.qt.io/qt-4.8/qevent.html#Type-enum

        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_BracketRight:
                self.show_next(cycle=True)
            elif key == Qt.Key_BracketLeft:
                self.show_previous(cycle=True)
            return True

        elif event.type() == QEvent.Wheel:
            # eat wheel event from gview viewport. default behavior is to trigger down scroll

            out_factor = .9
            in_factor = 1. / out_factor

            if event.delta() < 0: # negative means towards user
                self.gview.scale(out_factor, out_factor)
            else:
                self.gview.scale(in_factor, in_factor)

            return True

        return False
