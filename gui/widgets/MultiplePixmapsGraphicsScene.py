from PyQt4.QtCore import *
from PyQt4.QtGui import *

from custom_widgets import *
from SignalEmittingItems import *

from gui_utilities import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from data_manager import DataManager
from metadata import *

class MultiplePixmapsGraphicsScene(QGraphicsScene):
    """
    Variant that supports overlaying multiple pixmaps and adjusting opacity of each.
    """

    active_image_updated = pyqtSignal()

    def __init__(self, id, pixmap_labels, gview=None, parent=None):
        """
        Args:
            pixmap_labels (list of str): keys that specify different image groups
        """

        super(QGraphicsScene, self).__init__(parent=parent)

        self.pixmapItems = {l: QGraphicsPixmapItem() for l in pixmap_labels}
        self.data_feeders = {}

        for pm in self.pixmapItems.itervalues():
            self.addItem(pm)

        self.gview = gview
        self.id = id

        self.active_sections = None
        self.active_indices = None

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


    def set_opacity(self, pixmap_label, opacity):
        self.pixmapItems[pixmap_label].setOpacity(opacity)

    def set_data_feeder(self, feeder, pixmap_label):
        if hasattr(self, 'data_feeders') and pixmap_label in self.data_feeders and self.data_feeders[pixmap_label] == feeder:
            return
        self.data_feeders[pixmap_label] = feeder
        if hasattr(feeder, 'se'): # Only implemented for image reader, not volume resection reader.
            self.connect(feeder.se, SIGNAL("image_loaded(int)"), self.image_loaded)

    @pyqtSlot(int)
    def image_loaded(self, int):
        self.active_image_updated.emit()

    def set_active_indices(self, indices, emit_changed_signal=True):

        if indices == self.active_indices:
            return

        # old_indices = self.active_indices

        print self.id, ': Set active index to', indices, ', emit_changed_signal', emit_changed_signal

        self.active_indices = indices
        self.active_sections = {label: self.data_feeders[label].sections[i] for label, i in self.active_indices.iteritems()}
        print self.id, ': Set active section to', self.active_sections

        try:
            self.update_image()
        except Exception as e:
            sys.stderr.write('Failed to update image: %s.\n' % e)
            for label in self.pixmapItems.keys():
                self.pixmapItems[label].setVisible(False)

        if emit_changed_signal:
            self.active_image_updated.emit()

    def set_active_sections(self, sections, emit_changed_signal=True):
        """
        Args:
            sections (str or int dict): {set_name: image label} .
        """

        print self.id, ': Set active sections to', sections

        indices = {set_name: self.data_feeders[set_name].sections.index(sec) for set_name, sec in sections.iteritems()}
        self.set_active_indices(indices, emit_changed_signal=emit_changed_signal)

    def update_image(self):

        indices = self.active_indices
        # sections = self.active_sections
        # indices = {label: self.data_feeders[label].all_sections.index(sec) for label, sec in sections.iteritems()}

        for set_name, idx in indices.iteritems():
            try:
                qimage = self.data_feeders[set_name].retrieve_i(i=idx)
                pixmap = QPixmap.fromImage(qimage)
                self.pixmapItems[set_name].setPixmap(pixmap)
                self.pixmapItems[set_name].setVisible(True)
            except:
                sys.stderr.write("%s: set_name=%s, index=%s fails to show. Skip.\n" % (self.id, set_name, idx))
        # self.set_active_indices(indices)

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
