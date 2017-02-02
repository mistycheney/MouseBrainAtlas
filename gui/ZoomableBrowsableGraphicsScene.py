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

class ZoomableBrowsableGraphicsScene(QGraphicsScene):

    active_image_updated = pyqtSignal()
    # gscene_clicked = pyqtSignal(object)

    def __init__(self, id, gview=None, parent=None):
        super(QGraphicsScene, self).__init__(parent=parent)

        self.pixmapItem = QGraphicsPixmapItem()
        self.addItem(self.pixmapItem)

        # self.gui = gui
        self.gview = gview
        self.id = id

        self.qimages = None
        self.active_section = None
        self.active_i = None
        # self.active_dataset = None

        self.installEventFilter(self)

        self.showing_which = 'histology'

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

    def get_requested_index_and_section(self, i=None, sec=None):
        if i is None and sec is None:# if index is None and section is None:
            if hasattr(self, 'active_i'):
                i = self.active_i
        elif sec is not None:
            # if section in self.data_feeder.sections:
            if sec in self.data_feeder.all_sections:
                # index = self.data_feeder.sections.index(section)
                i = self.data_feeder.all_sections.index(sec)
            else:
                raise Exception('Not implemented.')
        return i, sec

    def set_data_feeder(self, feeder):
        if hasattr(self, 'data_feeder') and self.data_feeder == feeder:
            return

        self.data_feeder = feeder

        self.active_section = None
        self.active_i = None

        # if hasattr(self, 'active_i') and self.active_i is not None:
        #     self.update_image()
        #     self.active_image_updated.emit()

    def set_active_i(self, i, emit_changed_signal=True):

        # print self.id, 'goal active_i =', i, 'current active_i =', self.active_i

        if i == self.active_i:
            return

        old_i = self.active_i

        print self.id, ': Set active index to', i, ', emit_changed_signal', emit_changed_signal

        self.active_i = i
        if hasattr(self.data_feeder, 'sections'):
            self.active_section = self.data_feeder.all_sections[self.active_i]
            print self.id, ': Set active section to', self.active_section

        try:
            self.update_image()
        except Exception as e: # if failed, do not change active_i or active_section
            sys.stderr.write('Error setting index to %d\n' % i)
            # self.active_i = old_i
            # self.active_section = self.data_feeder.all_sections[old_i]
            self.pixmapItem.setVisible(False)
            raise e

        if emit_changed_signal:
            self.active_image_updated.emit()

    def set_active_section(self, sec, emit_changed_signal=True):

        # print self.id, 'current active_section = ', self.active_section

        if sec == self.active_section:
            return

        print self.id, ': Set active section to', sec
        self.active_section = sec

        if hasattr(self.data_feeder, 'sections'):

            if sec not in self.data_feeder.all_sections:
                self.pixmapItem.setVisible(False)
                self.active_i = None
                sys.stderr.write('Section %d is not loaded.\n' % sec)
                raise Exception('Section %d is not loaded.\n' % sec)
            else:
                i = self.data_feeder.all_sections.index(sec)
                self.set_active_i(i, emit_changed_signal=emit_changed_signal)

        # self.active_section = sec

    def update_image(self, i=None, sec=None):

        if i is None:
            i = self.active_i
            assert i >= 0 and i < len(self.data_feeder.all_sections)
        elif self.data_feeder.all_sections is not None:
        # elif self.data_feeder.sections is not None:
            if sec is None:
                sec = self.active_section
            # i = self.data_feeder.sections.index(sec)
            assert sec in self.data_feeder.all_sections
            i = self.data_feeder.all_sections.index(sec)

        image = self.data_feeder.retrive_i(i=i)

        histology_pixmap = QPixmap.fromImage(image)

        self.pixmapItem.setPixmap(histology_pixmap)
        self.pixmapItem.setVisible(True)


    def set_downsample_factor(self, downsample):
        if self.data_feeder.downsample == downsample:
            return
        # if self.downsample == downsample:
        #     return
        #
        # self.downsample = downsample
        self.data_feeder.set_downsample_factor(downsample)
        self.update_image()

    def show_next(self, cycle=False):
        if cycle:
            self.set_active_i((self.active_i + 1) % self.data_feeder.n)
        else:
            self.set_active_i(min(self.active_i + 1, self.data_feeder.n - 1))

    def show_previous(self, cycle=False):
        if cycle:
            self.set_active_i((self.active_i - 1) % self.data_feeder.n)
        else:
            self.set_active_i(max(self.active_i - 1, 0))

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

        # if event.type() == QEvent.GraphicsSceneMousePress:
        #
        #     self.gscene_clicked.emit(self)

        return False


SimpleGraphicsScene = ZoomableBrowsableGraphicsScene
