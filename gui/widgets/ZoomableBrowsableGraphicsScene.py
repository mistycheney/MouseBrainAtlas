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

        gview.setScene(self)

        # self.gui = gui
        self.gview = gview
        self.id = id

        self.qimages = None
        self.active_section = None
        self.active_i = None
        # self.active_dataset = None

        self.installEventFilter(self)

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
        if i is None and sec is None:
            if hasattr(self, 'active_i'):
                i = self.active_i
                sec = self.data_feeder.sections[i]
        elif sec is not None:
            if sec in self.data_feeder.sections:
                i = self.data_feeder.sections.index(sec)
            else:
                raise Exception('Section %d is not valid.' % sec)
        elif i is not None:
            sec = self.data_feeder.sections[i]

        assert i is not None and sec is not None
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
            self.active_section = self.data_feeder.sections[self.active_i]
            print self.id, ': Set active section to', self.active_section

        try:
            self.update_image()
        except Exception as e: # if failed, do not change active_i or active_section
            sys.stderr.write('Error setting index to %d\n' % i)
            # self.active_i = old_i
            # self.active_section = self.data_feeder.sections[old_i]
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
            if sec not in self.data_feeder.sections:
                self.pixmapItem.setVisible(False)
                self.active_i = None
                sys.stderr.write('Section %s is not loaded.\n' % sec)
                raise Exception('Section %s is not loaded.\n' % sec)
            else:
                i = self.data_feeder.sections.index(sec)
                self.set_active_i(i, emit_changed_signal=emit_changed_signal)

        # self.active_section = sec

    def update_image(self, i=None, sec=None):

        if sec is not None:
            assert sec in self.data_feeder.sections
            i = self.data_feeder.sections.index(sec)
        elif i is None:
            assert self.active_i is not None
            i = self.active_i

        image = self.data_feeder.retrive_i(i=i)

        pixmap = QPixmap.fromImage(image)

        self.pixmapItem.setPixmap(pixmap)
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


class SimpleGraphicsScene4(ZoomableBrowsableGraphicsScene):
    """
    Variant that supports adding points.
    """

    anchor_point_added = pyqtSignal(int)

    def __init__(self, id, gview=None, parent=None):
        super(SimpleGraphicsScene4, self).__init__(id=id, gview=gview, parent=parent)
        self.anchor_circle_items = []
        self.anchor_label_items = []
        self.mode = 'idle'
        self.active_image_updated.connect(self.clear_points)

    def clear_points(self):
        # pass
        for circ in self.anchor_circle_items:
            self.removeItem(circ)
        self.anchor_circle_items = []

        for lbl in self.anchor_label_items:
            self.removeItem(lbl)
        self.anchor_label_items = []

    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)
        # action_add_anchor_point = myMenu.addAction("Add anchor point")
        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))
        # if selected_action == action_add_anchor_point:
        #     self.set_mode('add point')

    def set_mode(self, mode):
        print 'mode', self.mode, '->', mode
        if self.mode == mode:
            return
        self.mode = mode

    def eventFilter(self, obj, event):
        if event.type() == QEvent.GraphicsSceneMousePress:
            pos = event.scenePos()
            x = pos.x()
            y = pos.y()

            if self.mode == 'add point':

                radius = 5

                ellipse = QGraphicsEllipseItemModified2(-radius, -radius, 2*radius, 2*radius, scene=self)
                ellipse.setPos(x, y)
                ellipse.setPen(Qt.red)
                ellipse.setBrush(Qt.red)
                ellipse.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
                ellipse.setZValue(99)

                self.anchor_circle_items.append(ellipse)

                index = self.anchor_circle_items.index(ellipse)

                label = QGraphicsSimpleTextItem(QString(str(index)), parent=ellipse)
                label.setPos(0,0)
                label.setScale(1)
                label.setBrush(Qt.black)
                label.setZValue(50)
                self.anchor_label_items.append(label)

                self.anchor_point_added.emit(index)

        return super(SimpleGraphicsScene4, self).eventFilter(obj, event)

class SimpleGraphicsScene3(ZoomableBrowsableGraphicsScene):
    """
    Variant for sorted section gscene.
    Support context menu to specify FIRST, LAST, ANCHOR.
    Support adjusting the crop box.
    """

    # bad_status_changed = pyqtSignal(int)
    first_section_set = pyqtSignal(int)
    last_section_set = pyqtSignal(int)
    anchor_set = pyqtSignal(int)
    # move_forward_requested = pyqtSignal()
    # move_backward_requested = pyqtSignal()
    # edit_transform_requested = pyqtSignal()

    def __init__(self, id, gview=None, parent=None):
        super(SimpleGraphicsScene3, self).__init__(id=id, gview=gview, parent=parent)

        # self.bad_status_indicator = QGraphicsSimpleTextItem(QString('X'), scene=self)
        # self.bad_status_indicator.setPos(50,50)
        # self.bad_status_indicator.setScale(5)
        # self.bad_status_indicator.setVisible(False)
        # self.bad_status_indicator.setBrush(Qt.red)

        # self.first_section_indicator = QGraphicsSimpleTextItem(QString('FIRST'), scene=self)
        # self.first_section_indicator.setPos(50,50)
        # self.first_section_indicator.setScale(5)
        # self.first_section_indicator.setVisible(False)
        # self.first_section_indicator.setBrush(Qt.red)
        #
        # self.last_section_indicator = QGraphicsSimpleTextItem(QString('LAST'), scene=self)
        # self.last_section_indicator.setPos(50,50)
        # self.last_section_indicator.setScale(5)
        # self.last_section_indicator.setVisible(False)
        # self.last_section_indicator.setBrush(Qt.red)

        self.box = QGraphicsRectItem(100,100,100,100,scene=self)
        self.box.setPen(QPen(Qt.red, 5))
        # self.box.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
        self.box.setFlags(QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges)
        self.box.setVisible(False)

        self.corners = {}
        radius = 20
        for c in ['ll', 'lr', 'ul', 'ur']:
            ellipse = QGraphicsEllipseItemModified3(-radius, -radius, 2*radius, 2*radius, scene=self)
            if c == 'll':
                ellipse.setPos(200,100)
            elif c == 'lr':
                ellipse.setPos(200,200)
            elif c == 'ul':
                ellipse.setPos(100,100)
            elif c == 'ur':
                ellipse.setPos(100,200)

            ellipse.setPen(Qt.red)
            ellipse.setBrush(Qt.red)
            ellipse.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
            # ellipse.setZValue(99)
            ellipse.setVisible(False)
            ellipse.signal_emitter.moved.connect(self.corner_moved)
            ellipse.signal_emitter.pressed.connect(self.corner_pressed)
            ellipse.signal_emitter.released.connect(self.corner_released)
            self.corners[c] = ellipse

        self.serving_ellipse = None

    def corner_pressed(self, ellipse):
        self.serving_ellipse = ellipse

    def corner_released(self, ellipse):
        self.serving_ellipse = None

    def corner_moved(self, ellipse, new_x, new_y):

        if self.serving_ellipse is not None and self.serving_ellipse != ellipse:
            return

        self.serving_ellipse = ellipse

        corner_label = self.corners.keys()[self.corners.values().index(ellipse)]
        if corner_label == 'll':
            self.corners['lr'].setY(new_y)
            self.corners['ul'].setX(new_x)
        elif corner_label == 'lr':
            self.corners['ll'].setY(new_y)
            self.corners['ur'].setX(new_x)
        elif corner_label == 'ul':
            self.corners['ur'].setY(new_y)
            self.corners['ll'].setX(new_x)
        elif corner_label == 'ur':
            self.corners['ul'].setY(new_y)
            self.corners['lr'].setX(new_x)

        ul_pos = self.corners['ul'].scenePos()
        lr_pos = self.corners['lr'].scenePos()

        self.box.setRect(ul_pos.x(), ul_pos.y(), lr_pos.x()-ul_pos.x(), lr_pos.y()-ul_pos.y())

    def set_box(self, xmin, xmax, ymin, ymax):
        for c in self.corners.values():
            c.setVisible(True)
        self.corners['ul'].setPos(xmin, ymin)
        self.corners['ur'].setPos(xmax, ymin)
        self.corners['ll'].setPos(xmin, ymax)
        self.corners['lr'].setPos(xmax, ymax)

        self.box.setVisible(True)
        self.box.setRect(xmin, ymin, xmax-xmin, ymax-ymin)

    # def set_bad_sections(self, secs):
    #     self.bad_sections = secs

    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)

        # is_bad = self.active_section in self.bad_sections
        # action_setBad = myMenu.addAction("Unmark as Bad" if is_bad else "Mark as Bad")
        # action_moveForward = myMenu.addAction("Move forward")
        # action_moveBackward = myMenu.addAction("Move backward")

        box_on = self.box.isVisible()
        action_toggleBox = myMenu.addAction("Show crop box" if not box_on else 'Hide crop box')
        # action_edit_transform = myMenu.addAction("Edit transform to previous")

        action_setFirst = myMenu.addAction("Set as first")
        action_setLast = myMenu.addAction("Set as last")
        action_setAnchor = myMenu.addAction("Set as anchor")

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))

        # if selected_action == action_setBad:
        #     if is_bad:
        #         self.bad_status_indicator.setVisible(False)
        #         self.bad_status_changed.emit(0)
        #     else:
        #         self.bad_status_indicator.setVisible(True)
        #         self.bad_status_changed.emit(1)
        if selected_action == action_toggleBox:
            self.box.setVisible(not box_on)
            for el in self.corners.values():
                el.setVisible(not box_on)
        elif selected_action == action_setFirst:
            self.first_section_set.emit(self.active_section)
            # self.first_section_indicator.setVisible(True)
        elif selected_action == action_setLast:
            self.last_section_set.emit(self.active_section)
            # self.last_section = self.active_section
            # self.status_updated.emit('LAST')
            # self.last_section_indicator.setVisible(True)
        elif selected_action == action_setAnchor:
            self.anchor_set.emit(self.active_section)

        # elif selected_action == action_moveForward:
        #     self.move_forward_requested.emit()
        # elif selected_action == action_moveBackward:
        #     self.move_backward_requested.emit()
        # elif selected_action == action_edit_transform:
        #     self.edit_transform_requested.emit()

    def set_active_i(self, i, emit_changed_signal=True):
        super(SimpleGraphicsScene3, self).set_active_i(i, emit_changed_signal=True)
        # self.bad_status_indicator.setVisible(self.active_section in self.bad_sections)

        # self.first_section_indicator.setVisible(self.active_section == self.first_section)
        # self.last_section_indicator.setVisible(self.active_section == self.last_section)


class SimpleGraphicsScene2(ZoomableBrowsableGraphicsScene):
    """
    Variant for slide position gscenes.
    """

    status_updated = pyqtSignal(int, str)
    send_to_sorted_requested = pyqtSignal(int)

    def __init__(self, id, gview=None, parent=None):
        super(SimpleGraphicsScene2, self).__init__(id=id, gview=gview, parent=parent)

    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)

        setStatus_menu = QMenu("Set to", myMenu)
        myMenu.addMenu(setStatus_menu)
        action_setNormal = setStatus_menu.addAction('Normal')
        action_setRescan = setStatus_menu.addAction('Rescan')
        action_setPlaceholder = setStatus_menu.addAction('Placeholder')
        action_setNonexisting = setStatus_menu.addAction('Nonexisting')
        actions_setStatus = {action_setNormal: 'Normal', action_setRescan: 'Rescan',
                            action_setPlaceholder: 'Placeholder', action_setNonexisting: 'Nonexisting'}

        action_sendToSorted = myMenu.addAction("Send to sorted scene")

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))

        if selected_action in actions_setStatus:
            status = actions_setStatus[selected_action]
            self.status_updated.emit(self.id, status)
        elif selected_action == action_sendToSorted:
            self.send_to_sorted_requested.emit(self.id)
