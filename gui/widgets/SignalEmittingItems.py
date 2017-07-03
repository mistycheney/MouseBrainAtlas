#! /usr/bin/env python

import sys
import os
import numpy as np
import time

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from enum import Enum
from gui_utilities import *

from datetime import datetime

class QGraphicsPathItemModified(QGraphicsPathItem):

    # def __init__(self, path, parent=None, gscene=None, orientation=None, position=None, vertex_radius=None):
    def __init__(self, path, parent=None, gscene=None, orientation=None, position=None, index=None, vertex_radius=None):

        super(self.__class__, self).__init__(path, parent=parent)

        self.setPath(path)

        self.signal_emitter = PolygonSignalEmitter(parent=self)

        self.gscene = gscene

        self.vertex_circles = []
        self.closed = False

        self.orientation = orientation
        self.position = position

        self.index = index

        if vertex_radius is None:
            self.vertex_radius = 20
        else:
            self.vertex_radius = vertex_radius

        self.type = None
        self.edit_history = []

        self.side = None

        # self.endorsers = set([])
        # self.creator = None

        # if section is None:
        #     self.section = section

        # SignalEmitter.vertex_added = pyqtSignal(object)

        # self.o.vertex_added = pyqtSignal(object)
        # self.o.pressed = pyqtSignal()

    def set_contour_id(self, contour_id):
        self.contour_id = contour_id

    def set_edit_history(self, edit_history):
        self.edit_history = edit_history

    def add_edit(self, editor):
        self.edit_history.append({'username': editor, 'timestamp': datetime.now().strftime("%m%d%Y%H%M%S")})

    # def set_creator(self, creator):
    #     if self.creator is not None:
    #         sys.stderr.write('Creator has been set for polygon, ignored.\n')
    #     else:
    #         self.creator = creator
    #
    # def set_endorsers(self, endorsers):
    #     self.endorsers = endorsers
    #
    # def add_endorser(self, endorser):
    #     self.endorsers.add(endorser)

    def set_side(self, side, side_manually_assigned):
        self.side = side
        self.side_manually_assigned = side_manually_assigned

    def set_type(self, t):

        if self.type == 'interpolated' and t is None:
            curr_pen = self.pen()
            curr_pen.setColor(Qt.red)
            self.setPen(curr_pen)
        elif self.type == None and t == 'interpolated':
            curr_pen = self.pen()
            curr_pen.setColor(Qt.green)
            self.setPen(curr_pen)
        self.type = t

    def set_label(self, label, label_pos=None):

        if not hasattr(self, 'label'):

            self.label = label
            self.label_textItem = QGraphicsSimpleTextItem(QString(label), parent=self)

            if label_pos is None:
                centroid = np.mean([(v.scenePos().x(), v.scenePos().y()) for v in self.vertex_circles], axis=0)
                self.label_textItem.setPos(centroid[0], centroid[1])
            else:
                self.label_textItem.setPos(label_pos[0], label_pos[1])
            self.label_textItem.setScale(1.5)
            self.label_textItem.setFlags(QGraphicsItem.ItemIgnoresTransformations | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
            self.label_textItem.setZValue(99)
            # self.accepted_proposals_allSections[sec][polygon]['labelTextArtist'] = textItem

            # self.history_allSections[sec].append({
            #     'type': 'set_label',
            #     'polygon': polygon,
            #     'label': label
            #     })

        else:
            self.label = label
            self.label_textItem.setText(label)

        self.signal_emitter.label_added.emit(self.label_textItem)


    # def open_label_selection_dialog(self):
    #
    #     if len(self.recent_labels) > 0:
    #         tuples_recent = [(abbr, fullname) for abbr, fullname in self.structure_names.iteritems() if abbr in self.recent_labels]
    #         tuples_nonrecent = [(abbr, fullname) for abbr, fullname in self.structure_names.iteritems() if abbr not in self.recent_labels]
    #         self.structure_names = OrderedDict(tuples_recent, tuples_nonrecent)
    #
    #     self.label_selection_dialog = AutoCompleteInputDialog(parent=self, labels=[abbr + ' (' + fullname + ')' for abbr, fullname in self.structure_names.iteritems()])
    #     self.label_selection_dialog.setWindowTitle('Select Structure Name')
    #
    #     # if hasattr(self, 'invalid_labelname'):
    #     #	 print 'invalid_labelname', self.invalid_labelname
    #     # else:
    #     #	 print 'no labelname set'
    #
    #     if hasattr(self, 'label'):
    #         abbr_unsided = self.label
    #         # if '_' in abbr: # if side has been set
    #         #     abbr = abbr[:-2]
    #         # abbr_unsided = abbr[:-2] if '_L' in abbr or '_R' in abbr else abbr
    #
    #         self.label_selection_dialog.comboBox.setEditText( abbr_unsided + ' (' + self.structure_names[abbr_unsided] + ')')
    #
    #     # else:
    #     #     self.label = ''
    #
    #     self.label_selection_dialog.set_okay_callback(self.label_dialog_text_changed)
    #
    #     # self.label_selection_dialog.accepted.connect(self.label_dialog_text_changed)
    #     # self.label_selection_dialog.textValueSelected.connect(self.label_dialog_text_changed)
    #
    #     self.label_selection_dialog.exec_()
    #
    #     # choose left or right side
    #
    #     # self.left_right_selection_dialog = QInputDialog(self)
    #     # self.left_right_selection_dialog.setLabelText('Enter L or R, or leave blank for single structure')
    #     #
    #     # if self.selected_section < (self.first_sec + self.last_sec)/2:
    #     #     self.left_right_selection_dialog.setTextValue(QString('L'))
    #     # else:
    #     #     self.left_right_selection_dialog.setTextValue(QString('R'))
    #     #
    #     # self.left_right_selection_dialog.exec_()
    #     #
    #     # left_right = str(self.left_right_selection_dialog.textValue())
    #     #
    #     # if left_right == 'L' or left_right == 'R':
    #     #     abbr = self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label']
    #     #     abbr_sided = abbr + '_' + left_right
    #     #     self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label'] = abbr_sided
    #     #     self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'].setText(abbr_sided)


    # def label_dialog_text_changed(self):
    #
    #     print 'label_dialog_text_changed'
    #
    #     text = str(self.label_selection_dialog.comboBox.currentText())
    #
    #     import re
    #     m = re.match('^(.+?)\s*\((.+)\)$', text)
    #
    #     if m is None:
    #         QMessageBox.warning(self, 'oops', 'structure name must be of the form "abbreviation (full description)"')
    #         return
    #
    #     else:
    #         abbr, fullname = m.groups()
    #         if not (abbr in self.structure_names.keys() and fullname in self.structure_names.values()):  # new label
    #             if abbr in self.structure_names:
    #                 QMessageBox.warning(self, 'oops', 'structure with abbreviation %s already exists: %s' % (abbr, fullname))
    #                 return
    #             else:
    #                 self.structure_names[abbr] = fullname
    #                 self.new_labelnames[abbr] = fullname
    #
    #     print self.accepted_proposals_allSections.keys()
    #     print self.selected_section
    #
    #     self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label'] = abbr
    #
    #     if 'labelTextArtist' in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon] and \
    #             self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'] is not None:
    #         # label exists
    #         self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'].setText(abbr)
    #     else:
    #         # label not exist, create
    #         self.add_label_to_polygon(self.selected_polygon, abbr)
    #
    #     self.recent_labels.insert(0, abbr)
    #
    #     self.label_selection_dialog.accept()

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

        # ellipse.signal_emitter.clicked.connect(self.vertex_clicked)

        # self.auto_extend_view(x, y)

        # self.map_vertex_to_polygon[ellipse] = polygon
        #
        # self.history_allSections[sec].append({
        #     'type': 'add_vertex',
        #     'polygon': polygon,
        #     'new_index': new_index if new_index != -1 else len(self.accepted_proposals_allSections[sec][polygon]['vertexCircles'])-1,
        #     'pos': (x,y)
        #     })

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
    		new_path = delete_vertices_merge(self.path(), indices_to_remove)

    		# self.history_allSections[self.selected_section].append({
    		# 	'type': 'delete_vertices_merge',
    		# 	'polygon': polygon,
    		# 	'new_polygon': new_polygon,
    		# 	'indices_to_remove': indices_to_remove,
    		# 	'label': self.accepted_proposals_allSections[self.selected_section][new_polygon]['label']
    		# 	})

    	else:
    		paths_to_remove, paths_to_keep = split_path(polygon.path(), indices_to_remove)

    		# new_polygons = []
    		# for path in paths_to_keep:
    		# 	new_polygon = self.add_polygon_by_vertices_label(path, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon]['label'])
    		# 	new_polygons.append(new_polygon)
            #
    		# self.remove_polygon(polygon)

    		# self.history_allSections[self.selected_section].append({
    		# 	'type': 'delete_vertices_split',
    		# 	'polygon': polygon,
    		# 	'new_polygons': new_polygons,
    		# 	'indices_to_remove': indices_to_remove,
    		# 	'label': self.accepted_proposals_allSections[self.selected_section][new_polygons[0]]['label']
    		# 	})


    def set_closed(self, closed):
        self.closed = closed


    @pyqtSlot(object)
    def vertex_press(self, circle):

        # circle = self.sender().parent

        if self.vertex_circles.index(circle) == 0 and len(self.vertex_circles) > 2 and not self.closed:
            # (self.mode == 'add vertices randomly' or self.mode == 'add vertices consecutively'):
            # the last condition is to prevent setting the flag when one clicks vertex 0 in idle mode.
            # print 'close polygon'
            self.closed = True
            self.close()

            self.signal_emitter.evoke_label_selection.emit()
            self.signal_emitter.polygon_completed.emit()

    def close(self):
        # def close_polygon(self, polygon=None):
        path = self.path()
        path.closeSubpath()
        self.setPath(path)

    # def itemChange(self, change, val):
    #     # print change
    #     if change == QGraphicsItem.ItemPositionChange:
    #         old_pos = self.pos()
    #         offset_x = val.x() - old_pos.x()
    #         offset_y = val.y() - old_pos.y()
    #
    #         # move all vertices together with the polygon
    #         # for circ in self.vertex_circles:
    #         #     circ.translate(offset_x, offset_y)
    #
    #     return val

    def mousePressEvent(self, event):

        # self.press_scene_x = event.scenePos().x()
        # self.press_scene_y = event.scenePos().y()

        # self.center_scene_x_before_move = self.scenePos().x()
        # self.center_scene_y_before_move = self.scenePos().y()

        # self.gui.active_polygon = self

        QGraphicsPathItem.mousePressEvent(self, event)
        self.signal_emitter.press.emit(self)

        # # label position
        # if 'labelTextArtist' in self.gui.polygonElements[self.gui.active_section][self.gui.active_polygon]:
        #     label_pos_before_move = self.gui.polygonElements[self.gui.active_section][self.gui.active_polygon]['labelTextArtist'].scenePos()
        #     self.label_pos_before_move_x = label_pos_before_move.x()
        #     self.label_pos_before_move_y = label_pos_before_move.y()
        #
        # if hasattr(self, 'label_textItem'):
        #     label_pos_before_move = self.label_textItem.scenePos()
        #     self.label_pos_before_move_x = label_pos_before_move.x()
        #     self.label_pos_before_move_y = label_pos_before_move.y()

    # def mouseReleaseEvent(self, event):
        # pass

        # print self, 'received mouseReleaseEvent'
        #
        # release_scene_pos = event.scenePos()
        # self.release_scene_x = release_scene_pos.x()
        # self.release_scene_y = release_scene_pos.y()
        #
        # QGraphicsPathItem.mouseReleaseEvent(self, event)
        # self.signal_emitter.released.emit()
        #
        # self.press_scene_x = None
        # self.press_scene_y = None
        #
        # self.center_scene_x_before_move = None
        # self.center_scene_y_before_move = None

    # def mouseMoveEvent(self, event):
    #     print self, 'received mouseMoveEvent'
    #     self.signal_emitter.moved.emit(event.scenePos().x(), event.scenePos().y(), self.press_scene_x, self.press_scene_y)
    #
    #     if not self.gui.mode == Mode.IDLE:
    #         QGraphicsPathItem.mouseMoveEvent(self, event)


class QGraphicsEllipseItemModified2(QGraphicsEllipseItem):
    """
    This variant has no polygon associated with it.
    """

    def __init__(self, x, y, w, h, parent=None, scene=None):
        super(self.__class__, self).__init__(x,y,w,h, parent=parent, scene=scene)
        self.signal_emitter = VertexSignalEmitter(parent=self)

    def itemChange(self, change, val):
        # print change
        if change == QGraphicsItem.ItemPositionChange:
            old_pos = self.scenePos()
            # print 'old', old_pos.x(), old_pos.y()
            new_x = val.toPoint().x()
            new_y = val.toPoint().y()
            # print 'new', new_x, new_y

        return val
        # return val

    def mousePressEvent(self, event):

        # print self, 'received mousePressEvent'
        QGraphicsEllipseItem.mousePressEvent(self, event)
        self.signal_emitter.press.emit(self)

class QGraphicsEllipseItemModified3(QGraphicsEllipseItem):
    """
    Basic QGraphicsEllipseItem that supports emitting MOVED signal.
    """

    def __init__(self, x, y, w, h, parent=None, scene=None):
        super(self.__class__, self).__init__(x,y,w,h, parent=parent, scene=scene)
        self.signal_emitter = VertexSignalEmitter(parent=self)

    def itemChange(self, change, val):
        # print change
        if change == QGraphicsItem.ItemPositionChange:
            # old_pos = self.scenePos()
            # print 'old', old_pos.x(), old_pos.y()
            new_x = val.toPoint().x()
            new_y = val.toPoint().y()
            # print 'new', new_x, new_y
            self.signal_emitter.moved.emit(self, new_x, new_y)

        return val

    def mousePressEvent(self, event):
        QGraphicsEllipseItem.mousePressEvent(self, event)
        self.signal_emitter.pressed.emit(self)

    def mouseReleaseEvent(self, event):
        QGraphicsEllipseItem.mouseReleaseEvent(self, event)
        self.signal_emitter.released.emit(self)

class QGraphicsEllipseItemModified(QGraphicsEllipseItem):
    """
    Extend base class by:
    - emit pressed and moved signals
    - define an associated polygon
    """

    def __init__(self, x, y, w, h, parent=None, polygon=None):
        super(self.__class__, self).__init__(x,y,w,h, parent=parent)
        self.signal_emitter = VertexSignalEmitter(parent=self)

        # self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, 1)
        # self.setFlag(QGraphicsItem.ItemIsMovable, 0)
        # self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, 1)

        self.polygon = polygon # This is the same as parent. Consider merge two variables.
        # self.polygon.vertex_circles.append(self)

        # moved = pyqtSignal(int, int, int, int)
        # pressed = pyqtSignal()
        # released = pyqtSignal()

    def itemChange(self, change, val):
        # print change

        if change == QGraphicsItem.ItemPositionChange:
            # old_pos = self.scenePos()
            # print 'old', old_pos.x(), old_pos.y()
            new_x = val.toPoint().x()
            new_y = val.toPoint().y()
            # print 'new', new_x, new_y
            self.signal_emitter.moved.emit(self, new_x, new_y)

        # if change == QGraphicsItem.ItemPositionChange:
        #     old_pos = self.scenePos()
        #     # print 'old', old_pos.x(), old_pos.y()
        #     new_x = val.toPoint().x()
        #     new_y = val.toPoint().y()
        #     # print 'new', new_x, new_y
        #
        #     if self in self.polygon.vertex_circles:
        #         # When circle is just created, itemChange will be called, but it is not added to the list yet.
        #
        #         vertex_index = self.polygon.vertex_circles.index(self)
        #         # print vertex_index
        #
        #         new_path = self.polygon.path()
        #
        #         if vertex_index == 0 and polygon_is_closed(path=new_path): # closed
        #             new_path.setElementPositionAt(0, new_x, new_y)
        #             new_path.setElementPositionAt(len(self.polygon.vertex_circles), new_x, new_y)
        #         else:
        #             new_path.setElementPositionAt(vertex_index, new_x, new_y)
        #
        #         self.polygon.setPath(new_path)

        #
        # elif change == QGraphicsItem.ItemPositionHasChanged:
        #     print 'has', val

        return val
        # return val

    def mousePressEvent(self, event):

        # print self, 'received mousePressEvent'
        QGraphicsEllipseItem.mousePressEvent(self, event)
        self.signal_emitter.press.emit(self)

        # self.press_scene_x = event.scenePos().x()
        # self.press_scene_y = event.scenePos().y()
        #
        # self.center_scene_x_before_move = self.scenePos().x()
        # self.center_scene_y_before_move = self.scenePos().y()
        #
        # self.gui.active_vertex = self
        # self.gui.active_polygon = polygon

    # def mouseReleaseEvent(self, event):
    #     QGraphicsEllipseItem.mouseReleaseEvent(self, event)
    #     pass

        # print self, 'received mouseReleaseEvent'
        #
        # release_scene_pos = event.scenePos()
        # self.release_scene_x = release_scene_pos.x()
        # self.release_scene_y = release_scene_pos.y()
        #
        # QGraphicsEllipseItem.mouseReleaseEvent(self, event)
        # self.signal_emitter.released.emit()
        #
        # self.press_scene_x = None
        # self.press_scene_y = None
        #
        # self.center_scene_x_before_move = None
        # self.center_scene_y_before_move = None

    # def mouseMoveEvent(self, event):
    #     # print self, 'received mouseMoveEvent'
    #     self.signal_emitter.moved.emit(event.scenePos().x(), event.scenePos().y(), self.press_scene_x, self.press_scene_y)
    #     QGraphicsEllipseItem.mouseMoveEvent(self, event)


class PolygonSignalEmitter(QObject):
# http://pyqt.sourceforge.net/Docs/PyQt4/new_style_signals_slots.html
# http://stackoverflow.com/a/12638536

    # moved = pyqtSignal(int, int, int, int)
    press = pyqtSignal(object)
    release = pyqtSignal()
    vertex_added = pyqtSignal(object)
    polygon_changed = pyqtSignal()
    # vertex_deleted = pyqtSignal(object)
    evoke_label_selection = pyqtSignal()
    label_added = pyqtSignal(object)
    # polygon_closed = pyqtSignal([], [object])
    polygon_completed = pyqtSignal()
    property_changed = pyqtSignal(str, object)

    def __init__(self, parent):
        super(self.__class__, self).__init__()
        self.parent = parent

class VertexSignalEmitter(QObject):

    # moved = pyqtSignal(int, int, int, int)
    press = pyqtSignal(object)
    release = pyqtSignal()
    pressed = pyqtSignal(object)
    released = pyqtSignal(object)
    moved = pyqtSignal(object, int, int)

    def __init__(self, parent):
        super(self.__class__, self).__init__()
        self.parent = parent
