#! /usr/bin/env python

import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

import sys
import os
import numpy as np

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

import time

from enum import Enum

class Mode(Enum):
	REVIEW_PROPOSAL = 'review proposal'
	IDLE = 'idle'
	MOVING_POLYGON = 'moving polygon'
	MOVING_VERTEX = 'moving vertex'
	CREATING_NEW_POLYGON = 'create new polygon'
	ADDING_VERTICES_CONSECUTIVELY = 'adding vertices consecutively'
	ADDING_VERTICES_RANDOMLY = 'adding vertices randomly'
	KEEP_SELECTION = 'keep selection'
	SELECT_UNCERTAIN_SEGMENT = 'select uncertain segment'
	DELETE_ROI_MERGE = 'delete roi (merge)'
	DELETE_ROI_DUPLICATE = 'delete roi (duplicate)'
	DELETE_BETWEEN = 'delete edges between two vertices'
	CONNECT_VERTICES = 'connect two vertices'


class SignalEmitter(QObject):
    moved = pyqtSignal(int, int, int, int)
    clicked = pyqtSignal()
    released = pyqtSignal()
    
    def __init__(self, parent):
        super(SignalEmitter, self).__init__()
        self.parent = parent

class QGraphicsPathItemModified(QGraphicsPathItem):

	def __init__(self, path, parent=None, gui=None):
		super(self.__class__, self).__init__(path, parent=parent)
		self.signal_emitter = SignalEmitter(parent=self)
		# self.just_created = True # this flag is used to make sure a click is not emitted right after item creation
								# basically, ignore the first press and release event
		self.gui = gui

	def mousePressEvent(self, event):

	# if not self.just_created:
		# print self, 'received mousePressEvent'

		self.press_scene_x = event.scenePos().x()
		self.press_scene_y = event.scenePos().y()

		self.center_scene_x_before_move = self.scenePos().x()
		self.center_scene_y_before_move = self.scenePos().y()

		self.gui.selected_polygon = self

		QGraphicsPathItem.mousePressEvent(self, event)

		self.signal_emitter.clicked.emit()

		if 'labelTextArtist' in self.gui.accepted_proposals_allSections[self.gui.selected_section][self.gui.selected_polygon]:
			label_pos_before_move = self.gui.accepted_proposals_allSections[self.gui.selected_section][self.gui.selected_polygon]['labelTextArtist'].scenePos()
			self.label_pos_before_move_x = label_pos_before_move.x()
			self.label_pos_before_move_y = label_pos_before_move.y()

		# self.just_created = False

	def mouseReleaseEvent(self, event):
		# if not self.just_created:
		print self, 'received mouseReleaseEvent'
		
		release_scene_pos = event.scenePos()
		self.release_scene_x = release_scene_pos.x()
		self.release_scene_y = release_scene_pos.y()

		QGraphicsPathItem.mouseReleaseEvent(self, event)
		self.signal_emitter.released.emit()

		self.press_scene_x = None
		self.press_scene_y = None

		self.center_scene_x_before_move = None
		self.center_scene_y_before_move = None

	def mouseMoveEvent(self, event):
		print self, 'received mouseMoveEvent'
		self.signal_emitter.moved.emit(event.scenePos().x(), event.scenePos().y(), self.press_scene_x, self.press_scene_y)

		if not self.gui.mode == Mode.IDLE:
			QGraphicsPathItem.mouseMoveEvent(self, event)


class QGraphicsEllipseItemModified(QGraphicsEllipseItem):

	def __init__(self, x, y, w, h, gui=None, parent=None):
		super(self.__class__, self).__init__(x,y,w,h, parent=parent)
		self.signal_emitter = SignalEmitter(parent=self)

		# self.just_created = True # this flag is used to make sure a click is not emitted right after item creation
								# basically, ignore the first press and release event
		self.gui = gui

	def mousePressEvent(self, event):

		print self, 'received mousePressEvent'
		QGraphicsEllipseItem.mousePressEvent(self, event)

		self.signal_emitter.clicked.emit()

		self.press_scene_x = event.scenePos().x()
		self.press_scene_y = event.scenePos().y()

		self.center_scene_x_before_move = self.scenePos().x()
		self.center_scene_y_before_move = self.scenePos().y()

		self.gui.selected_vertex = self

		self.gui.selected_polygon = self.gui.inverse_lookup[self]

		# for p, props in self.gui.accepted_proposals_allSections[self.gui.selected_section].iteritems():
		# 	if self in props['vertexCircles']:
		# 		self.gui.selected_polygon = p
		# 		break

		# self.just_created = False
		# print 'just created UNSET'

	def mouseReleaseEvent(self, event):
		# if not self.just_created:
		print self, 'received mouseReleaseEvent'

		release_scene_pos = event.scenePos()
		self.release_scene_x = release_scene_pos.x()
		self.release_scene_y = release_scene_pos.y()

		QGraphicsEllipseItem.mouseReleaseEvent(self, event)
		self.signal_emitter.released.emit()

		self.press_scene_x = None
		self.press_scene_y = None

		self.center_scene_x_before_move = None
		self.center_scene_y_before_move = None

	def mouseMoveEvent(self, event):
		# print self, 'received mouseMoveEvent'
		self.signal_emitter.moved.emit(event.scenePos().x(), event.scenePos().y(), self.press_scene_x, self.press_scene_y)
		QGraphicsEllipseItem.mouseMoveEvent(self, event)
