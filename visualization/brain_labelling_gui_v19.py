#! /usr/bin/env python

import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

import sys
import os
import datetime
from random import random
import subprocess
import time
import json
from pprint import pprint
import cPickle as pickle
from itertools import groupby
from operator import itemgetter

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

from ui_BrainLabelingGui_v13 import Ui_BrainLabelingGui

from matplotlib.colors import ListedColormap, NoNorm, ColorConverter

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import LinearRing as ShapelyLineRing

from skimage.color import label2rgb

from visualization_utilities import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

from collections import defaultdict, OrderedDict, deque

from operator import attrgetter

import requests

from joblib import Parallel, delayed

# from LabelingPainter import LabelingPainter
from custom_widgets import *
from SignalEmittingItems import *

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

class ProposalType(Enum):
	GLOBAL = 'global'
	LOCAL = 'local'
	FREEFORM = 'freeform'
	ALGORITHM = 'algorithm'

class PolygonType(Enum):
	CLOSED = 'closed'
	OPEN = 'open'
	TEXTURE = 'textured'
	TEXTURE_WITH_CONTOUR = 'texture with contour'
	DIRECTION = 'directionality'

SELECTED_POLYGON_LINEWIDTH = 2
UNSELECTED_POLYGON_LINEWIDTH = 2
SELECTED_CIRCLE_SIZE = 30
UNSELECTED_CIRCLE_SIZE = 5
CIRCLE_PICK_THRESH = 1000.
PAN_THRESHOLD = 10

PEN_WIDTH = 10

HISTORY_LEN = 20

AUTO_EXTEND_VIEW_TOLERANCE = 200

# NUM_NEIGHBORS_PRELOAD = 1 # preload neighbor sections before and after this number
VERTEX_CIRCLE_RADIUS = 10
	
class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
	def __init__(self, parent=None, stack=None):
		"""
		Initialization of BrainLabelingGUI.
		"""

		self.params_dir = '../params'

		# self.app = QApplication(sys.argv)
		QMainWindow.__init__(self, parent)

		# self.init_data(stack)
		self.stack = stack

		self.recent_labels = []

		# self.history = deque(maxlen=HISTORY_LEN)

		deque(maxlen=HISTORY_LEN)

		self.history_allSections = defaultdict(list)

		self.new_labelnames = {}
		if os.path.exists(os.environ['REPO_DIR']+'/visualization/newStructureNames.txt'):
			with open(os.environ['REPO_DIR']+'/visualization/newStructureNames.txt', 'r') as f:
				for ln in f.readlines():
					abbr, fullname = ln.split('\t')
					self.new_labelnames[abbr] = fullname.strip()
			self.new_labelnames = OrderedDict(sorted(self.new_labelnames.items()))

		self.structure_names = {}
		with open(os.environ['REPO_DIR']+'/visualization/structure_names.txt', 'r') as f:
			for ln in f.readlines():
				abbr, fullname = ln.split('\t')
				self.structure_names[abbr] = fullname.strip()
		self.structure_names = OrderedDict(self.new_labelnames.items() + sorted(self.structure_names.items()))

		self.first_sec, self.last_sec = section_range_lookup[self.stack]
		# self.midline_sec = midline_section_lookup[self.stack]
		self.midline_sec = (self.first_sec + self.last_sec)/2

		self.red_pen = QPen(Qt.red)
		self.red_pen.setWidth(PEN_WIDTH)
		self.blue_pen = QPen(Qt.blue)
		self.blue_pen.setWidth(PEN_WIDTH)
		self.green_pen = QPen(Qt.green)
		self.green_pen.setWidth(PEN_WIDTH)

		self.initialize_brain_labeling_gui()
		# self.labeling_painters = {}

		self.gscenes = {} # exactly one for each section {section: gscene}
		self.gviews = [self.section1_gview, self.section2_gview, self.section3_gview] # exactly one for each section {section: gscene}

		self.accepted_proposals_allSections = {}
		self.inverse_lookup = {}
		self.polygon_inverse_lookup = {}

		self.lateral_position_lookup = dict(zip(range(self.first_sec, self.midline_sec+1), -np.linspace(2.64, 0, self.midline_sec-self.first_sec+1)) + \
											zip(range(self.midline_sec, self.last_sec+1), np.linspace(0, 2.64, self.last_sec-self.midline_sec+1)))

	def load_active_set(self, sections=None):

		if sections is None:
			self.sections = [self.section, self.section2, self.section3]
		else:

			minsec = min(sections)
			maxsec = max(sections)

			self.sections = range(max(self.first_sec, minsec), min(self.last_sec, maxsec+1))
			
			print self.sections

			self.dms = dict([(i, DataManager(
			    # data_dir=os.environ['DATA_DIR'], 
			    data_dir='/media/yuncong/MyPassport', 
			         repo_dir=os.environ['REPO_DIR'], 
			         # result_dir=os.environ['RESULT_DIR'], 
			         # labeling_dir=os.environ['LOCAL_LABELING_DIR'],
			         labeling_dir='/home/yuncong/CSHL_data_labelings_losslessAlignCropped',
			         # labeling_dir='/home/yuncong/CSHL_autoAnnotations_snake',
			    stack=stack, section=i, segm_params_id='tSLIC200', load_mask=False)) 
			for i in self.sections])
				# for i in range(self.first_sec, self.last_sec+1)])

			t = time.time()

			print 'self.sections', self.sections

			if hasattr(self, 'pixmaps'):
				for i in self.sections:
					if i not in self.pixmaps:
						print 'new load', i
						self.pixmaps[i] = QPixmap(self.dms[i]._get_image_filepath(version='rgb-jpg'))
						# self.pixmaps[i] = QPixmap(self.dms[i]._get_image_filepath(version='stereotactic-rgb-jpg'))

				to_remove = []
				for i in self.pixmaps:
					if i not in self.sections:
						to_remove.append(i)
				
				print 'to_remove', to_remove

				for i in to_remove:
					m = self.pixmaps.pop(i)
					del m

					if i in self.gscenes:
						s = self.gscenes.pop(i)
						del s
					
			else:	
			
				self.pixmaps = dict([(i, QPixmap(self.dms[i]._get_image_filepath(version='rgb-jpg'))) for i in self.sections])
				# self.pixmaps = dict([(i, QPixmap(self.dms[i]._get_image_filepath(version='stereotactic-rgb-jpg'))) for i in self.sections])
			

			print 'load image', time.time() - t


	def paint_panel(self, panel_id, sec, labeling_username=None):
		'''
		Show section `sec` in panel `panel_id`.

		Args:
			panel_id (int): the index of panel
			sec (int): index of the section to show

		'''

		# if not hasattr(self, 'grid_pixmap'):
		# 	self.grid_pixmap = QPixmap('/home/yuncong/CSHL_data_processed/MD594_lossless_aligned_cropped_stereotacticGrids.png')

		gview = self.gviews[panel_id]

		if sec in self.gscenes:
			print 'gscene exists'
			gscene = self.gscenes[sec]
		else:
			print 'new gscene'
			pixmap = self.pixmaps[sec]
			gscene = QGraphicsScene(gview)
			gscene.addPixmap(pixmap)
			# gscene.addPixmap(self.grid_pixmap)

			self.accepted_proposals_allSections[sec] = {}

			gscene.update(0, 0, gview.width(), gview.height())
			gscene.keyPressEvent = self.key_pressed

			gscene.installEventFilter(self)

			self.gscenes[sec] = gscene

			ret = self.dms[sec].load_proposal_review_result(None, 'latest', 'consolidated')
			if ret is not None:
				usr, ts, suffix, annotations = ret
				self.load_labelings(annotations, sec=sec)

		if panel_id == 0:
			self.section1_gscene = gscene
		elif panel_id == 1:
			self.section2_gscene = gscene
		elif panel_id == 2:
			self.section3_gscene = gscene

		gview.setScene(gscene)

		gview.show()

		# self.section1_gview.setInteractive(True)
		# self.section1_gview.setDragMode(QGraphicsView.RubberBandDrag)
		# self.section1_gview.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
		
		# self.labeling_painters[panel_id] = LabelingPainter(gview, gscene, pixmap)

	# def reload_brain_labeling_gui(self):

	def initialize_brain_labeling_gui(self):

		self.colors = np.loadtxt('100colors.txt', skiprows=1)
		self.label_cmap = ListedColormap(self.colors, name='label_cmap')

		self.setupUi(self)

		# self.button_autoDetect.clicked.connect(self.autoDetect_callback)
		# self.button_updateDB.clicked.connect(self.updateDB_callback)
		# self.button_loadLabeling.clicked.connect(self.load_callback)
		self.button_saveLabeling.clicked.connect(self.save_callback)
		self.button_quit.clicked.connect(self.close)

		self.lineEdit_username.returnPressed.connect(self.username_changed)

		self.button_loadLabelingSec1.clicked.connect(self.load_callback1)
		self.button_loadLabelingSec2.clicked.connect(self.load_callback2)
		self.button_loadLabelingSec3.clicked.connect(self.load_callback3)

		# self.display_buttons = [self.img_radioButton, self.textonmap_radioButton, self.dirmap_radioButton]
		# self.img_radioButton.setChecked(True)

		# for b in self.display_buttons:
		# 	b.toggled.connect(self.display_option_changed)

		# self.radioButton_globalProposal.toggled.connect(self.mode_changed)
		# self.radioButton_localProposal.toggled.connect(self.mode_changed)

		# self.buttonSpOnOff.clicked.connect(self.display_option_changed)
		# self.button_labelsOnOff.clicked.connect(self.toggle_labels)
		# self.button_contoursOnOff.clicked.connect(self.toggle_contours)
		# self.button_verticesOnOff.clicked.connect(self.toggle_vertices)

		# self.thumbnail_list = QListWidget(parent=self)
		self.thumbnail_list.setIconSize(QSize(200,200))
		self.thumbnail_list.setResizeMode(QListWidget.Adjust)
		# self.thumbnail_list.itemDoubleClicked.connect(self.section_changed)
		self.thumbnail_list.itemDoubleClicked.connect(self.init_section_selected)

		for i in range(self.first_sec, self.last_sec):
			item = QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnail_aligned_cropped.tif"%{'sec':i, 'stack': self.stack}), str(i))
			self.thumbnail_list.addItem(item)

		self.thumbnail_list.resizeEvent = self.thumbnail_list_resized
		self.init_thumbnail_list_width = self.thumbnail_list.width()

		self.section1_gscene = None
		self.section2_gscene = None
		self.section3_gscene = None

	def username_changed(self):
		self.username = str(self.sender().text())
		print 'username changed to', self.username


	def showContextMenu(self, pos):
		myMenu = QMenu(self)
		action_newPolygon = myMenu.addAction("New polygon")
		action_deletePolygon = myMenu.addAction("Delete polygon")
		action_setLabel = myMenu.addAction("Set label")
		action_setUncertain = myMenu.addAction("Set uncertain segment")
		action_deleteROIDup = myMenu.addAction("Delete vertices in ROI (duplicate)")
		action_deleteROIMerge = myMenu.addAction("Delete vertices in ROI (merge)")
		action_deleteBetween = myMenu.addAction("Delete edges between two vertices")
		action_closePolygon = myMenu.addAction("Close polygon")
		action_insertVertex = myMenu.addAction("Insert vertex")
		action_appendVertex = myMenu.addAction("Append vertex")
		action_connectVertex = myMenu.addAction("Connect vertex")
		
		action_doneDrawing = myMenu.addAction("Done drawing")

		selected_action = myMenu.exec_(self.gviews[self.selected_panel_id].viewport().mapToGlobal(pos))
		if selected_action == action_newPolygon:
			print 'new polygon'

			invalid_proposals = []
			for p, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				p.setEnabled(False)

				if 'vertexCircles' not in props:
					invalid_proposals.append(p)
				else:
					for circ in props['vertexCircles']:
						circ.setEnabled(False)

				if 'labelTextArtist' not in props:
					invalid_proposals.append(p)
				else:
					props['labelTextArtist'].setEnabled(False)

			print 'invalid_proposals', invalid_proposals

			self.close_curr_polygon = False

			self.selected_polygon = self.add_polygon(QPainterPath(), self.red_pen)
			assert self.selected_polygon is not None

			self.set_mode(Mode.ADDING_VERTICES_CONSECUTIVELY)

		elif selected_action == action_deletePolygon:
			self.remove_polygon(self.selected_polygon)

		elif selected_action == action_setLabel:
			self.open_label_selection_dialog()

		elif selected_action == action_setUncertain:
			self.set_mode(Mode.SELECT_UNCERTAIN_SEGMENT)

		elif selected_action == action_deleteROIDup:
			self.set_mode(Mode.DELETE_ROI_DUPLICATE)
		
		elif selected_action == action_deleteROIMerge:
			self.set_mode(Mode.DELETE_ROI_MERGE)

		elif selected_action == action_deleteBetween:
			self.set_mode(Mode.DELETE_BETWEEN)

		elif selected_action == action_closePolygon:
			new_path = self.selected_polygon.path()
			new_path.closeSubpath()
			self.selected_polygon.setPath(new_path)

		elif selected_action == action_insertVertex:
			self.set_mode(Mode.ADDING_VERTICES_RANDOMLY)

		elif selected_action == action_appendVertex:
			if self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'].index(self.selected_vertex) == 0:
				self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'] = self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'][::-1]
				reversed_path = self.selected_polygon.path().toReversed()
				self.selected_polygon.setPath(reversed_path)

			self.set_mode(Mode.ADDING_VERTICES_CONSECUTIVELY)

		elif selected_action == action_connectVertex:
			self.set_mode(Mode.CONNECT_VERTICES)

		elif selected_action == action_doneDrawing:
			self.set_mode(Mode.IDLE)


	@pyqtSlot()
	def polygon_pressed(self):
		self.polygon_is_moved = False

		print [p.zValue() for p in self.accepted_proposals_allSections[self.selected_section]]

	@pyqtSlot(int, int, int, int)
	def polygon_moved(self, x, y, x0, y0):

		offset_scene_x = x - x0
		offset_scene_y = y - y0

		self.selected_polygon = self.sender().parent

		for i, circ in enumerate(self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles']):
			elem = self.selected_polygon.path().elementAt(i)
			scene_pt = self.selected_polygon.mapToScene(elem.x, elem.y)
			circ.setPos(scene_pt)

		self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'].setPos(self.selected_polygon.label_pos_before_move_x + offset_scene_x, 
										self.selected_polygon.label_pos_before_move_y + offset_scene_y)

		self.polygon_is_moved = True
			
	@pyqtSlot()
	def polygon_released(self):
		
		self.selected_polygon = self.sender().parent

		curr_polygon_path = self.selected_polygon.path()

		for i in range(curr_polygon_path.elementCount()):
			elem = curr_polygon_path.elementAt(i)
			scene_pt = self.selected_polygon.mapToScene(elem.x, elem.y)
			
			curr_polygon_path.setElementPositionAt(i, scene_pt.x(), scene_pt.y())
		
		self.selected_polygon.setPath(curr_polygon_path)
		self.selected_polygon.setPos(0,0)

		if self.mode == Mode.MOVING_VERTEX and self.polygon_is_moved:
			self.history_allSections[self.selected_section].append({'type': 'drag_polygon', 'polygon': self.selected_polygon, 'mouse_moved': (self.selected_polygon.release_scene_x - self.selected_polygon.press_scene_x, \
																										self.selected_polygon.release_scene_y - self.selected_polygon.press_scene_y),
																	'label': self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label']})
			self.polygon_is_moved = False

			
	@pyqtSlot(int, int, int, int)
	def vertex_moved(self, x, y, x0, y0):

		offset_scene_x = x - x0
		offset_scene_y = y - y0

		self.selected_vertex_circle = self.sender().parent
		
		self.selected_vertex_center_x_new = self.selected_vertex_circle.center_scene_x_before_move + offset_scene_x
		self.selected_vertex_center_y_new = self.selected_vertex_circle.center_scene_y_before_move + offset_scene_y

		for p, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
			if self.selected_vertex_circle in props['vertexCircles']:
				self.selected_polygon = p
				break

		vertex_index = self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'].index(self.selected_vertex_circle)
		# print 'vertex_index', vertex_index

		curr_polygon_path = self.selected_polygon.path()

		if vertex_index == 0 and polygon_is_closed(path=curr_polygon_path): # closed

			# print self.selected_vertex_center_x_new, self.selected_vertex_center_y_new

			curr_polygon_path.setElementPositionAt(0, self.selected_vertex_center_x_new, self.selected_vertex_center_y_new)
			curr_polygon_path.setElementPositionAt(len(self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles']), \
											self.selected_vertex_center_x_new, self.selected_vertex_center_y_new)

		else:
			curr_polygon_path.setElementPositionAt(vertex_index, self.selected_vertex_center_x_new, self.selected_vertex_center_y_new)

		self.selected_polygon.setPath(curr_polygon_path)

		self.vertex_is_moved = True


	@pyqtSlot()
	def vertex_clicked(self):
		print self.sender().parent, 'clicked'

		self.vertex_is_moved = False

		clicked_vertex = self.sender().parent

		self.selected_polygon = self.inverse_lookup[clicked_vertex]

		assert clicked_vertex in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles']

		# trying to close the polygon
		print 'clicked index', self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'].index(clicked_vertex)

		if self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'].index(clicked_vertex) == 0 and \
			len(self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles']) > 2 and \
			(self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY or self.mode == Mode.ADDING_VERTICES_RANDOMLY): 
			# the last condition is to prevent setting the flag when one clicks vertex 0 in idle mode.
			print 'close curr polygon SET'
			self.close_curr_polygon = True

	@pyqtSlot()
	def vertex_released(self):
		print self.sender().parent, 'released'

		clicked_vertex = self.sender().parent

		if self.mode == Mode.MOVING_VERTEX and self.vertex_is_moved:
			self.history_allSections[self.selected_section].append({'type': 'drag_vertex', 'polygon': self.selected_polygon, 'vertex': clicked_vertex, \
								 'mouse_moved': (clicked_vertex.release_scene_x - clicked_vertex.press_scene_x, \
								 	clicked_vertex.release_scene_y - clicked_vertex.press_scene_y), \
								 'label': self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label']})

			self.vertex_is_moved = False
			self.print_history()
		
		elif self.mode == Mode.DELETE_BETWEEN:
			vertex_index = self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'].index(clicked_vertex)
			print 'vertex_index', vertex_index 

			rect = clicked_vertex.rect()
			clicked_vertex.setRect(rect.x()-.5*VERTEX_CIRCLE_RADIUS, rect.y()-.5*VERTEX_CIRCLE_RADIUS, 3*VERTEX_CIRCLE_RADIUS, 3*VERTEX_CIRCLE_RADIUS)

			if hasattr(self, 'first_vertex_index_to_delete') and self.first_vertex_index_to_delete is not None:
				self.second_vertex_index_to_delete = vertex_index

				self.delete_between(self.selected_polygon, self.first_vertex_index_to_delete, self.second_vertex_index_to_delete)
				
				# first_vertex = self.accepted_proposals[self.selected_polygon]['vertexCircles'][self.first_vertex_index_to_delete]
				# rect = first_vertex.rect()
				# first_vertex.setRect(rect.x()-50, rect.y()-50, 100, 100)
				
				self.first_vertex_index_to_delete = None

				# second_vertex = self.accepted_proposals[self.selected_polygon]['vertexCircles'][self.second_vertex_index_to_delete]
				# rect = second_vertex.rect()
				# second_vertex.setRect(rect.x()-50, rect.y()-50, 100, 100)

				self.second_vertex_index_to_delete = None

				self.set_mode(Mode.IDLE)

			else:
				self.first_vertex_index_to_delete = vertex_index

		elif self.mode == Mode.CONNECT_VERTICES:
			vertex_index = self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'].index(clicked_vertex)

			print 'vertex_index', vertex_index 

			rect = clicked_vertex.rect()
			clicked_vertex.setRect(rect.x()-.5*VERTEX_CIRCLE_RADIUS, rect.y()-.5*VERTEX_CIRCLE_RADIUS, 3*VERTEX_CIRCLE_RADIUS, 3*VERTEX_CIRCLE_RADIUS)

			if hasattr(self, 'first_vertex_index_to_connect') and self.first_vertex_index_to_connect is not None:
				self.second_polygon = self.selected_polygon
				self.second_vertex_index_to_connect = vertex_index

				self.connect_vertices(self.first_polygon, self.first_vertex_index_to_connect, self.second_polygon, self.second_vertex_index_to_connect)
				
				if self.first_polygon == self.second_polygon: # not creating new polygon, so need to restore the vertex circle sizes

					first_vertex = self.accepted_proposals_allSections[self.selected_section][self.first_polygon]['vertexCircles'][self.first_vertex_index_to_connect]
					rect = first_vertex.rect()
					first_vertex.setRect(rect.x()+.5*VERTEX_CIRCLE_RADIUS, rect.y()+.5*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS)
				
					second_vertex = self.accepted_proposals_allSections[self.selected_section][self.second_polygon]['vertexCircles'][self.second_vertex_index_to_connect]
					rect = second_vertex.rect()
					second_vertex.setRect(rect.x()+.5*VERTEX_CIRCLE_RADIUS, rect.y()+.5*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS)

				self.first_polygon = None
				self.first_vertex_index_to_connect = None

				self.second_polygon = None
				self.second_vertex_index_to_connect = None

				self.set_mode(Mode.IDLE)

			else:
				self.first_polygon = self.selected_polygon
				self.first_vertex_index_to_connect = vertex_index


	def set_flag_all(self, flag, enabled):

		if hasattr(self, 'accepted_proposals_allSections'):

			for ac in self.accepted_proposals_allSections.itervalues():
				for p, props in ac.iteritems():
					p.setFlag(flag, enabled)
					for circ in props['vertexCircles']:
						circ.setFlag(flag, enabled)
					if 'labelTextArtist' in props:
						props['labelTextArtist'].setFlag(flag, enabled)


	def eventFilter(self, obj, event):

		if event.type() == QEvent.GraphicsSceneMousePress or event.type() == QEvent.GraphicsSceneMouseRelease or event.type() == QEvent.Wheel:

			if obj == self.section1_gscene or obj == self.section1_gview.viewport() :
				self.selected_section = self.section
				self.selected_panel_id = 0
			elif obj == self.section2_gscene or obj == self.section2_gview.viewport() :
				self.selected_section = self.section2
				self.selected_panel_id = 1
			elif obj == self.section3_gscene or obj == self.section3_gview.viewport() :
				self.selected_section = self.section3
				self.selected_panel_id = 2

		# if hasattr(self, 'selected_section'):
		# 	print 'selected_section = ', self.selected_section

		if (obj == self.section1_gview.viewport() or \
			obj == self.section2_gview.viewport() or \
			obj == self.section3_gview.viewport()) and event.type() == QEvent.Wheel:
			self.zoom_scene(event)
			return True

		if obj == self.section1_gscene:
			obj_type = 'gscene'
			gscene = self.section1_gscene
			gview = self.section1_gview
			# self.selected_section = self.section
			# self.selected_panel_id = 0
		elif obj == self.section2_gscene:
			obj_type = 'gscene'
			gscene = self.section2_gscene
			gview = self.section2_gview
			# self.selected_section = self.section2
			# self.selected_panel_id = 1
		elif obj == self.section3_gscene:
			obj_type = 'gscene'
			gscene = self.section3_gscene
			gview = self.section3_gview
			# self.selected_section = self.section3
			# self.selected_panel_id = 2
		else:
			obj_type = 'other'

		if obj_type == 'gscene' and event.type() == QEvent.GraphicsSceneMousePress:

			### with this, single click can select an item; without this only double click can select an item (WEIRD !!!)
			gview.translate(0.1, 0.1)
			gview.translate(-0.1, -0.1)

			##########################################

			obj.mousePressEvent(event) # let gscene handle the event (i.e. determine which item or whether an item receives it)
			# here self.vertex_clicked() will be triggered by the signal emitted from obj

			if self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY:

				return True

			elif self.mode == Mode.ADDING_VERTICES_RANDOMLY:

				return True

			elif self.mode == Mode.IDLE:

				self.press_x = event.pos().x()
				self.press_y = event.pos().y()

				self.press_screen_x = event.screenPos().x()
				self.press_screen_y = event.screenPos().y()

				self.pressed = True

				return True
				
			return False

		if obj_type == 'gscene' and event.type() == QEvent.GraphicsSceneMouseMove:

			# print 'event filter: mouse move'

			if self.mode == Mode.ADDING_VERTICES_RANDOMLY or self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY:
				return True

			if self.mode == Mode.MOVING_VERTEX:
				obj.mouseMoveEvent(event)

			elif self.mode == Mode.IDLE:
				if hasattr(self, 'event_caused_by_panning') and self.event_caused_by_panning:
					# self.event_caused_by_panning = False
					return True

				if self.pressed:

					self.event_caused_by_panning = True

					self.curr_scene_x = event.scenePos().x()
					self.curr_scene_y = event.scenePos().y()

					self.last_scene_x = event.lastScenePos().x()
					self.last_scene_y = event.lastScenePos().y()

					self.section1_gview.translate(self.curr_scene_x - self.last_scene_x, self.curr_scene_y - self.last_scene_y)
					self.section2_gview.translate(self.curr_scene_x - self.last_scene_x, self.curr_scene_y - self.last_scene_y)
					self.section3_gview.translate(self.curr_scene_x - self.last_scene_x, self.curr_scene_y - self.last_scene_y)
					# these move canvas and trigger GraphicsSceneMouseMove event again, causing recursion

					self.event_caused_by_panning = False
					return True

			return False

		if obj_type == 'gscene' and event.type() == QEvent.GraphicsSceneMouseRelease:

			obj.mouseReleaseEvent(event)

			pos = event.scenePos()
			x, y = (pos.x(), pos.y())

			if self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY:

				if hasattr(self, 'close_curr_polygon') and self.close_curr_polygon:

					print 'close curr polygon UNSET'
					self.close_curr_polygon = False
					
					self.close_polygon()

					if 'label' not in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]:
						self.open_label_selection_dialog()

					self.set_mode(Mode.IDLE)
				
				else:

					polygon_goto(self.selected_polygon, x, y)

					self.add_vertex_to_polygon(self.selected_polygon, x, y)
					self.print_history()

				return True

			elif self.mode == Mode.ADDING_VERTICES_RANDOMLY:

				if hasattr(self, 'close_curr_polygon') and self.close_curr_polygon:

					print 'close curr polygon UNSET'
					self.close_curr_polygon = False
					
					self.close_polygon()

					self.set_mode(Mode.IDLE)

				elif self.selected_polygon.path().elementCount() == 0: # just created a new polygon, no vertices yet; in this case, just use self.selected_polygon
					pass
				else:
					# self.selected_polygon is pre-selected. This is not ideal. 
					# Should determine which polygon is selected based on location of the click.

					pt = Point(x, y)
					polygons = self.accepted_proposals_allSections[self.selected_section].keys()
					distances_to_polygons = [polygon_to_shapely(polygon=p).distance(pt) for p in polygons]
					self.selected_polygon = polygons[np.argmin(distances_to_polygons)]

				new_index = find_vertex_insert_position(self.selected_polygon, x, y)
				print 'new index', new_index

				path = self.selected_polygon.path()
				new_path = QPainterPath()
				for i in range(path.elementCount()+1): # +1 is important, because the new_index can be after the last vertex
					if i == new_index:
						path_goto(new_path, x, y)
					if i < path.elementCount():
						elem = path.elementAt(i)
						path_goto(new_path, elem.x, elem.y)

				self.selected_polygon.setPath(new_path)

				self.add_vertex_to_polygon(self.selected_polygon, x, y, new_index)

				self.print_history()
				return True

			# elif self.mode == Mode.MOVING_VERTEX:

			# 	# pos = event.scenePos()
			# 	# x, y = (pos.x(), pos.y())
			# 	return True

			elif self.mode == Mode.IDLE:
				self.release_scene_x = event.scenePos().x()
				self.release_scene_y = event.scenePos().y()

				self.pressed = False

				return True

			elif self.mode == Mode.DELETE_ROI_MERGE or self.mode == Mode.DELETE_ROI_DUPLICATE:

				selected_polygons = self.analyze_rubberband_selection()

				for p, vs in selected_polygons.iteritems():
					print p, vs
					if self.mode == Mode.DELETE_ROI_DUPLICATE:
						self.delete_vertices(p, vs)
					elif self.mode == Mode.DELETE_ROI_MERGE:
						self.delete_vertices(p, vs, merge=True)

				self.set_mode(Mode.IDLE)

			elif self.mode == Mode.SELECT_UNCERTAIN_SEGMENT:

				selected_polygons = self.analyze_rubberband_selection()

				for polygon, vertex_indices in selected_polygons.iteritems():
					self.set_uncertain(polygon, vertex_indices)

				self.set_mode(Mode.IDLE)

			return False

		return False


	def analyze_rubberband_selection(self):

		items = self.gscenes[self.selected_section].selectedItems()

		vertices_selected = [i for i in items if i not in self.accepted_proposals_allSections[self.selected_section] and isinstance(i, QGraphicsEllipseItemModified)]

		polygons = defaultdict(list)
		for v in vertices_selected:
			for p, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				if v in props['vertexCircles']:
					polygons[p].append(props['vertexCircles'].index(v))

		return polygons # {polygon: vertex_indices}


	def subpath(self, path, begin, end):

		new_path = QPainterPath()

		is_closed = polygon_is_closed(path=path)
		n = path.elementCount() - 1 if is_closed else path.elementCount()

		if not is_closed:
			assert end >= begin
			begin = max(0, begin)
			end = min(n-1, end)
		else:
			assert end != begin # cannot handle this, because there is no way a path can have the same first and last points but is not closed
			if end < begin: 
				end = end + n

		for i in range(begin, end + 1):
			elem = path.elementAt(i % n)
			if new_path.elementCount() == 0:
				new_path.moveTo(elem.x, elem.y)
			else:
				new_path.lineTo(elem.x, elem.y)

		assert new_path.elementCount() > 0

		return new_path

	def set_uncertain(self, polygon, vertex_indices):

		uncertain_paths, certain_paths = self.split_path(polygon.path(), vertex_indices)

		for path in uncertain_paths:
			new_uncertain_polygon = self.add_polygon_by_vertices_label(path, self.green_pen, self.accepted_proposals_allSections[self.selected_section][polygon]['label'])

		for path in certain_paths:
			new_certain_polygon = self.add_polygon_by_vertices_label(path, self.red_pen, self.accepted_proposals_allSections[self.selected_section][polygon]['label'])

		self.remove_polygon(polygon)

		# self.history.append({'type': 'set_uncertain_segment', 'old_polygon': polygon, 'new_certain_polygon': new_certain_polygon, 'new_uncertain_polygon': new_uncertain_polygon})
		# print 'history:', [h['type'] for h in self.history]


	def connect_vertices(self, polygon1, vertex_ind1, polygon2, vertex_ind2):
		'''
		Connect two vertices, in different polygons or in the same polygon.

		Args:
			polygon1 (QGraphicsPathItemModified): first polygon
			vertex_ind1 (int): index of the first vertex
			polygon2 (QGraphicsPathItemModified): second polygon, can be the same as polygon1
			vertex_ind2 (int): index of the second vertex

		Returns:
			QGraphicsPathItemModified: the connected polygon

		'''

		if polygon1 == polygon2:

			path = polygon1.path()
			is_closed = polygon_is_closed(path=path)
			n = path.elementCount() -1 if is_closed else path.elementCount()
			assert (vertex_ind1 == 0 and vertex_ind2 == n-1) or (vertex_ind1 == n-1 and vertex_ind2 == 0)
			path.closeSubpath()
			polygon1.setPath(path)

			return polygon1

		else:

			path1 = polygon1.path()
			is_closed = polygon_is_closed(path=path1)
			n1 = path1.elementCount() -1 if is_closed else path1.elementCount()

			path2 = polygon2.path()
			is_closed2 = polygon_is_closed(path=path2)
			n2 = path2.elementCount() -1 if is_closed2 else path2.elementCount()

			assert not is_closed and not is_closed2 and vertex_ind1 in [0, n1-1] and vertex_ind2 in [0, n2-1]

			if vertex_ind1 == 0 and vertex_ind2 == 0:
				reversed_path1 = path1.toReversed()
				for i in range(path2.elementCount()):
					elem = path2.elementAt(i) 
					reversed_path1.lineTo(elem.x, elem.y)
				new_polygon = self.add_polygon_by_vertices_label(reversed_path1, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon1]['label'])
				
			elif vertex_ind1 == n1-1 and vertex_ind2 == n2-1:

				reversed_path2 = path2.toReversed()
				for i in range(reversed_path2.elementCount()):
					elem = reversed_path2.elementAt(i)
					path1.lineTo(elem.x, elem.y)
				new_polygon = self.add_polygon_by_vertices_label(path1, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon1]['label'])
				
			elif vertex_ind1 == 0 and vertex_ind2 == n2-1:
				for i in range(path1.elementCount()):
					elem = path1.elementAt(i)
					path2.lineTo(elem.x, elem.y)
				new_polygon = self.add_polygon_by_vertices_label(path2, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon1]['label'])

			elif vertex_ind1 == n1-1 and vertex_ind2 == 0:
				for i in range(path2.elementCount()):
					elem = path2.elementAt(i)
					path1.lineTo(elem.x, elem.y)
				new_polygon = self.add_polygon_by_vertices_label(path1, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon1]['label'])

			self.remove_polygon(polygon1)
			self.remove_polygon(polygon2)

			return new_polygon


	def remove_polygon(self, polygon, log=True):
		'''
		Remove a polygon.

		Args:
			polygon (QGraphicsPathItemModified): the polygon to remove

		'''

		# remove vertex circles
		for circ in self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles']:
			self.gscenes[self.selected_section].removeItem(circ)

		# remove labelTextArtist
		if 'labelTextArtist' in self.accepted_proposals_allSections[self.selected_section][polygon]:
			self.gscenes[self.selected_section].removeItem(self.accepted_proposals_allSections[self.selected_section][polygon]['labelTextArtist'])

		self.gscenes[self.selected_section].removeItem(polygon)

		self.accepted_proposals_allSections[self.selected_section].pop(polygon)

		if log:
			self.history_allSections[self.selected_section].append({
				'type': 'remove_polygon', 
				'polygon': self.selected_polygon
				})

		# self.save_selected_section(self.selected_section)


	def add_polygon_by_vertices_label(self, path, pen=None, label='unknown', sec=None):
		'''
		Function for adding polygon, along with vertex circles.
		Step 1: create polygon
		Step 2: create vertices
		Step 3: reorder overlapping polygons if any
		Step 4: add label
		'''

		self.history_allSections[sec].append({
			'type': 'add_polygon_by_vertices_label_begin'
			})

		new_polygon = self.add_polygon(path, pen, sec=sec)
		self.populate_polygon_with_vertex_circles(new_polygon)
		self.restack_polygons(new_polygon)
		self.add_label_to_polygon(new_polygon, label=label)

		self.history_allSections[sec].append({
			'type': 'add_polygon_by_vertices_label_end'
			})

		return new_polygon


	def add_polygon(self, path=QPainterPath(), pen=None, z_value=50, uncertain=False, sec=None):
		'''
		Add a polygon.

		Args:
			path (QPainterPath): path of the polygon
			pen (QPen): pen used to draw polygon

		Returns:
			QGraphicsPathItemModified: added polygon

		'''

		# if path is None:
		# 	path = QPainterPath()

		if pen is None:
			pen = self.red_pen

		if sec is None:
			sec = self.selected_section

		polygon = QGraphicsPathItemModified(path, gui=self)
		
		polygon.setPen(pen)
		polygon.setZValue(z_value)
		polygon.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)

		polygon.signal_emitter.clicked.connect(self.polygon_pressed)
		polygon.signal_emitter.moved.connect(self.polygon_moved)
		polygon.signal_emitter.released.connect(self.polygon_released)

		self.gscenes[sec].addItem(polygon)

		self.accepted_proposals_allSections[sec][polygon] = {'vertexCircles': [], 'uncertain': uncertain}

		self.polygon_inverse_lookup[polygon] = sec

		self.history_allSections[sec].append({
			'type': 'add_polygon', 
			'polygon': polygon
			})

		return polygon


	def add_label_to_polygon(self, polygon, label, label_pos=None):
		'''
		Add label to a polygon.

		Args:
			polygon (QGraphicsPathItemModified): the polygon to add label
			label (str): the label
			label_pos (tuple): label position

		'''

		sec = self.polygon_inverse_lookup[polygon]

		self.accepted_proposals_allSections[sec][polygon]['label'] = label

		textItem = QGraphicsSimpleTextItem(QString(label))

		if label_pos is None:
			centroid = np.mean([(v.scenePos().x(), v.scenePos().y()) for v in self.accepted_proposals_allSections[sec][polygon]['vertexCircles']], axis=0)
			textItem.setPos(centroid[0], centroid[1])
		else:
			textItem.setPos(label_pos[0], label_pos[1])

		textItem.setScale(1.5)

		textItem.setFlags(QGraphicsItem.ItemIgnoresTransformations | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)

		textItem.setZValue(99)
		self.accepted_proposals_allSections[sec][polygon]['labelTextArtist'] = textItem

		self.gscenes[sec].addItem(textItem)

		self.history_allSections[sec].append({
			'type': 'set_label', 
			'polygon': polygon,
			'label': label
			})



	def add_vertex_to_polygon(self, polygon, x, y, new_index=-1, sec=None):
		'''
		Add vertex circle to polygon.

		Args:
			polygon (QGraphicsPathItemModified): polygon
			x (int): x of vertex
			y (int): y of vertex
			new_index (int): index of the vertex

		Returns:
			QGraphicsEllipseItemModified: the added vertex object

		'''

		sec = self.polygon_inverse_lookup[polygon]

		ellipse = QGraphicsEllipseItemModified(-VERTEX_CIRCLE_RADIUS, -VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS, gui=self)
		ellipse.setPos(x,y)

		ellipse.setPen(Qt.blue)
		ellipse.setBrush(Qt.blue)

		ellipse.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
		ellipse.signal_emitter.moved.connect(self.vertex_moved)
		ellipse.signal_emitter.clicked.connect(self.vertex_clicked)
		ellipse.signal_emitter.released.connect(self.vertex_released)

		self.gscenes[sec].addItem(ellipse)

		ellipse.setZValue(99)

		if new_index == -1:
			self.accepted_proposals_allSections[sec][polygon]['vertexCircles'].append(ellipse)
		else:
			self.accepted_proposals_allSections[sec][polygon]['vertexCircles'].insert(new_index, ellipse)

		self.auto_extend_view(x, y)

		self.inverse_lookup[ellipse] = polygon

		self.history_allSections[sec].append({
			'type': 'add_vertex', 
			'polygon': polygon,
			'new_index': new_index if new_index != -1 else len(self.accepted_proposals_allSections[sec][polygon]['vertexCircles'])-1,
			'pos': (x,y)
			})

		return ellipse


	def populate_polygon_with_vertex_circles(self, polygon):
		'''
		Add vertex circles to polygon
		
		Args:
			polygon (QGraphicsPathItemModified): the polygon

		'''

		path = polygon.path()
		is_closed = polygon_is_closed(path=path)

		# if is_closed:
		# 	self.accepted_proposals[polygon]['subtype'] = PolygonType.CLOSED
		# else:
		# 	self.accepted_proposals[polygon]['subtype'] = PolygonType.OPEN

		n = polygon_num_vertices(path=path, closed=is_closed)

		for i in range(n):
			elem = path.elementAt(i)
			circ = self.add_vertex_to_polygon(polygon, elem.x, elem.y, new_index=-1)


	def restack_polygons(self, polygon):
		'''
		Adjust the z-order of a polygon, given other overlapping polygons.

		Args:
			polygon (QGraphicsPathItemModified): the polygon

		Returns:
			list of QGraphicsPathItemModified: polygons overlapping with this polygon

		'''

		sec = self.polygon_inverse_lookup[polygon]

		path = polygon.path()

		n = polygon_num_vertices(path=path)

		overlap_polygons = set([])
		for p in self.accepted_proposals_allSections[sec]:
			if p != polygon:
				for i in range(n):
					elem = path.elementAt(i)
					if p.path().contains(QPointF(elem.x, elem.y)) or p.path().intersects(path):
						print 'overlap_with', overlap_polygons
						overlap_polygons.add(p)

		for p in overlap_polygons:
			if p.path().contains(path): # if new polygon is within existing polygon, it must has higher z value
				new_z = max(polygon.zValue(), p.zValue()+1)
				print polygon, '=>', new_z
				polygon.setZValue(new_z)

			elif path.contains(p.path()):  # if new polygon wraps existing polygon, it must has lower z value
				new_z = min(polygon.zValue(), p.zValue()-1)
				print polygon, '=>', new_z
				polygon.setZValue(new_z)

		return overlap_polygons


	def close_polygon(self, polygon=None):

		if polygon is None:
			polygon = self.selected_polygon

		path = polygon.path()
		path.closeSubpath()
		polygon.setPath(path)

		self.accepted_proposals_allSections[self.selected_section][polygon]['subtype'] = PolygonType.CLOSED

		self.restack_polygons(polygon)

		print 'accepted', self.accepted_proposals_allSections[self.selected_section].keys()

		for p, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
			p.setEnabled(True)
			for circ in props['vertexCircles']:
				circ.setEnabled(True)
			if 'labelTextArtist' in props:
				props['labelTextArtist'].setEnabled(True)

		self.history_allSections[self.selected_section].append({
			'type': 'close_polygon', 
			'polygon': polygon
			})

		self.print_history()

	###################################
	# Key Binding and Mouse Callbacks #
	###################################

	def zoom_scene(self, event):

		pos = self.gviews[self.selected_panel_id].mapToScene(event.pos())

		out_factor = .9
		in_factor = 1./out_factor
		
		if event.delta() < 0: # negative means towards user

			offset_x = (1 - out_factor) * pos.x()
			offset_y = (1 - out_factor) * pos.y()

			# self.section1_gview.translate(-pos.x(), -pos.y())
			self.section1_gview.scale(out_factor, out_factor)
			self.section1_gview.translate(in_factor * offset_x, in_factor * offset_y)

			# self.section2_gview.translate(pos.x(), pos.y())
			self.section2_gview.scale(out_factor, out_factor)
			self.section2_gview.translate(in_factor * offset_x, in_factor * offset_y)
			# self.section2_gview.translate(-pos.x(), -pos.y())

			self.section3_gview.scale(out_factor, out_factor)
			self.section3_gview.translate(in_factor * offset_x, in_factor * offset_y)

		else:
			offset_x = (in_factor - 1) * pos.x()
			offset_y = (in_factor - 1) * pos.y()

			# self.section1_gview.translate(-pos.x(), -pos.y())
			self.section1_gview.scale(in_factor, in_factor)
			self.section1_gview.translate(-out_factor * offset_x, -out_factor * offset_y)
			# self.section1_gview.translate(pos.x(), pos.y())

			# self.section2_gview.translate(pos.x(), pos.y())
			self.section2_gview.scale(in_factor, in_factor)
			self.section2_gview.translate(-out_factor * offset_x, -out_factor * offset_y)
			# self.section2_gview.translate(-pos.x(), -pos.y())

			self.section3_gview.scale(in_factor, in_factor)
			self.section3_gview.translate(-out_factor * offset_x, -out_factor * offset_y)

		sceneRect = self.section1_gview.mapToScene(self.section1_gview.viewport().rect())
		# print sceneRect.x(), sceneRect.y(), sceneRect.width(), sceneRect.height()

	def key_pressed(self, event):

		if event.key() == Qt.Key_D:
			self.set_mode(Mode.DELETE_ROI_DUPLICATE)

		elif event.key() == Qt.Key_I:
			self.set_mode(Mode.ADDING_VERTICES_RANDOMLY)

		elif event.key() == Qt.Key_A:
			if self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'].index(self.selected_vertex) == 0:
				self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'] = self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['vertexCircles'][::-1]
				reversed_path = self.selected_polygon.path().toReversed()
				self.selected_polygon.setPath(reversed_path)

			self.set_mode(Mode.ADDING_VERTICES_CONSECUTIVELY)

		elif event.key() == Qt.Key_Left:
			# print 'left'
			self.section1_gview.translate(200, 0)
			self.section2_gview.translate(200, 0)
			self.section3_gview.translate(200, 0)

		elif event.key() == Qt.Key_Right:
			# print 'right'
			self.section1_gview.translate(-200, 0)
			self.section2_gview.translate(-200, 0)
			self.section3_gview.translate(-200, 0)

		elif event.key() == Qt.Key_Up:
			# print 'up'
			self.section1_gview.translate(0, 200)
			self.section2_gview.translate(0, 200)
			self.section3_gview.translate(0, 200)
			
		elif event.key() == Qt.Key_Down:
			# print 'down'
			self.section1_gview.translate(0, -200)
			self.section2_gview.translate(0, -200)
			self.section3_gview.translate(0, -200)

		elif event.key() == Qt.Key_Equal:
			pos = self.gviews[self.selected_panel_id].mapToScene(self.gviews[self.selected_panel_id].mapFromGlobal(QCursor.pos()))
			
			out_factor = .9
			in_factor = 1./out_factor

			offset_x = (1-out_factor) * pos.x()
			offset_y = (1-out_factor) * pos.y()
		
			self.section1_gview.scale(out_factor, out_factor)
			self.section1_gview.translate(in_factor * offset_x, in_factor * offset_y)

			self.section2_gview.scale(out_factor, out_factor)
			self.section2_gview.translate(in_factor * offset_x, in_factor * offset_y)

			self.section3_gview.scale(out_factor, out_factor)
			self.section3_gview.translate(in_factor * offset_x, in_factor * offset_y)

		elif event.key() == Qt.Key_Minus:
			pos = self.gviews[self.selected_panel_id].mapToScene(self.gviews[self.selected_panel_id].mapFromGlobal(QCursor.pos()))

			out_factor = .9
			in_factor = 1./out_factor

			offset_x = (in_factor - 1) * pos.x()
			offset_y = (in_factor - 1) * pos.y()
			
			self.section1_gview.scale(in_factor, in_factor)
			self.section1_gview.translate(-out_factor * offset_x, -out_factor * offset_y)

			self.section2_gview.scale(in_factor, in_factor)
			self.section2_gview.translate(-out_factor * offset_x, -out_factor * offset_y)

			self.section3_gview.scale(in_factor, in_factor)
			self.section3_gview.translate(-out_factor * offset_x, -out_factor * offset_y)

		elif event.key() == Qt.Key_Space:
			if self.mode == Mode.IDLE:
				self.set_mode(Mode.MOVING_VERTEX)
			else:
			# elif self.mode == Mode.MOVING_VERTEX:
				self.set_mode(Mode.IDLE)
				# self.selected_polygon = None

		# elif event.key() == Qt.Key_Return:
		# 	print 'enter pressed'

		# 	if self.selected_polygon is not None:
		# 		# self.accepted_proposals[self.selected_polygon]['subtype'] = PolygonType.OPEN

		# 		if 'label' not in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon] or self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label'] == '':
		# 			self.complete_polygon()

		# 	self.set_mode(Mode.IDLE)
		# 	# self.selected_polygon = None

		elif event.key() == Qt.Key_C:

			self.close_polygon()

			if 'label' not in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]:
				self.open_label_selection_dialog()

			self.set_mode(Mode.IDLE)

		elif event.key() == Qt.Key_Backspace:
			self.undo()

		elif event.key() == Qt.Key_3 or event.key() == Qt.Key_4:

			if event.key() == Qt.Key_3:
				if self.section == self.first_sec or self.section - 1 not in self.sections:
					return
				else:
					for s1 in range(self.section-1, min(self.sections), -1):
						if s1 != self.section2 and s1 != self.section3:
							self.section = s1
							break
			else:
				if self.section == self.last_sec or self.section + 1 not in self.sections:
					return
				else:
					for s1 in range(self.section+1, max(self.sections)):
						if s1 != self.section2 and s1 != self.section3:
							self.section = s1
							break

			self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', Left %d' %self.section3 + ' (%.3f)' % self.lateral_position_lookup[self.section3] + \
														', middle %d'%self.section + ' (%.3f)' % self.lateral_position_lookup[self.section] + \
														', right %d'%self.section2 + ' (%.3f)' % self.lateral_position_lookup[self.section2])

			self.paint_panel(0, self.section)


		elif event.key() == Qt.Key_5 or event.key() == Qt.Key_6:

			if event.key() == Qt.Key_5:
				if self.section2 == self.first_sec or self.section2 - 1 not in self.sections:
					return
				else:
					for s2 in range(self.section2-1, min(self.sections), -1):
						if s2 != self.section and s2 != self.section3:
							self.section2 = s2
							break

			else:
				if self.section2 == self.last_sec or self.section2 + 1 not in self.sections:
					return
				else:
					for s2 in range(self.section2+1, max(self.sections)):
						if s2 != self.section and s2 != self.section3:
							self.section2 = s2
							break

			self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', Left %d' %self.section3 + ' (%.3f)' % self.lateral_position_lookup[self.section3] + \
											', middle %d'%self.section + ' (%.3f)' % self.lateral_position_lookup[self.section] + \
											', right %d'%self.section2 + ' (%.3f)' % self.lateral_position_lookup[self.section2])

			self.paint_panel(1, self.section2)


		elif event.key() == Qt.Key_1 or event.key() == Qt.Key_2:

			if event.key() == Qt.Key_1:
				if self.section3 == self.first_sec or self.section3 - 1 not in self.sections:
					return
				else:
					for s3 in range(self.section3-1, min(self.sections), -1):
						if s3 != self.section and s3 != self.section2:
							self.section3 = s3
							break

			else:
				if self.section3 == self.last_sec or self.section3 + 1 not in self.sections:
					return
				else:
					for s3 in range(self.section3+1, max(self.sections)):
						if s3 != self.section and s3 != self.section2:
							self.section3 = s3
							break
			
			self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', Left %d' %self.section3 + ' (%.3f)' % self.lateral_position_lookup[self.section3] + \
											', middle %d'%self.section + ' (%.3f)' % self.lateral_position_lookup[self.section] + \
											', right %d'%self.section2 + ' (%.3f)' % self.lateral_position_lookup[self.section2])

			self.paint_panel(2, self.section3)

	##########################

	def thumbnail_list_resized(self, event):
		new_size = 200 * event.size().width() / self.init_thumbnail_list_width
		self.thumbnail_list.setIconSize( QSize(new_size , new_size ) )

	def toggle_labels(self):

		self.labels_on = not self.labels_on

		if not self.labels_on:

			for polygon, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				props['labelTextArtist'].setVisible(False)

			self.button_labelsOnOff.setText('Turns Labels ON')

		else:
			for polygon, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				props['labelTextArtist'].setVisible(True)

			self.button_labelsOnOff.setText('Turns Labels OFF')

	def toggle_contours(self):

		self.contours_on = not self.contours_on

		if not self.contours_on:

			for polygon, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				polygon.setVisible(False)
				if self.vertices_on:
					for circ in props['vertexCircles']:
						circ.setVisible(False)

			self.button_contoursOnOff.setText('Turns Contours ON')

		else:
			for polygon, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				polygon.setVisible(True)
				if self.vertices_on:
					for circ in props['vertexCircles']:
						circ.setVisible(True)

			self.button_contoursOnOff.setText('Turns Contours OFF')


	def toggle_vertices(self):

		self.vertices_on = not self.vertices_on

		if not self.vertices_on:

			for polygon, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				for circ in props['vertexCircles']:
					circ.setVisible(False)

			self.button_verticesOnOff.setText('Turns Vertices ON')

		else:
			for polygon, props in self.accepted_proposals_allSections[self.selected_section].iteritems():
				for circ in props['vertexCircles']:
					circ.setVisible(True)

			self.button_verticesOnOff.setText('Turns Vertices OFF')



	def detect_landmark(self, labels):

		payload = {'labels': labels, 'section': self.dm.slice_ind}
		r = requests.get('http://gcn-20-32.sdsc.edu:5000/top_down_detect', params=payload)
		print r.url
		return r.json()

	def autoDetect_callback(self):
		self.labelsToDetect = ListSelection([abbr + ' (' + fullname + ')' for abbr, fullname in self.structure_names.iteritems()], parent=self)
		self.labelsToDetect.exec_()

		if len(self.labelsToDetect.selected) > 0:
		
			returned_alg_proposal_dict = self.detect_landmark([x.split()[0] for x in list(self.labelsToDetect.selected)]) 
			# list of tuples (sps, dedges, sig)

			for label, (sps, dedges, sig) in returned_alg_proposal_dict.iteritems():

				props = {}

				props['vertices'] = self.dm.vertices_from_dedges(dedges)
				patch = Polygon(props['vertices'], closed=True, edgecolor=self.boundary_colors[0], fill=False, linewidth=UNSELECTED_POLYGON_LINEWIDTH)
				patch.set_picker(True)
				self.axis.add_patch(patch)

				props['vertexPatches'] = []
				for x,y in props['vertices']:
					vertex_circle = plt.Circle((x, y), radius=UNSELECTED_CIRCLE_SIZE, color=self.boundary_colors[1], alpha=.8)
					vertex_circle.set_picker(CIRCLE_PICK_THRESH)
					props['vertexPatches'].append(vertex_circle)
					self.axis.add_patch(vertex_circle)
					vertex_circle.set_picker(True)


				centroid = np.mean(props['vertices'], axis=0)
				props['labelTextArtist'] = Text(centroid[0], centroid[1], label, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
				self.axis.add_artist(props['labelTextArtist'])
				props['labelTextArtist'].set_picker(True)

				self.accepted_proposals[patch] = props

				props['sps'] = sps
				props['dedges'] = dedges
				props['sig'] = sig
				props['type'] = ProposalType.ALGORITHM
				props['label'] = label
		
		self.canvas.draw()

	def on_pick(self, event):

		self.object_picked = False

		if event.mouseevent.name == 'scroll_event':
			return

		print 'pick callback triggered'

		self.picked_artists.append(event.artist)


	def load_callback1(self):
		self.load_callback(panel_ind=1)

	def load_callback2(self):
		self.load_callback(panel_ind=2)

	def load_callback3(self):
		self.load_callback(panel_ind=3)


	def load_labelings(self, annotations, sec=None):
		'''
		Load labelings and paint corresponding gscene.

		'''

		self.accepted_proposals_allSections[sec] = {}

		for props in annotations:
			path = vertices_to_path(props['vertices'])
			polygon = self.add_polygon_by_vertices_label(path=path, label=props['label'], sec=sec)

	# def resolve_gscene_section(self, gscene=None, sec=None):
	# 	if gscene is None:
	# 		if sec is None:
	# 			assert hasattr(self, 'selected_section')
	# 			sec = self.selected_section
	# 		gscene = self.gscenes[sec]
	# 	else:
	# 		assert sec is None # so there is no possibility of conflict
	# 		sec = self.gscenes.keys()[self.gscenes.values().index(gscene)]

	# 	return gscene, sec


	def load_callback(self, panel_ind=None):

		if panel_ind == 1:
			sec = self.section
		elif panel_ind == 2:
			sec = self.section2
		elif panel_ind == 3:
			sec = self.section3
		else:
			return

		fname = str(QFileDialog.getOpenFileName(self, 'Open file', self.dms[sec].labelings_dir))
		stack, sec, username, timestamp, suffix = os.path.basename(fname[:-4]).split('_')
		ret = self.dms[sec].load_proposal_review_result(username, timestamp, suffix)
		if ret is not None:
			annotations = ret[3]
			self.load_labelings(annotations, sec=sec)
			

	def open_label_selection_dialog(self):

		print 'open_label_selection_dialog'

		if hasattr(self, 'recent_labels') and self.recent_labels is not None and len(self.recent_labels) > 0:
			self.structure_names = OrderedDict([(abbr, fullname) for abbr, fullname in self.structure_names.iteritems() if abbr in self.recent_labels] + \
							[(abbr, fullname) for abbr, fullname in self.structure_names.iteritems() if abbr not in self.recent_labels])

		self.label_selection_dialog = AutoCompleteInputDialog(parent=self, labels=[abbr + ' (' + fullname + ')' for abbr, fullname in self.structure_names.iteritems()])
		# self.label_selection_dialog = QInputDialog(self)
		self.label_selection_dialog.setWindowTitle('Select landmark label')

		# if hasattr(self, 'invalid_labelname'):
		#     print 'invalid_labelname', self.invalid_labelname
		# else:
		#     print 'no labelname set'

		if 'label' in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]:
			self.label_selection_dialog.comboBox.setEditText(self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label']+' ('+self.structure_names[self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label']]+')')
		else:
			self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label'] = ''

		self.label_selection_dialog.set_test_callback(self.label_dialog_text_changed)

		# self.label_selection_dialog.accepted.connect(self.label_dialog_text_changed)
		# self.label_selection_dialog.textValueSelected.connect(self.label_dialog_text_changed)

		self.label_selection_dialog.exec_()

	def label_dialog_text_changed(self):

		print 'label_dialog_text_changed'

		text = str(self.label_selection_dialog.comboBox.currentText())

		import re
		m = re.match('^(.+?)\s*\((.+)\)$', text)

		if m is None:
			QMessageBox.warning(self, 'oops', 'structure name must be of the form "abbreviation (full description)"')
			return

		else:
			abbr, fullname = m.groups()
			if not (abbr in self.structure_names.keys() and fullname in self.structure_names.values()):  # new label
				if abbr in self.structure_names:
					QMessageBox.warning(self, 'oops', 'structure with abbreviation %s already exists: %s' % (abbr, fullname))
					return
				else:
					self.structure_names[abbr] = fullname
					self.new_labelnames[abbr] = fullname

		print self.accepted_proposals_allSections.keys()
		print self.selected_section

		self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['label'] = abbr

		if 'labelTextArtist' in self.accepted_proposals_allSections[self.selected_section][self.selected_polygon] and \
				self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'] is not None:
			# label exists
			self.accepted_proposals_allSections[self.selected_section][self.selected_polygon]['labelTextArtist'].setText(abbr)
		else:
			# label not exist, create
			self.add_label_to_polygon(self.selected_polygon, abbr)

		self.recent_labels.insert(0, abbr)

		self.label_selection_dialog.accept()


	def init_section_selected(self, item):
		self.statusBar().showMessage('Loading...')
		sec = int(str(item.text()))

		self.section = sec
		self.section2 = sec + 1
		self.section3 = sec - 1

		# self.load_active_set(sections=range(sec-1, sec+2))
		self.load_active_set(sections=range(sec-NUM_NEIGHBORS_PRELOAD, sec+NUM_NEIGHBORS_PRELOAD+1))
		
		self.extend_head = False
		self.connecting_vertices = False

		self.seg_loaded = False
		self.superpixels_on = False
		self.labels_on = True
		self.contours_on = True
		self.vertices_on = True

		# self.shuffle_global_proposals = True # instead of local proposals

		self.pressed = False           # related to pan (press and drag) vs. select (click)

		for gview in [self.section1_gview, self.section2_gview, self.section3_gview]:
			gview.setMouseTracking(False)
			gview.setVerticalScrollBarPolicy( Qt.ScrollBarAlwaysOff ) 
			gview.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOff )
			gview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
			gview.setTransformationAnchor(QGraphicsView.NoAnchor)
			gview.setContextMenuPolicy(Qt.CustomContextMenu)
			# gview.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
			gview.customContextMenuRequested.connect(self.showContextMenu)

			# if not hasattr(self, 'contextMenu_set') or (hasattr(self, 'contextMenu_set') and not self.contextMenu_set):
			# 	gview.customContextMenuRequested.connect(self.showContextMenu)
			# 	self.contextMenu_set = True

			gview.viewport().installEventFilter(self)

		# if not hasattr(self, 'username') or self.username is None:
		# 	username, okay = QInputDialog.getText(self, "Username", "Please enter username of the labelings you want to load:", QLineEdit.Normal, 'yuncong')
		# 	if not okay: return
		# 	self.username_toLoad = str(username)

		self.paint_panel(0, self.section)
		self.paint_panel(1, self.section2)
		self.paint_panel(2, self.section3)

		self.show()

		self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', Left %d' %self.section3 + ' (%.3f)' % self.lateral_position_lookup[self.section3] + \
											', middle %d'%self.section + ' (%.3f)' % self.lateral_position_lookup[self.section] + \
											', right %d'%self.section2 + ' (%.3f)' % self.lateral_position_lookup[self.section2])

		self.set_mode(Mode.IDLE)

		self.statusBar().showMessage('Loaded')


	def clicked_navMap(self, event):

		scene_x = event.scenePos().x()
		scene_y = event.scenePos().y()

		cur_data_left, cur_data_right = self.axis.get_xlim()
		cur_data_bottom, cur_data_top = self.axis.get_ylim()

		cur_data_center_x = .5 * cur_data_left + .5 * cur_data_right
		cur_data_center_y = .5 * cur_data_bottom + .5 * cur_data_top

		offset_x = scene_x / self.navMap_scaling_x - cur_data_center_x
		offset_y = scene_y / self.navMap_scaling_y - cur_data_center_y

		# print offset_x, offset_y

		new_data_left = cur_data_left + offset_x
		new_data_right = cur_data_right + offset_x
		new_data_bottom = cur_data_bottom + offset_y
		new_data_top = cur_data_top + offset_y

		# print new_data_left, new_data_right, new_data_bottom, new_data_top

		if new_data_right > self.dm.image_width:
			return
		if new_data_bottom > self.dm.image_height:
			return
		if new_data_left < 0:
			return
		if new_data_top < 0:
			return

		self.navRect.setRect(new_data_left * self.navMap_scaling_x, new_data_top * self.navMap_scaling_y, 
			self.navMap_scaling_x * (new_data_right - new_data_left), 
			self.navMap_scaling_y * (new_data_bottom - new_data_top))

		self.graphicsScene_navMap.update(0, 0, self.graphicsView_navMap.size().width(), self.graphicsView_navMap.size().height())
		self.graphicsView_navMap.setSceneRect(0, 0, self.dm.image_width*self.navMap_scaling_x, self.dm.image_height*self.navMap_scaling_y)

		self.axis.set_xlim([new_data_left, new_data_right])
		self.axis.set_ylim([new_data_bottom, new_data_top])

		self.canvas.draw()


	def save_selected_section(self, sec=None):

		timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

		if not hasattr(self, 'username') or self.username is None:
			username, okay = QInputDialog.getText(self, "Username", "Please enter your username:", QLineEdit.Normal, 'anon')
			if not okay: return
			self.username = str(username)
			self.lineEdit_username.setText(self.username)

		if sec is not None:

			# print self.accepted_proposals_allSections[sec]

			accepted_proposal_props = []
			for polygon, props in self.accepted_proposals_allSections[sec].iteritems():

				props_saved = props.copy()

				# props_saved['vertices'] = [(v.scenePos().x(), v.scenePos().y()) for v in props['vertexCircles']]

				path = polygon.path()

				if path.elementCount() > 1 and polygon_is_closed(path=path):
					props_saved['subtype'] = PolygonType.CLOSED
					props_saved['vertices'] = [(int(path.elementAt(i).x), int(path.elementAt(i).y)) for i in range(path.elementCount()-1)]
				else:
					props_saved['subtype'] = PolygonType.OPEN
					props_saved['vertices'] = [(int(path.elementAt(i).x), int(path.elementAt(i).y)) for i in range(path.elementCount())]

				label_pos = props['labelTextArtist'].scenePos()
				props_saved['labelPos'] = (label_pos.x(), label_pos.y())

				props_saved.pop('vertexCircles')
				props_saved.pop('labelTextArtist')

				accepted_proposal_props.append(props_saved)

			# print '#############'
			# print accepted_proposal_props

			labeling_path = self.dms[sec].save_proposal_review_result(accepted_proposal_props, self.username, timestamp, suffix='consolidated')

			# print self.new_labelnames
			self.dms[sec].add_labelnames(self.new_labelnames, os.environ['REPO_DIR']+'/visualization/newStructureNames.txt')

			self.statusBar().showMessage('Labelings saved to %s' % (self.username+'_'+timestamp))

			# if sec in self.gscenes:
			# 	pix = QPixmap(self.dms[sec].image_width/8, self.dms[sec].image_height/8)
			# 	painter = QPainter(pix)
				
			# 	self.gscenes[sec].render(painter, QRectF(0,0,self.dms[sec].image_width/8, self.dms[sec].image_height/8), 
			# 							QRectF(0,0,self.dms[sec].image_width, self.dms[sec].image_height))
			# 	pix.save(labeling_path[:-4] + '.jpg', "JPG")
			# 	print 'Preview image saved to', labeling_path[:-4] + '.jpg'
			# 	del painter
			# 	del pix


	def print_history(self, num_entries=5):
		print 'history:', [h['type'] for h in self.history_allSections[self.selected_section]][-5:]

	def save_history(self):
		'''
		Save edit history
		'''

		timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

		h = {}
		for sec, history_items in self.history_allSections.iteritems():
			new_history_items = []
			for hist_item in history_items:
				# new_hist_item = {}
				# for k, v in hist_item.iteritems():
				# 	if k == 'type':
				# 		new_hist_item[k] = v
				# 	elif k == 'mouse_moved':
				# 		new_hist_item[k] = v

				new_hist_item = hist_item.copy()

				if 'polygon' in new_hist_item:
					new_hist_item.pop('polygon')
				if 'new_polygon' in new_hist_item:
					new_hist_item.pop('new_polygon')
				if 'new_polygons' in new_hist_item:
					new_hist_item.pop('new_polygons')
				if 'vertex' in new_hist_item:
					new_hist_item.pop('vertex')

				new_history_items.append(new_hist_item)
			h[sec] = new_history_items

		log_filepath = '/home/yuncong/CSHL_labelingHistory/%(stack)s_%(timestamp)s.pkl' % {'stack': self.stack, 'timestamp': timestamp}

		print 'Log saved to', log_filepath

		pickle.dump(h, open(log_filepath, 'w'))

	def save_callback(self):
		'''
		Callback when save button is clicked
		'''

		for sec, ac in self.accepted_proposals_allSections.iteritems():
			if sec in self.gscenes and sec in self.dms:
				print sec
				self.save_selected_section(sec)

		self.save_history()

	############################################
	# matplotlib canvas CALLBACKs
	############################################


	def undo_add_vertex(self, polygon, index):

		vertex = self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles'][index]
		is_closed = polygon_is_closed(polygon=polygon)
		n = polygon_num_vertices(polygon=polygon, closed=is_closed)

		# path = QPainterPath()

		# if n == 1:
		# 	# if only one vertex in polygon, then undo removes the entire polygon
		# 	# self.section1_gscene.removeItem(polygon)
		# 	self.gscenes[self.selected_section].removeItem(polygon)
		# 	if 'labelTextArtist' in self.accepted_proposals_allSections[self.selected_section][polygon]:
		# 		# self.section1_gscene.removeItem(self.accepted_proposals_allSections[self.selected_section][polygon]['labelTextArtist'])
		# 		self.gscenes[self.selected_section].removeItem(self.accepted_proposals_allSections[self.selected_section][polygon]['labelTextArtist'])
		# 	self.accepted_proposals_allSections[self.selected_section].pop(polygon)
		# 	# self.section1_gscene.removeItem(vertex)
		# 	self.gscenes[self.selected_section].removeItem(vertex)

		# 	self.set_mode(Mode.IDLE)
		# else:

		# if it is open, then undo removes the last vertex

		self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles'].remove(vertex)
		self.gscenes[self.selected_section].removeItem(vertex)

		vs = vertices_from_polygon(polygon=polygon)
		if index == n - 1:
			new_vs = vs[:n-1]
		elif index == 0:
			new_vs = vs[1:]
		else:
			new_vs = np.r_[vs[:index], vs[index+1:]]
		
		print index
		print vs
		print new_vs

		new_path = vertices_to_path(new_vs, closed=is_closed)
		polygon.setPath(new_path)

	def undo_close_polygon(self, polygon):

		new_path = vertices_to_path(vertices_from_polygon(polygon=polygon, closed=True), closed=False)
		polygon.setPath(new_path)

	def undo_set_label(self, polygon):

		props = self.accepted_proposals_allSections[self.selected_section][polygon]
		props.pop('label')
		self.gscenes[self.selected_section].removeItem(props['labelTextArtist'])
		props.pop('labelTextArtist')


	def undo(self):

		if len(self.history_allSections[self.selected_section]) == 0:
			return

		history_item = self.history_allSections[self.selected_section].pop()

		print 'undo', history_item['type']

		if history_item['type'] == 'drag_polygon':

			polygon = history_item['polygon']
			moved_x, moved_y = history_item['mouse_moved']

			for circ in self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles']:
				curr_pos = circ.scenePos()
				circ.setPos(curr_pos.x() - moved_x, curr_pos.y() - moved_y)

			path = polygon.path()
			for i in range(polygon.path().elementCount()):
				elem = polygon.path().elementAt(i)
				scene_pos = polygon.mapToScene(elem.x, elem.y)
				path.setElementPositionAt(i, scene_pos.x() - moved_x, scene_pos.y() - moved_y)

			polygon.setPath(path)

			if 'labelTextArtist' in self.accepted_proposals_allSections[self.selected_section][polygon]:
				curr_label_pos = self.accepted_proposals_allSections[self.selected_section][polygon]['labelTextArtist'].scenePos()
				self.accepted_proposals_allSections[self.selected_section][polygon]['labelTextArtist'].setPos(curr_label_pos.x() - moved_x, curr_label_pos.y() - moved_y)

			# self.section1_gscene.update(0, 0, self.section1_gview.width(), self.section1_gview.height())
			self.gscenes[self.selected_section].update(0, 0, self.gviews[self.selected_panel_id].width(), self.gviews[self.selected_panel_id].height())

		elif history_item['type'] == 'drag_vertex':

			polygon = history_item['polygon']
			vertex = history_item['vertex']
			moved_x, moved_y = history_item['mouse_moved']

			curr_pos = vertex.scenePos()
			vertex.setPos(curr_pos.x() - moved_x, curr_pos.y() - moved_y)

			vertex_index = self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles'].index(vertex)

			path = polygon.path()
			elem_first = path.elementAt(0)
			elem_last = path.elementAt(path.elementCount()-1)
			is_closed = (elem_first.x == elem_last.x) & (elem_first.y == elem_last.y)

			if vertex_index == 0 and is_closed:
				path.setElementPositionAt(0, curr_pos.x() - moved_x, curr_pos.y() - moved_y)
				path.setElementPositionAt(len(self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles']), curr_pos.x() - moved_x, curr_pos.y() - moved_y)
			else:
				path.setElementPositionAt(vertex_index, curr_pos.x() - moved_x, curr_pos.y() - moved_y)

			polygon.setPath(path)

			# self.section1_gscene.update(0, 0, self.section1_gview.width(), self.section1_gview.height())
			self.gscenes[self.selected_section].update(0, 0, self.gviews[self.selected_panel_id].width(), self.gviews[self.selected_panel_id].height())

		elif history_item['type'] == 'close_polygon':

			self.undo_close_polygon(history_item['polygon'])

		elif history_item['type'] == 'add_vertex':

			self.undo_add_vertex(history_item['polygon'], history_item['new_index'])
			if polygon_num_vertices(history_item['polygon']) == 0:
				prev_item = self.history_allSections[self.selected_section].pop()
				assert prev_item['type'] == 'add_polygon'
				self.remove_polygon(history_item['polygon'], log=False) # do not log, since this is an undo action, not a regular action

		elif history_item['type'] == 'set_label':

			self.undo_set_label(history_item['polygon'])

			# pass
			# polygon = history_item['polygon']
			# index = history_item['new_index']

			# vertex = self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles'][index]
			# # vertex = history_item['vertex']

			# path = polygon.path()
			# elem_first = path.elementAt(0)
			# elem_last = path.elementAt(path.elementCount()-1)
			# is_closed = (elem_first.x == elem_last.x) & (elem_first.y == elem_last.y)
			# print 'is_closed', is_closed

			# path = QPainterPath()

			# n = len(self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles'])

			# if n == 1:
			# 	# if only one vertex in polygon, then undo removes the entire polygon
			# 	# self.section1_gscene.removeItem(polygon)
			# 	self.gscenes[self.selected_section].removeItem(polygon)
			# 	if 'labelTextArtist' in self.accepted_proposals_allSections[self.selected_section][polygon]:
			# 		# self.section1_gscene.removeItem(self.accepted_proposals_allSections[self.selected_section][polygon]['labelTextArtist'])
			# 		self.gscenes[self.selected_section].removeItem(self.accepted_proposals_allSections[self.selected_section][polygon]['labelTextArtist'])
			# 	self.accepted_proposals_allSections[self.selected_section].pop(polygon)
			# 	# self.section1_gscene.removeItem(vertex)
			# 	self.gscenes[self.selected_section].removeItem(vertex)

			# 	self.set_mode(Mode.IDLE)
			# else:

			# 	if not is_closed:
			# 		# if it is open, then undo removes the last vertex
			# 		self.accepted_proposals_allSections[self.selected_section][polygon]['vertexCircles'].remove(vertex)
			# 		# self.section1_gscene.removeItem(vertex)
			# 		self.gscenes[self.selected_section].removeItem(vertex)

			# 		for i in range(n-1):
			# 			elem = polygon.path().elementAt(i)
			# 			if i == 0:
			# 				path.moveTo(elem.x, elem.y)
			# 			else:
			# 				path.lineTo(elem.x, elem.y)
			# 	else:
			# 		# if it is closed, then undo opens it, without removing any vertex
			# 		for i in range(n):
			# 			elem = polygon.path().elementAt(i)
			# 			if i == 0:
			# 				path.moveTo(elem.x, elem.y)
			# 			else:
			# 				path.lineTo(elem.x, elem.y)

			# 	polygon.setPath(path)

		elif history_item['type'] == 'set_uncertain_segment':
			pass

			# old_polygon = history_item['old_polygon']
			# new_certain_polygon = history_item['new_certain_polygon']
			# new_uncertain_polygon = history_item['new_uncertain_polygon']

			# label = self.accepted_proposals_allSections[self.selected_section][new_certain_polygon]['label']

			# self.remove_polygon(new_certain_polygon)
			# self.remove_polygon(new_uncertain_polygon)

			# # self.section1_gscene.addItem(old_polygon)
			# self.gscenes[self.selected_section].removeItem(vertex)
			# overlap_polygons = self.populate_polygon_with_vertex_circles(old_polygon)
			# self.restack_polygons(old_polygon, overlap_polygons)
			# self.add_label_to_polygon(old_polygon, label=label)

		elif history_item['type'] == 'add_polygon':
			self.remove_polygon(history_item['polygon'])

		elif history_item['type'] == 'remove_polygon':
			pass

		elif history_item['type'] == 'delete_vertices_merge':
			self.remove_polygon(history_item['new_polygon'])
			self.add_polygon_by_vertices_label(history_item['polygon'])
			
		self.print_history()

	def set_mode(self, mode):

		if hasattr(self, 'mode'):
			if self.mode != mode:
				print self.mode, '=>', mode
		else:
			print mode

		self.mode = mode

		if mode == Mode.MOVING_VERTEX:
			self.set_flag_all(QGraphicsItem.ItemIsMovable, True)
		else:
			self.set_flag_all(QGraphicsItem.ItemIsMovable, False)

		if mode == Mode.SELECT_UNCERTAIN_SEGMENT or mode == Mode.DELETE_ROI_MERGE or mode == Mode.DELETE_ROI_DUPLICATE:
			self.gviews[self.selected_panel_id].setDragMode(QGraphicsView.RubberBandDrag)
		else:
			if hasattr(self, 'selected_panel_id'): # right after launch, selected_panel_id is not set
				self.gviews[self.selected_panel_id].setDragMode(QGraphicsView.NoDrag)

		self.statusBar().showMessage(self.mode.value)

		
	def update_navMap(self):

		cur_xmin, cur_xmax = self.axis.get_xlim()
		cur_ybottom, cur_ytop = self.axis.get_ylim()
		self.navRect.setRect(cur_xmin * self.navMap_scaling_x, cur_ybottom * self.navMap_scaling_y, self.navMap_scaling_x * (cur_xmax - cur_xmin), self.navMap_scaling_y * (cur_ytop - cur_ybottom))
		self.graphicsScene_navMap.update(0, 0, self.graphicsView_navMap.size().width(), self.graphicsView_navMap.size().height())
		self.graphicsView_navMap.setSceneRect(0, 0, self.dm.image_width*self.navMap_scaling_x, self.dm.image_height*self.navMap_scaling_y)



	def find_proper_offset(self, offset_x, offset_y):

		if self.cur_ylim[0] - offset_y > self.dm.image_height:
			offset_y = self.cur_ylim[0] - self.dm.image_height
		elif self.cur_ylim[1] - offset_y < 0:
			offset_y = self.cur_ylim[1]

		if self.cur_xlim[1] - offset_x > self.dm.image_width:
			offset_x = self.dm.image_width - self.cur_xlim[1]
		elif self.cur_xlim[0] - offset_x < 0:
			offset_x = self.cur_xlim[0]

		return offset_x, offset_y


	def split_path(self, path, vertex_indices):
		'''
		Split path.

		Args:
			path (QPainterPath): input path
			vertex_indices (list of tuples or n x 2 numpy array): indices in the cut box

		Returns:
			list of QPainterPath: paths in cut box
			list of QPainterPath: paths outside of cut box

		'''

		is_closed = polygon_is_closed(path=path)
		n = polygon_num_vertices(path=path, closed=is_closed)

		segs_in, segs_out = self.split_array(vertex_indices, n, is_closed)

		print segs_in, segs_out

		in_paths = []
		out_paths = []

		for b, e in segs_in:
			in_path = self.subpath(path, b, e)
			in_paths.append(in_path)

		for b, e in segs_out:
			out_path = self.subpath(path, b-1, e+1)
			out_paths.append(out_path)

		return in_paths, out_paths

	def split_array(self, vertex_indices, n, is_closed):

		cache = [i in vertex_indices for i in range(n)]

		i = 0

		sec_outs = []
		sec_ins = []

		sec_in = [None,None]
		sec_out = [None,None]

		while i != (n+1 if is_closed else n):

			if cache[i%n] and not cache[(i+1)%n]:
				sec_in[1] = i%n
				sec_ins.append(sec_in)
				sec_in = [None,None]

				sec_out[0] = (i+1)%n
			elif not cache[i%n] and cache[(i+1)%n]:
				sec_out[1] = i%n
				sec_outs.append(sec_out)
				sec_out = [None,None]

				sec_in[0] = (i+1)%n
			
			i += 1

		if sec_in[0] is not None or sec_in[1] is not None:
			sec_ins.append(sec_in)

		if sec_out[0] is not None or sec_out[1] is not None:
			sec_outs.append(sec_out)

		tmp = [None, None]
		for sec in sec_ins:
			if sec[0] is None and sec[1] is not None:
				tmp[1] = sec[1]
			elif sec[0] is not None and sec[1] is None:
				tmp[0] = sec[0]
		if tmp[0] is not None and tmp[1] is not None:
			sec_ins = [s for s in sec_ins if s[0] is not None and s[1] is not None] + [tmp]
		else:
			sec_ins = [s for s in sec_ins if s[0] is not None and s[1] is not None]

		tmp = [None, None]
		for sec in sec_outs:
			if sec[0] is None and sec[1] is not None:
				tmp[1] = sec[1]
			elif sec[0] is not None and sec[1] is None:
				tmp[0] = sec[0]
		if tmp[0] is not None and tmp[1] is not None:
			sec_outs = [s for s in sec_outs if s[0] is not None and s[1] is not None] + [tmp]
		else:
			sec_outs = [s for s in sec_outs if s[0] is not None and s[1] is not None]

		if not is_closed:
			sec_ins2 = []
			for sec in sec_ins:
				if sec[0] > sec[1]:
					sec_ins2.append([sec[0], n-1])
					sec_ins2.append([0, sec[1]])
				else:
					sec_ins2.append(sec)

			sec_outs2 = []
			for sec in sec_outs:
				if sec[0] > sec[1]:
					sec_outs2.append([sec[0], n-1])
					sec_outs2.append([0, sec[1]])
				else:
					sec_outs2.append(sec)

			return sec_ins2, sec_outs2

		else:
			return sec_ins, sec_outs


	# def is_path_closed(self, path):

	# 	elem_first = path.elementAt(0)
	# 	elem_last = path.elementAt(path.elementCount()-1)
	# 	is_closed = (elem_first.x == elem_last.x) & (elem_first.y == elem_last.y)

	# 	return is_closed

	def delete_vertices(self, polygon, indices_to_remove, merge=False):

		if merge:
			new_polygon = self.delete_vertices_merge(polygon, indices_to_remove)

			self.history_allSections[self.selected_section].append({
				'type': 'delete_vertices_merge',
				'polygon': polygon,
				'new_polygon': new_polygon,
				'indices_to_remove': indices_to_remove,
				'label': self.accepted_proposals_allSections[self.selected_section][new_polygon]['label']
				})

		else:
			paths_to_remove, paths_to_keep = self.split_path(polygon.path(), indices_to_remove)

			new_polygons = []
			for path in paths_to_keep:
				new_polygon = self.add_polygon_by_vertices_label(path, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon]['label'])
				new_polygons.append(new_polygon)

			self.remove_polygon(polygon)

			self.history_allSections[self.selected_section].append({
				'type': 'delete_vertices_split',
				'polygon': polygon,
				'new_polygons': new_polygons,
				'indices_to_remove': indices_to_remove,
				'label': self.accepted_proposals_allSections[self.selected_section][new_polygons[0]]['label']
				})


	def delete_between(self, polygon, first_index, second_index):

		print first_index, second_index

		if second_index < first_index:	# ensure first_index is smaller than second_index
			temp = first_index
			first_index = second_index
			second_index = temp

		path = polygon.path()

		n = polygon_num_vertices(path=path)

		if (second_index - first_index > first_index + n - second_index):
			indices_to_remove = range(second_index, n+1) + range(0, first_index+1)
		else:
			indices_to_remove = range(first_index, second_index+1)

		print indices_to_remove

		paths_to_remove, paths_to_keep = self.split_path(path, indices_to_remove)

		for new_path in paths_to_keep:

			self.add_polygon_by_vertices_label(new_path, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon]['label'])	

		self.remove_polygon(polygon)


	def delete_vertices_merge(self, polygon, indices_to_remove):

		path = polygon.path()

		is_closed = polygon_is_closed(path=path)
		n = polygon_num_vertices(path=path, closed=is_closed)

		segs_to_remove, segs_to_keep = self.split_array(indices_to_remove, n, is_closed)
		print segs_to_remove, segs_to_keep

		new_path = QPainterPath()
		for b, e in sorted(segs_to_keep):
			if e < b: e = e + n
			for i in range(b, e + 1):
				elem = path.elementAt(i % n)
				if new_path.elementCount() == 0:
					new_path.moveTo(elem.x, elem.y)
				else:
					new_path.lineTo(elem.x, elem.y)

		if is_closed:
			new_path.closeSubpath()
				
		new_polygon = self.add_polygon_by_vertices_label(new_path, pen=self.red_pen, label=self.accepted_proposals_allSections[self.selected_section][polygon]['label'])
		
		self.remove_polygon(polygon)

		return new_polygon
			

	def auto_extend_view(self, x, y):
		# always make just placed vertex at the center of the view

		scene_rect = self.section1_gview.mapToScene(self.section1_gview.viewport().rect()).boundingRect()
		cur_xmin = scene_rect.x()
		cur_ymin = scene_rect.y()
		cur_xmax = cur_xmin + scene_rect.width()
		cur_ymax = cur_ymin + scene_rect.height()

		if abs(x - cur_xmin) < AUTO_EXTEND_VIEW_TOLERANCE or abs(x - cur_xmax) < AUTO_EXTEND_VIEW_TOLERANCE:
			cur_xcenter = cur_xmin * .6 + cur_xmax * .4 if abs(x - cur_xmin) < AUTO_EXTEND_VIEW_TOLERANCE else cur_xmin * .4 + cur_xmax * .6
			translation_x = cur_xcenter - x

			self.section1_gview.translate(translation_x, 0)
			self.section2_gview.translate(translation_x, 0)
			self.section3_gview.translate(translation_x, 0)

			# print 'translation_x', translation_x

		if abs(y - cur_ymin) < AUTO_EXTEND_VIEW_TOLERANCE or abs(y - cur_ymax) < AUTO_EXTEND_VIEW_TOLERANCE:
			cur_ycenter = cur_ymin * .6 + cur_ymax * .4 if abs(y - cur_ymin) < AUTO_EXTEND_VIEW_TOLERANCE else cur_ymin * .4 + cur_ymax * .6
			translation_y = cur_ycenter - y

			self.section1_gview.translate(0, translation_y)
			self.section2_gview.translate(0, translation_y)
			self.section3_gview.translate(0, translation_y)

			# print 'translation_y', translation_y


	def turn_superpixels_off(self):
		self.statusBar().showMessage('Supepixels OFF')

		self.buttonSpOnOff.setText('Turn Superpixels ON')

		self.segm_handle.remove()
		self.superpixels_on = False
		
		# self.axis.imshow(self.masked_img, cmap=plt.cm.Greys_r,aspect='equal')
		# self.orig_image_handle = self.axis.imshow(self.masked_img, aspect='equal')

	def turn_superpixels_on(self):
		self.statusBar().showMessage('Supepixels ON')

		self.buttonSpOnOff.setText('Turn Superpixels OFF')

		if self.segm_transparent is None:
			self.segm_transparent = self.dm.load_pipeline_result('segmentationTransparent')
			self.my_cmap = plt.cm.Reds
			self.my_cmap.set_under(color="white", alpha="0")

		if not self.seg_loaded:
			self.load_segmentation()

		self.superpixels_on = True
		
		if hasattr(self, 'segm_handle'):
			self.segm_handle.set_data(self.segm_transparent)
		else:
			self.segm_handle = self.axis.imshow(self.segm_transparent, aspect='equal', 
								cmap=self.my_cmap, alpha=1.)



	def cancel_current_circle(self):

		if self.selected_circle is not None:
			self.selected_circle.set_radius(UNSELECTED_CIRCLE_SIZE)
			self.selected_circle = None

			self.selected_vertex_index = None


	def cancel_current_selection(self):

		if self.selected_proposal_polygon is not None:

			if self.selected_proposal_polygon.get_linewidth() != UNSELECTED_POLYGON_LINEWIDTH:
				self.selected_proposal_polygon.set_linewidth(UNSELECTED_POLYGON_LINEWIDTH)

			if self.selected_proposal_polygon in self.axis.patches:
				if self.selected_proposal_polygon not in self.accepted_proposals_allSections[self.selected_section]:
					self.selected_proposal_polygon.remove()
					for vertex_circ in self.selected_proposal_vertexCircles:
						vertex_circ.remove()

		self.selected_proposal_polygon = None
		self.selected_proposal_vertexCircles = None

		if self.selected_circle is not None:
			self.selected_circle.set_radius(UNSELECTED_CIRCLE_SIZE)
			self.selected_circle = None

			self.selected_vertex_index = None


		self.canvas.draw()


	def mode_changed(self):

		self.cancel_current_selection()

		if self.radioButton_globalProposal.isChecked():

			self.shuffle_global_proposals = True

			if not self.superpixels_on:
				self.turn_superpixels_on()

			if not hasattr(self, 'global_proposal_tuples'):
				self.load_global_proposals()

		elif self.radioButton_localProposal.isChecked():

			if not self.superpixels_on:
				self.turn_superpixels_on()

			self.shuffle_global_proposals = False

			if not hasattr(self, 'local_proposal_tuples'):
				self.load_local_proposals()

		self.canvas.draw()

	# def display_option_changed(self):
	# 	if self.sender() == self.buttonSpOnOff:

	# 		if not self.superpixels_on:
	# 			self.turn_superpixels_on()
	# 		else:
	# 			self.turn_superpixels_off()
	# 	else:
	# 		print 'not implemented'
	# 		return

	# 		# if self.under_img is not None:
	# 		#   self.under_img.remove()

	# 		self.axis.clear()

	# 		if self.sender() == self.img_radioButton:

	# 			# self.axis.clear()
	# 			# self.axis.axis('off')

	# 			# self.under_img = self.axis.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
	# 			self.axis.imshow(self.dm.image_rgb_jpg, aspect='equal', cmap=plt.cm.Greys_r)
	# 			# self.superpixels_on = False

	# 		elif self.sender() == self.textonmap_radioButton:

	# 			# self.axis.clear()
	# 			# self.axis.axis('off')

	# 			if self.textonmap_vis is None:
	# 				self.textonmap_vis = self.dm.load_pipeline_result('texMapViz')

	# 			# if self.under_img is not None:
	# 			#   self.under_img.remove()

	# 			# self.under_img = self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
	# 			self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
	# 			# self.superpixels_on = False

	# 		elif self.sender() == self.dirmap_radioButton:

	# 			# self.axis.clear()
	# 			# self.axis.axis('off')

	# 			if self.dirmap_vis is None:
	# 				self.dirmap_vis = self.dm.load_pipeline_result('dirMap', 'jpg')
	# 				self.dirmap_vis[~self.dm.mask] = 0


	# 			# self.under_img = self.axis.imshow(self.dirmap_vis, aspect='equal')
	# 			self.axis.imshow(self.dirmap_vis, aspect='equal')

	# 			# if not self.seg_loaded:
	# 			#   self.load_segmentation()

	# 			# self.superpixels_on = False

	# 		# elif self.sender() == self.labeling_radioButton:
	# 		#   pass

	# 	self.axis.axis('off')
	# 	# self.axis.set_xlim([self.newxmin, self.newxmax])
	# 	# self.axis.set_ylim([self.newymin, self.newymax])
	# 	# self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
	# 	self.canvas.draw()

	# 	self.axis2.axis('off')
	# 	# self.axis2.set_xlim([self.newxmin, self.newxmax])
	# 	# self.axis2.set_ylim([self.newymin, self.newymax])
	# 	# self.fig2.subplots_adjust(left=0, bottom=0, right=1, top=1)
	# 	self.canvas2.draw()

			   
if __name__ == "__main__":
	from sys import argv, exit
	appl = QApplication(argv)

	import argparse
	import sys
	import time

	parser = argparse.ArgumentParser(
	    formatter_class=argparse.RawDescriptionHelpFormatter,
	    description='Compute texton map')

	parser.add_argument("stack_name", type=str, help="stack name")
	parser.add_argument("-n", "--num_neighbors", type=int, help="number of neighbor sections to preload, default %(default)d", default=1)
	args = parser.parse_args()

	stack = args.stack_name
	NUM_NEIGHBORS_PRELOAD = args.num_neighbors
	m = BrainLabelingGUI(stack=stack)

	m.showMaximized()
	m.raise_()
	exit(appl.exec_())
