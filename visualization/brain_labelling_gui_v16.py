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

from matplotlib.backend_bases import key_press_handler, MouseEvent, KeyEvent
from matplotlib.backends.backend_qt4agg import (
	FigureCanvasQTAgg as FigureCanvas,
	NavigationToolbar2QT as NavigationToolbar)
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

from ui_BrainLabelingGui_v11 import Ui_BrainLabelingGui

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib.colors import ListedColormap, NoNorm, ColorConverter
from matplotlib.path import Path
from matplotlib.text import Text

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import LinearRing as ShapelyLineRing

from skimage.color import label2rgb

from visualization_utilities import *

sys.path.append(os.environ['LOCAL_REPO_DIR'] + '/utilities')
from utilities2015 import *

from collections import defaultdict, OrderedDict, deque
from scipy.spatial.distance import cdist

from operator import attrgetter

import requests

from joblib import Parallel, delayed

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

SELECTED_POLYGON_LINEWIDTH = 5
UNSELECTED_POLYGON_LINEWIDTH = 3
SELECTED_CIRCLE_SIZE = 30
UNSELECTED_CIRCLE_SIZE = 5
CIRCLE_PICK_THRESH = 1000.
PAN_THRESHOLD = 10

HISTORY_LEN = 20

AUTO_EXTEND_VIEW_TOLERANCE = 200

# NUM_NEIGHBORS_PRELOAD = 1 # preload neighbor sections before and after this number
VERTEX_CIRCLE_RADIUS = 10

class ListSelection(QDialog):
	def __init__(self, item_ls, parent=None):
		super(ListSelection, self).__init__(parent)

		self.setWindowTitle('Detect which landmarks ?')

		self.selected = set([])

		self.listWidget = QListWidget()
		for item in item_ls:    
			w_item = QListWidgetItem(item)
			self.listWidget.addItem(w_item)

			w_item.setFlags(w_item.flags() | Qt.ItemIsUserCheckable)
			w_item.setCheckState(False)

		self.listWidget.itemChanged.connect(self.OnSingleClick)

		layout = QGridLayout()
		layout.addWidget(self.listWidget,0,0,1,3)

		self.but_ok = QPushButton("OK")
		layout.addWidget(self.but_ok ,1,1)
		self.but_ok.clicked.connect(self.OnOk)

		self.but_cancel = QPushButton("Cancel")
		layout.addWidget(self.but_cancel ,1,2)
		self.but_cancel.clicked.connect(self.OnCancel)

		self.setLayout(layout)
		self.setGeometry(300, 200, 460, 350)

	def OnSingleClick(self, item):
		if not item.checkState():
		#   item.setCheckState(False)
			self.selected = self.selected - {str(item.text())}
		#   print self.selected
		else:
		#   item.setCheckState(True)
			self.selected.add(str(item.text()))

		print self.selected

	def OnOk(self):
		self.close()

	def OnCancel(self):
		self.selected = set([])
		self.close()


class CustomQCompleter(QCompleter):
	# adapted from http://stackoverflow.com/a/26440173
	def __init__(self, *args):#parent=None):
		super(CustomQCompleter, self).__init__(*args)
		self.local_completion_prefix = ""
		self.source_model = None
		self.filterProxyModel = QSortFilterProxyModel(self)
		self.usingOriginalModel = False

	def setModel(self, model):
		self.source_model = model
		self.filterProxyModel = QSortFilterProxyModel(self)
		self.filterProxyModel.setSourceModel(self.source_model)
		super(CustomQCompleter, self).setModel(self.filterProxyModel)
		self.usingOriginalModel = True

	def updateModel(self):
		if not self.usingOriginalModel:
			self.filterProxyModel.setSourceModel(self.source_model)

		pattern = QRegExp(self.local_completion_prefix,
								Qt.CaseInsensitive,
								QRegExp.FixedString)

		self.filterProxyModel.setFilterRegExp(pattern)

	def splitPath(self, path):
		self.local_completion_prefix = path
		self.updateModel()
		if self.filterProxyModel.rowCount() == 0:
			self.usingOriginalModel = False
			self.filterProxyModel.setSourceModel(QStringListModel([path]))
			return [path]

		return []

class AutoCompleteComboBox(QComboBox):
	# adapted from http://stackoverflow.com/a/26440173
	def __init__(self, labels, *args, **kwargs):
		super(AutoCompleteComboBox, self).__init__(*args, **kwargs)

		self.setEditable(True)
		self.setInsertPolicy(self.NoInsert)

		self.comp = CustomQCompleter(self)
		self.comp.setCompletionMode(QCompleter.PopupCompletion)
		self.setCompleter(self.comp)#
		self.setModel(labels)

		self.clearEditText()

	def setModel(self, strList):
		self.clear()
		self.insertItems(0, strList)
		self.comp.setModel(self.model())

	def focusInEvent(self, event):
		# self.clearEditText()
		super(AutoCompleteComboBox, self).focusInEvent(event)

	def keyPressEvent(self, event):
		key = event.key()
		if key == Qt.Key_Return:

			# make sure that the completer does not set the
			# currentText of the combobox to "" when pressing enter
			text = self.currentText()
			self.setCompleter(None)
			self.setEditText(text)
			self.setCompleter(self.comp)

		return super(AutoCompleteComboBox, self).keyPressEvent(event)

class AutoCompleteInputDialog(QDialog):

	def __init__(self, labels, *args, **kwargs):
		super(AutoCompleteInputDialog, self).__init__(*args, **kwargs)
		self.comboBox = AutoCompleteComboBox(parent=self, labels=labels)
		va = QVBoxLayout(self)
		va.addWidget(self.comboBox)
		box = QWidget(self)
		ha = QHBoxLayout(self)
		va.addWidget(box)
		box.setLayout(ha)
		self.OK = QPushButton("OK", self)
		self.OK.setDefault(True)
		# cancel = QPushButton("Cancel", self)
		ha.addWidget(self.OK)
		# ha.addWidget(cancel)

	def set_test_callback(self, callback):
		self.OK.clicked.connect(callback)
		# OK.clicked.connect(self.accept)
		# cancel.clicked.connect(self.reject)

class SignalEmitter(QObject):
    moved = pyqtSignal(int, int, int, int)
    clicked = pyqtSignal()
    released = pyqtSignal()
    
    def __init__(self, parent):
        super(SignalEmitter, self).__init__()
        self.parent = parent


class QGraphicsPathItemModified(QGraphicsPathItem):

	def __init__(self, parent=None, gui=None, *args, **kwargs):
		super(self.__class__, self).__init__(parent, *args, **kwargs)
		self.signal_emitter = SignalEmitter(parent=self)
		# self.just_created = True # this flag is used to make sure a click is not emitted right after item creation
								# basically, ignore the first press and release event
		self.gui = gui

	def mousePressEvent(self, event):

	# if not self.just_created:
		print self, 'received mousePressEvent'

		self.press_scene_x = event.scenePos().x()
		self.press_scene_y = event.scenePos().y()

		self.center_scene_x_before_move = self.scenePos().x()
		self.center_scene_y_before_move = self.scenePos().y()

		self.gui.selected_polygon = self

		QGraphicsPathItem.mousePressEvent(self, event)

		self.signal_emitter.clicked.emit()

		if 'labelTextArtist' in self.gui.accepted_proposals[self.gui.selected_polygon]:
			label_pos_before_move = self.gui.accepted_proposals[self.gui.selected_polygon]['labelTextArtist'].scenePos()
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

	def __init__(self, parent=None, *args, **kwargs):
		super(self.__class__, self).__init__(parent, *args, **kwargs)
		self.signal_emitter = SignalEmitter(parent=self)
		self.just_created = True # this flag is used to make sure a click is not emitted right after item creation
								# basically, ignore the first press and release event

	def mousePressEvent(self, event):
		if not self.just_created:
			print self, 'received mousePressEvent'
			QGraphicsEllipseItem.mousePressEvent(self, event)
			self.signal_emitter.clicked.emit()

			self.press_scene_x = event.scenePos().x()
			self.press_scene_y = event.scenePos().y()

			self.center_scene_x_before_move = self.scenePos().x()
			self.center_scene_y_before_move = self.scenePos().y()

		self.just_created = False

	def mouseReleaseEvent(self, event):
		if not self.just_created:
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
		print self, 'received mouseMoveEvent'
		self.signal_emitter.moved.emit(event.scenePos().x(), event.scenePos().y(), self.press_scene_x, self.press_scene_y)
		QGraphicsEllipseItem.mouseMoveEvent(self, event)


def load_dms(i):
	return DataManager(data_dir=os.environ['LOCAL_DATA_DIR'], 
		         repo_dir=os.environ['LOCAL_REPO_DIR'], 
		         result_dir=os.environ['LOCAL_RESULT_DIR'], 
		         labeling_dir=os.environ['LOCAL_LABELING_DIR'],
		    stack=stack, section=i, segm_params_id='tSLIC200', load_mask=False)
	
def load_pixmap(dm):
	return QPixmap(dm._get_image_filepath(version='rgb-jpg'))

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
		self.initialize_brain_labeling_gui()

		self.recent_labels = []

		self.history = deque(maxlen=HISTORY_LEN)

	def init_data(self, section):

		self.section = section
		self.section2 = section + 1
		self.section3 = section - 1

		sections = range(self.section-NUM_NEIGHBORS_PRELOAD, self.section+NUM_NEIGHBORS_PRELOAD+1)

		# t = time.time()
		# self.dms = dict(zip(sections, Parallel(n_jobs=4)(delayed(load_dms)(i) for i in sections)))
		# print 'load shape', time.time() - t

		self.dms = dict([(i, DataManager(
		    data_dir=os.environ['LOCAL_DATA_DIR'], 
		         repo_dir=os.environ['LOCAL_REPO_DIR'], 
		         result_dir=os.environ['LOCAL_RESULT_DIR'], 
		         labeling_dir=os.environ['LOCAL_LABELING_DIR'],
		    stack=stack, section=i, segm_params_id='tSLIC200', load_mask=False)) for i in sections])

		self.dm = self.dms[section]

		t = time.time()
		# self.pixmaps = dict(zip(self.dms.keys(), Parallel(n_jobs=4)(delayed(load_pixmap)(dm, parent=self) for dm in self.dms.itervalues())))
		# self.pixmaps = dict(zip(self.dms.keys(), Parallel(n_jobs=4)(delayed(QPixmap)(dm._get_image_filepath(version='rgb-jpg'), parent=self) for dm in self.dms.itervalues())))
		# RuntimeError: super-class __init__() of type QPixmap was never called
		# or pickle.PicklingError: Can't pickle <type 'instancemethod'>: it's not found as __builtin__.instancemethod

		if hasattr(self, 'pixmaps'):
			for i in sections:
				if i not in self.pixmaps:
					print 'new load', i
					self.pixmaps[i] = QPixmap(self.dms[i]._get_image_filepath(version='rgb-jpg'))

			to_remove = []
			for i in self.pixmaps:
				if i not in sections:
					print 'pop', i
					to_remove.append(i)
			
			for i in to_remove:
				self.pixmaps.pop(i)
		else:
			self.pixmaps = dict([(i, QPixmap(dm._get_image_filepath(version='rgb-jpg'))) for i, dm in self.dms.iteritems()])
		print 'load image', time.time() - t

		self.new_labelnames = {}
		if os.path.exists(self.dm.repo_dir+'/visualization/newStructureNames.txt'):
			with open(self.dm.repo_dir+'/visualization/newStructureNames.txt', 'r') as f:
				for ln in f.readlines():
					abbr, fullname = ln.split('\t')
					self.new_labelnames[abbr] = fullname.strip()
			self.new_labelnames = OrderedDict(sorted(self.new_labelnames.items()))

		self.structure_names = {}
		with open(self.dm.repo_dir+'/visualization/structure_names.txt', 'r') as f:
			for ln in f.readlines():
				abbr, fullname = ln.split('\t')
				self.structure_names[abbr] = fullname.strip()
		self.structure_names = OrderedDict(self.new_labelnames.items() + sorted(self.structure_names.items()))

		print self.dm.slice_ind

		self.segm_transparent = None
		self.under_img = None
		self.textonmap_vis = None
		self.dirmap_vis = None

		self.set_mode(Mode.IDLE)

		self.boundary_colors = [(0,1,1), (1,0,0), (0,0,0),(0,0,1)] # unknown, accepted, rejected

		self.accepted_proposals = defaultdict(dict)

		self.selected_proposal_polygon = None
		self.alternative_global_proposal_ind = 0
		self.alternative_local_proposal_ind = 0

		# self.just_added_vertex = False

	def paramSettings_clicked(self):
		pass

	def reload_brain_labeling_gui(self):

		self.extend_head = False
		self.connecting_vertices = False

		self.seg_loaded = False
		self.superpixels_on = False
		self.labels_on = True
		self.contours_on = True
		self.vertices_on = True

		self.shuffle_global_proposals = True # instead of local proposals

		self.pressed = False           # related to pan (press and drag) vs. select (click)
		
		self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', middle section %d' %self.section + ', left %d'%self.section3 + ', right %d'%self.section2)

		t = time.time()

		self.section1_pixmap = QPixmap(self.dm._get_image_filepath(version='rgb-jpg'))

		self.section1_gscene = QGraphicsScene(self.section1_gview)
		self.section1_gscene.addPixmap(self.section1_pixmap)

		self.section1_gview.setScene(self.section1_gscene)

		self.section1_gscene.update(0, 0, self.section1_gview.width(), self.section1_gview.height())

		self.section1_gscene.keyPressEvent = self.key_pressed

		self.section1_gview.viewport().installEventFilter(self)
		self.section1_gscene.installEventFilter(self)

		self.section1_gview.setMouseTracking(False)

		self.section1_gview.setVerticalScrollBarPolicy( Qt.ScrollBarAlwaysOff ) 
		self.section1_gview.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOff ) 
		
		self.section1_gview.setAlignment(Qt.AlignLeft | Qt.AlignTop)

		# self.section1_gview.setInteractive(True)
		# self.section1_gview.setDragMode(QGraphicsView.RubberBandDrag)

		# self.section1_gview.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

		self.section1_gview.show()

		self.red_pen = QPen(Qt.red)
		self.red_pen.setWidth(20)
		self.blue_pen = QPen(Qt.blue)
		self.blue_pen.setWidth(20)
		self.green_pen = QPen(Qt.green)
		self.green_pen.setWidth(20)

		self.section1_gview.setTransformationAnchor(QGraphicsView.NoAnchor)

		self.section1_gview.setContextMenuPolicy(Qt.CustomContextMenu)
		self.section1_gview.customContextMenuRequested.connect(self.showContextMenu)

		#####

		self.section2_pixmap = self.pixmaps[self.section2]

		self.section2_gscene = QGraphicsScene(self.section2_gview)
		self.section2_gscene.addPixmap(self.section2_pixmap)

		self.section2_gview.setScene(self.section2_gscene)
		self.section2_gscene.update(0, 0, self.section2_gview.width(), self.section2_gview.height())

		self.section2_gview.viewport().installEventFilter(self)

		self.section2_gview.setVerticalScrollBarPolicy( Qt.ScrollBarAlwaysOff ) 
		self.section2_gview.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOff ) 
		
		self.section2_gview.setAlignment(Qt.AlignLeft | Qt.AlignTop)

		self.section2_gview.setTransformationAnchor(QGraphicsView.NoAnchor)

		self.section2_gview.show()

		####

		self.section3_pixmap =  self.pixmaps[self.section3]

		self.section3_gscene = QGraphicsScene(self.section3_gview)
		self.section3_gscene.addPixmap(self.section3_pixmap)

		self.section3_gview.setScene(self.section3_gscene)
		self.section3_gscene.update(0, 0, self.section3_gview.width(), self.section3_gview.height())

		self.section3_gview.setVerticalScrollBarPolicy( Qt.ScrollBarAlwaysOff ) 
		self.section3_gview.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOff ) 

		self.section3_gview.viewport().installEventFilter(self)
		
		self.section3_gview.setAlignment(Qt.AlignLeft | Qt.AlignTop)

		self.section3_gview.setTransformationAnchor(QGraphicsView.NoAnchor)

		self.section3_gview.show()

		print 4, time.time() - t

		self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', middle section %d' %self.section + ', left %d'%self.section3 + ', right %d'%self.section2)

		self.set_mode(Mode.IDLE)

		self.show()

	def initialize_brain_labeling_gui(self):

		# self.menu = QMenu()

		# self.addVertex_Action = self.menu.addAction("Insert vertices")
		# self.extendPolygon_Action = self.menu.addAction("Extend from this vertex")
		# self.doneAddingVertex_Action = self.menu.addAction("Done adding vertex")

		# self.movePolygon_Action = self.menu.addAction("Move")

		# self.deleteVertex_Action = self.menu.addAction("Delete vertex")

		# self.deleteVerticesROI_Action = self.menu.addAction("Delete vertices in ROI (close)")
		# self.deleteVerticesROIOpen_Action = self.menu.addAction("Delete vertices in ROI (open)")

		# self.selectROI_Action = self.menu.addAction("Select ROI")

		# self.breakEdge_Action = self.menu.addAction("break edge")

		# self.newPolygon_Action = self.menu.addAction("New polygon")

		# self.accProp_Action = self.menu.addAction("Accept")
		# self.rejProp_Action = self.menu.addAction("Reject")

		# self.changeLabel_Action = self.menu.addAction('Change label')

		# A set of high-contrast colors proposed by Green-Armytage
		self.colors = np.loadtxt('100colors.txt', skiprows=1)
		self.label_cmap = ListedColormap(self.colors, name='label_cmap')

		self.setupUi(self)

		self.button_autoDetect.clicked.connect(self.autoDetect_callback)
		self.button_updateDB.clicked.connect(self.updateDB_callback)
		self.button_loadLabeling.clicked.connect(self.load_callback)
		self.button_saveLabeling.clicked.connect(self.save_callback)
		self.button_quit.clicked.connect(self.close)
		
		self.display_buttons = [self.img_radioButton, self.textonmap_radioButton, self.dirmap_radioButton]
		self.img_radioButton.setChecked(True)

		for b in self.display_buttons:
			b.toggled.connect(self.display_option_changed)

		self.radioButton_globalProposal.toggled.connect(self.mode_changed)
		self.radioButton_localProposal.toggled.connect(self.mode_changed)

		self.buttonSpOnOff.clicked.connect(self.display_option_changed)
		self.button_labelsOnOff.clicked.connect(self.toggle_labels)
		self.button_contoursOnOff.clicked.connect(self.toggle_contours)
		self.button_verticesOnOff.clicked.connect(self.toggle_vertices)

		# self.thumbnail_list = QListWidget(parent=self)
		self.thumbnail_list.setIconSize(QSize(200,200))
		self.thumbnail_list.setResizeMode(QListWidget.Adjust)
		self.thumbnail_list.itemDoubleClicked.connect(self.section_changed)

		section_range_lookup = {'MD593': (41,176), 'MD594': (47,186), 'MD595': (35,164), 'MD592': (46,185), 'MD589':(49,186)}
		first_sec, last_sec = section_range_lookup[self.stack]
		for i in range(first_sec, last_sec):
			item = QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnail_aligned_cropped.tif"%{'sec':i, 'stack': self.stack}), str(i))
			# item.setFont(QFont())
			self.thumbnail_list.addItem(item)

		self.thumbnail_list.resizeEvent = self.thumbnail_list_resized
		self.init_thumbnail_list_width = self.thumbnail_list.width()
		# print self.init_thumbnail_list_width


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
		# action_doneDrawing = myMenu.addAction("Done drawing")

		selected_action = myMenu.exec_(self.section1_gview.viewport().mapToGlobal(pos))
		if selected_action == action_newPolygon:
			print 'new polygon'

			print 'accepted'
			print self.accepted_proposals.keys()

			for p, props in self.accepted_proposals.iteritems():
				p.setEnabled(False)
				for circ in props['vertexCircles']:
					circ.setEnabled(False)
				props['labelTextArtist'].setEnabled(False)

			self.close_curr_polygon = False
			self.ignore_click = False

			curr_polygon_path = QPainterPath()
			# curr_polygon_path.setFillRule(Qt.WindingFill)

			self.selected_polygon = QGraphicsPathItemModified(curr_polygon_path, gui=self)

			self.selected_polygon.setZValue(50)
			self.selected_polygon.setPen(self.red_pen)
			self.selected_polygon.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
			# self.curr_polygon_closed = False

			self.selected_polygon.signal_emitter.clicked.connect(self.polygon_pressed)
			self.selected_polygon.signal_emitter.moved.connect(self.polygon_moved)
			self.selected_polygon.signal_emitter.released.connect(self.polygon_released)

			self.accepted_proposals[self.selected_polygon] = {'vertexCircles': []}

			self.overlap_with = set([])

			self.section1_gscene.addItem(self.selected_polygon)

			self.set_mode(Mode.ADDING_VERTICES_CONSECUTIVELY)

		elif selected_action == action_deletePolygon:

			print self.selected_polygon

			self.remove_polygon(self.selected_polygon)

			# for circ in self.accepted_proposals[self.selected_polygon]['vertexCircles']:
			# 	self.section1_gscene.removeItem(circ)

			# if 'labelTextArtist' in self.accepted_proposals[self.selected_polygon]:
			# 	self.section1_gscene.removeItem(self.accepted_proposals[self.selected_polygon]['labelTextArtist'])
			# self.section1_gscene.removeItem(self.selected_polygon)
			# self.accepted_proposals.pop(self.selected_polygon)

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

		# elif selected_action == action_doneDrawing:
			# self.set_mode(Mode.IDLE)


	def add_vertex(self, x, y):
		
		ellipse = QGraphicsEllipseItemModified(-VERTEX_CIRCLE_RADIUS, -VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS)
		ellipse.setPos(x,y)

		for polygon in self.accepted_proposals:
			if polygon != self.selected_polygon:
				if polygon.path().contains(QPointF(x,y)) or polygon.path().intersects(self.selected_polygon.path()):
					print 'overlap_with', self.overlap_with
					self.overlap_with.add(polygon)

		ellipse.setPen(Qt.blue)
		ellipse.setBrush(Qt.blue)

		ellipse.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
		ellipse.signal_emitter.moved.connect(self.vertex_moved)
		ellipse.signal_emitter.clicked.connect(self.vertex_clicked)
		ellipse.signal_emitter.released.connect(self.vertex_released)

		self.section1_gscene.addItem(ellipse)

		ellipse.setZValue(99)

		self.accepted_proposals[self.selected_polygon]['vertexCircles'].append(ellipse)

		print self.accepted_proposals[self.selected_polygon]['vertexCircles']

		# self.auto_extend_view(x, y)


	@pyqtSlot()
	def polygon_pressed(self):
		self.polygon_is_moved = False

		print [p.zValue() for p in self.accepted_proposals]

	@pyqtSlot(int, int, int, int)
	def polygon_moved(self, x, y, x0, y0):

		print x, y, x0, y0

		offset_scene_x = x - x0
		offset_scene_y = y - y0

		self.selected_polygon = self.sender().parent

		print self.accepted_proposals[self.selected_polygon]['vertexCircles']

		for i, circ in enumerate(self.accepted_proposals[self.selected_polygon]['vertexCircles']):
			elem = self.selected_polygon.path().elementAt(i)
			scene_pt = self.selected_polygon.mapToScene(elem.x, elem.y)
			circ.setPos(scene_pt)

		self.accepted_proposals[self.selected_polygon]['labelTextArtist'].setPos(self.selected_polygon.label_pos_before_move_x + offset_scene_x, 
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
			self.history.append({'type': 'drag_polygon', 'polygon': self.selected_polygon, 'mouse_moved': (self.selected_polygon.release_scene_x - self.selected_polygon.press_scene_x, \
																										self.selected_polygon.release_scene_y - self.selected_polygon.press_scene_y)})
			self.polygon_is_moved = False

			print 'history:', [h['type'] for h in self.history]

				
	@pyqtSlot(int, int, int, int)
	def vertex_moved(self, x, y, x0, y0):

		offset_scene_x = x - x0
		offset_scene_y = y - y0

		self.selected_vertex_circle = self.sender().parent
		
		self.selected_vertex_center_x_new = self.selected_vertex_circle.center_scene_x_before_move + offset_scene_x
		self.selected_vertex_center_y_new = self.selected_vertex_circle.center_scene_y_before_move + offset_scene_y

		for p, props in self.accepted_proposals.iteritems():
			if self.selected_vertex_circle in props['vertexCircles']:
				self.selected_polygon = p
				break

		vertex_index = self.accepted_proposals[self.selected_polygon]['vertexCircles'].index(self.selected_vertex_circle)
		print 'vertex_index', vertex_index

		curr_polygon_path = self.selected_polygon.path()

		if vertex_index == 0 and self.accepted_proposals[self.selected_polygon]['subtype'] == PolygonType.CLOSED: # closed

			print self.selected_vertex_center_x_new, self.selected_vertex_center_y_new

			curr_polygon_path.setElementPositionAt(0, self.selected_vertex_center_x_new, self.selected_vertex_center_y_new)
			curr_polygon_path.setElementPositionAt(len(self.accepted_proposals[self.selected_polygon]['vertexCircles']), \
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

		for p, props in self.accepted_proposals.iteritems():
			if clicked_vertex in props['vertexCircles']:
				self.selected_polygon = p
				break

		assert clicked_vertex in self.accepted_proposals[self.selected_polygon]['vertexCircles']

		if self.accepted_proposals[self.selected_polygon]['vertexCircles'].index(clicked_vertex) == 0 and \
			len(self.accepted_proposals[self.selected_polygon]['vertexCircles']) > 2 and \
			self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY:
			self.close_curr_polygon = True


		else:
			self.ignore_click = True

	@pyqtSlot()
	def vertex_released(self):
		print self.sender().parent, 'released'

		clicked_vertex = self.sender().parent

		if self.mode == Mode.MOVING_VERTEX and self.vertex_is_moved:
			self.history.append({'type': 'drag_vertex', 'polygon': self.selected_polygon, 'vertex': clicked_vertex, \
								 'mouse_moved': (clicked_vertex.release_scene_x - clicked_vertex.press_scene_x, \
								 	clicked_vertex.release_scene_y - clicked_vertex.press_scene_y)})

			self.vertex_is_moved = False

			print 'history:', [h['type'] for h in self.history]

		elif self.mode == Mode.DELETE_BETWEEN:
			vertex_index = self.accepted_proposals[self.selected_polygon]['vertexCircles'].index(clicked_vertex)
			print 'vertex_index', vertex_index 

			rect = clicked_vertex.rect()
			clicked_vertex.setRect(rect.x()-100, rect.y()-100, 200, 200)

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


	def set_flag_all(self, flag, enabled):

		if hasattr(self, 'accepted_proposals'):

			for p, props in self.accepted_proposals.iteritems():
				p.setFlag(flag, enabled)
				for circ in props['vertexCircles']:
					circ.setFlag(flag, enabled)
				if 'labelTextArtist' in props:
					props['labelTextArtist'].setFlag(flag, enabled)


	def eventFilter(self, obj, event):

		if obj == self.section1_gview.viewport() and event.type() == QEvent.Wheel:
			self.zoom_scene(event)
			return True

		if (obj == self.section2_gview.viewport() or obj == self.section3_gview.viewport()) and event.type() == QEvent.Wheel:
			return True

		if obj == self.section1_gscene and event.type() == QEvent.GraphicsSceneMousePress:

			### with this, single click can select an item; without this only double click can select an item (WEIRD !!!)
			self.section1_gview.translate(0.1, 0.1)
			self.section1_gview.translate(-0.1, -0.1)
			##########################################

			print 'enabled', [p.isEnabled() for p, props in self.accepted_proposals.iteritems()]

			# if self.mode == Mode.MOVING_VERTEX or self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY:
			obj.mousePressEvent(event) # let gscene handle the event (i.e. determine which item or whether an item receives it)

			if self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY:

				x = event.scenePos().x()
				y = event.scenePos().y()

				if hasattr(self, 'close_curr_polygon') and self.close_curr_polygon:

					path = self.selected_polygon.path()
					path.closeSubpath()
					self.selected_polygon.setPath(path)

					# self.history.append({'type': 'close_contour', 'polygon': self.selected_polygon})
					self.history.append({'type': 'add_vertex', 'polygon': self.selected_polygon, 'index': len(self.accepted_proposals[self.selected_polygon]['vertexCircles'])-1})
					print 'history:', [h['type'] for h in self.history]

				elif self.ignore_click:
					self.ignore_click = False

				else:

					self.add_vertex(x, y)

					path = self.selected_polygon.path()

					if len(self.accepted_proposals[self.selected_polygon]['vertexCircles']) == 1:
						path.moveTo(x,y)
					else:
						path.lineTo(x,y)

					self.selected_polygon.setPath(path)

					self.history.append({'type': 'add_vertex', 'polygon': self.selected_polygon, 'index': len(self.accepted_proposals[self.selected_polygon]['vertexCircles'])-1})
					print 'history:', [h['type'] for h in self.history]

			elif self.mode == Mode.IDLE:

				self.press_x = event.pos().x()
				self.press_y = event.pos().y()

				self.press_screen_x = event.screenPos().x()
				self.press_screen_y = event.screenPos().y()

				self.pressed = True

				return True
				
			return False

		if obj == self.section1_gscene and event.type() == QEvent.GraphicsSceneMouseMove:

			# print 'event filter: mouse move'

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

		if obj == self.section1_gscene and event.type() == QEvent.GraphicsSceneMouseRelease:

			obj.mouseReleaseEvent(event)

			if self.mode == Mode.MOVING_VERTEX or self.mode == Mode.ADDING_VERTICES_CONSECUTIVELY:

				if hasattr(self, 'close_curr_polygon') and self.close_curr_polygon:

					self.close_curr_polygon = False
					
					self.accepted_proposals[self.selected_polygon]['subtype'] = PolygonType.CLOSED
					self.complete_polygon()

					self.selected_polygon = None

					self.set_mode(Mode.IDLE)

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

		items = self.section1_gscene.selectedItems()

		vertices_selected = [i for i in items if i not in self.accepted_proposals and isinstance(i, QGraphicsEllipseItemModified)]

		polygons = defaultdict(list)
		for v in vertices_selected:
			for p, props in self.accepted_proposals.iteritems():
				if v in props['vertexCircles']:
					polygons[p].append(props['vertexCircles'].index(v))

		return polygons # {polygon: vertex_indices}


	def subpath(self, path, begin, end):

		new_path = QPainterPath()

		is_closed = self.is_path_closed(path)
		n = path.elementCount() - 1 if is_closed else path.elementCount()

		if not is_closed:
			assert end >= begin
			begin = max(0, begin)
			end = min(n-1, end)
		else:
			if end < begin: end = end + n

		for i in range(begin, end + 1):
			elem = path.elementAt(i % n)
			if new_path.elementCount() == 0:
				new_path.moveTo(elem.x, elem.y)
			else:
				new_path.lineTo(elem.x, elem.y)

		return new_path

	def set_uncertain(self, polygon, vertex_indices):

		uncertain_paths, certain_paths = self.split_path(polygon.path(), vertex_indices)

		for path in uncertain_paths:
			new_uncertain_polygon = self.add_polygon_vertices_label(path, self.green_pen, self.accepted_proposals[polygon]['label'])

		for path in certain_paths:
			new_certain_polygon = self.add_polygon_vertices_label(path, self.red_pen, self.accepted_proposals[polygon]['label'])

		self.remove_polygon(polygon)

		# self.history.append({'type': 'set_uncertain_segment', 'old_polygon': polygon, 'new_certain_polygon': new_certain_polygon, 'new_uncertain_polygon': new_uncertain_polygon})
		# print 'history:', [h['type'] for h in self.history]


	def add_polygon(self, path, pen, z_value=50, uncertain=False):

		polygon = QGraphicsPathItemModified(path, gui=self)
		polygon.setPen(pen)

		polygon.setZValue(z_value)
		polygon.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)

		polygon.signal_emitter.clicked.connect(self.polygon_pressed)
		polygon.signal_emitter.moved.connect(self.polygon_moved)
		polygon.signal_emitter.released.connect(self.polygon_released)

		self.section1_gscene.addItem(polygon)

		self.accepted_proposals[polygon] = {'vertexCircles': [], 'uncertain': uncertain}

		return polygon


	def add_label_to_polygon(self, polygon, label, label_pos=None):

		self.accepted_proposals[polygon]['label'] = label

		textItem = QGraphicsSimpleTextItem(QString(label))

		if label_pos is None:
			centroid = np.mean([(v.scenePos().x(), v.scenePos().y()) for v in self.accepted_proposals[polygon]['vertexCircles']], axis=0)
			textItem.setPos(centroid[0], centroid[1])
		else:
			textItem.setPos(label_pos[0], label_pos[1])

		textItem.setScale(1.5)

		textItem.setFlags(QGraphicsItem.ItemIgnoresTransformations | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)

		textItem.setZValue(99)
		self.accepted_proposals[polygon]['labelTextArtist'] = textItem

		self.section1_gscene.addItem(textItem)


	def add_vertices_to_polygon(self, polygon):

		if polygon not in self.accepted_proposals:
			self.accepted_proposals[polygon] = {'vertexCircles': []}

		path = polygon.path()
		elem_first = path.elementAt(0)
		elem_last = path.elementAt(path.elementCount()-1)
		is_closed = (elem_first.x == elem_last.x) & (elem_first.y == elem_last.y)

		if is_closed:
			self.accepted_proposals[polygon]['subtype'] = PolygonType.CLOSED
		else:
			self.accepted_proposals[polygon]['subtype'] = PolygonType.OPEN

		n = path.elementCount() - 1 if is_closed else path.elementCount()
		print n

		overlap_polygons = set([])

		for i in range(n):

			ellipse = QGraphicsEllipseItemModified(-VERTEX_CIRCLE_RADIUS, -VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS)

 			elem = path.elementAt(i)

			ellipse.setPos(elem.x, elem.y)

			for p in self.accepted_proposals:
				if p != polygon:
					if p.path().contains(QPointF(elem.x, elem.y)) or p.path().intersects(polygon.path()):
						print 'overlap_with', overlap_polygons
						overlap_polygons.add(p)

			ellipse.setPen(Qt.blue)
			ellipse.setBrush(Qt.blue)

			ellipse.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
			ellipse.signal_emitter.moved.connect(self.vertex_moved)
			ellipse.signal_emitter.clicked.connect(self.vertex_clicked)
			ellipse.signal_emitter.released.connect(self.vertex_released)

			self.section1_gscene.addItem(ellipse)

			ellipse.setZValue(99)

			self.accepted_proposals[polygon]['vertexCircles'].append(ellipse)

		return overlap_polygons


	def restack_polygons(self, polygon, overlapping_polygons):

		for p in overlapping_polygons:
			if p.path().contains(polygon.path()): # new polygon within existing polygon, it must has higher z value
				new_z = max(polygon.zValue(), p.zValue()+1)
				print polygon, '=>', new_z
				polygon.setZValue(new_z)

			elif polygon.path().contains(p.path()):  # new polygon wraps existing polygon, it must has lower z value
				new_z = min(polygon.zValue(), p.zValue()-1)
				print polygon, '=>', new_z
				polygon.setZValue(new_z)


	def complete_polygon(self):

		self.open_label_selection_dialog()

		self.restack_polygons(self.selected_polygon, self.overlap_with)

		print 'accepted', self.accepted_proposals.keys()

		for p, props in self.accepted_proposals.iteritems():
			p.setEnabled(True)
			for circ in props['vertexCircles']:
				circ.setEnabled(True)
			props['labelTextArtist'].setEnabled(True)

		self.save_callback()

	def zoom_scene(self, event):

		pos = self.section1_gview.mapToScene(event.pos())

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

	def key_pressed(self, event):

		if event.key() == Qt.Key_Left:
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
			pos = self.section1_gview.mapToScene(self.section1_gview.mapFromGlobal(QCursor.pos()))
			
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
			pos = self.section1_gview.mapToScene(self.section1_gview.mapFromGlobal(QCursor.pos()))

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
				self.selected_polygon = None

		elif event.key() == Qt.Key_Return:
			print 'enter pressed'

			self.accepted_proposals[self.selected_polygon]['subtype'] = PolygonType.OPEN

			self.complete_polygon()

			self.set_mode(Mode.IDLE)
			self.selected_polygon = None

		elif event.key() == Qt.Key_C:
			path = self.selected_polygon.path()
			path.closeSubpath()
			self.selected_polygon.setPath(path)

			self.history.append({'type': 'add_vertex', 'polygon': self.selected_polygon, 'index': len(self.accepted_proposals[self.selected_polygon]['vertexCircles'])-1})
			print 'history:', [h['type'] for h in self.history]

			self.close_curr_polygon = False
					
			self.accepted_proposals[self.selected_polygon]['subtype'] = PolygonType.CLOSED
			self.complete_polygon()

			self.selected_polygon = None

			self.set_mode(Mode.IDLE)

		elif event.key() == Qt.Key_Backspace:
			self.undo()

		elif event.key() == Qt.Key_3:

			# if not hasattr(self, 'section2'):
			# 	self.section2 = self.section - 1
			# else:
			if self.section2 == self.section - NUM_NEIGHBORS_PRELOAD:
				return
			else:
				self.section2 = self.section2 - 1
			
			self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', middle section %d' %self.section + ', left %d'%self.section3 + ', right %d'%self.section2)

			self.section2_pixmap = self.pixmaps[self.section2]

			self.section2_gscene = QGraphicsScene(self.section2_gview)
			self.section2_gscene.addPixmap(self.section2_pixmap)

			self.section2_gview.setScene(self.section2_gscene)
			# self.section2_gscene.update(0, 0, self.section2_gview.width(), self.section2_gview.height())

		elif event.key() == Qt.Key_4:
			# if not hasattr(self, 'section2'):
			# 	self.section2 = self.section + 1
			# else:
			if self.section2 == self.section + NUM_NEIGHBORS_PRELOAD:
				return
			else:
				self.section2 = self.section2 + 1

			self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', middle section %d' %self.section + ', left %d'%self.section3 + ', right %d'%self.section2)

			self.section2_pixmap = self.pixmaps[self.section2]

			self.section2_gscene = QGraphicsScene(self.section2_gview)
			self.section2_gscene.addPixmap(self.section2_pixmap)

			self.section2_gview.setScene(self.section2_gscene)
			# self.section2_gscene.update(0, 0, self.section2_gview.width(), self.section2_gview.height())

		elif event.key() == Qt.Key_1:

			# if not hasattr(self, 'section2'):
			# 	self.section2 = self.section - 1
			# else:
			if self.section3 == self.section - NUM_NEIGHBORS_PRELOAD:
				return
			else:
				self.section3 = self.section3 - 1
			
			self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', middle section %d' %self.section + ', left %d'%self.section3 + ', right %d'%self.section2)

			self.section3_pixmap = self.pixmaps[self.section3]

			self.section3_gscene = QGraphicsScene(self.section3_gview)
			self.section3_gscene.addPixmap(self.section3_pixmap)

			self.section3_gview.setScene(self.section3_gscene)
			# self.section3_gscene.update(0, 0, self.section2_gview.width(), self.section2_gview.height())
			
			# self.section3_gview.show()

		elif event.key() == Qt.Key_2:
			# if not hasattr(self, 'section2'):
			# 	self.section2 = self.section + 1
			# else:
			if self.section3 == self.section + NUM_NEIGHBORS_PRELOAD:
				return
			else:
				self.section3 = self.section3 + 1

			self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', middle section %d' %self.section + ', left %d'%self.section3 + ', right %d'%self.section2)

			self.section3_pixmap = self.pixmaps[self.section3]

			self.section3_gscene = QGraphicsScene(self.section3_gview)
			self.section3_gscene.addPixmap(self.section3_pixmap)

			self.section3_gview.setScene(self.section3_gscene)

		# elif event.key() == Qt.Key_U:
		# 	# must be in selecting_roi mode
		# 	if self.mode != Mode.SELECTING_ROI:
		# 		return


	##########################

	def thumbnail_list_resized(self, event):
		new_size = 200 * event.size().width() / self.init_thumbnail_list_width
		self.thumbnail_list.setIconSize( QSize(new_size , new_size ) )

	def toggle_labels(self):

		self.labels_on = not self.labels_on

		if not self.labels_on:

			for polygon, props in self.accepted_proposals.iteritems():
				props['labelTextArtist'].setVisible(False)

			self.button_labelsOnOff.setText('Turns Labels ON')

		else:
			for polygon, props in self.accepted_proposals.iteritems():
				props['labelTextArtist'].setVisible(True)

			self.button_labelsOnOff.setText('Turns Labels OFF')

	def toggle_contours(self):

		self.contours_on = not self.contours_on

		if not self.contours_on:

			for polygon, props in self.accepted_proposals.iteritems():
				polygon.setVisible(False)
				if self.vertices_on:
					for circ in props['vertexCircles']:
						circ.setVisible(False)

			self.button_contoursOnOff.setText('Turns Contours ON')

		else:
			for polygon, props in self.accepted_proposals.iteritems():
				polygon.setVisible(True)
				if self.vertices_on:
					for circ in props['vertexCircles']:
						circ.setVisible(True)

			self.button_contoursOnOff.setText('Turns Contours OFF')


	def toggle_vertices(self):

		self.vertices_on = not self.vertices_on

		if not self.vertices_on:

			for polygon, props in self.accepted_proposals.iteritems():
				for circ in props['vertexCircles']:
					circ.setVisible(False)

			self.button_verticesOnOff.setText('Turns Vertices ON')

		else:
			for polygon, props in self.accepted_proposals.iteritems():
				for circ in props['vertexCircles']:
					circ.setVisible(True)

			self.button_verticesOnOff.setText('Turns Vertices OFF')

	def updateDB_callback(self):
		cmd = 'rsync -az --include="*/" %(local_labeling_dir)s/%(stack)s yuncong@gcn-20-33.sdsc.edu:%(gordon_labeling_dir)s' % {'gordon_labeling_dir':os.environ['GORDON_LABELING_DIR'],
																			'local_labeling_dir':os.environ['LOCAL_LABELING_DIR'],
																			'stack': self.stack
																			}
		os.system(cmd)

		# cmd = 'rsync -az %(local_labeling_dir)s/labelnames.txt yuncong@gcn-20-33.sdsc.edu:%(gordon_labeling_dir)s' % {'gordon_labeling_dir':os.environ['GORDON_LABELING_DIR'],
		#                                                             'local_labeling_dir':os.environ['LOCAL_LABELING_DIR'],
		#                                                             }
		# os.system(cmd)
		self.statusBar().showMessage('labelings synced')

		# payload = {'section': self.dm.slice_ind}
		# r = requests.get('http://gcn-20-32.sdsc.edu:5000/update_db', params=payload)
		r = requests.get('http://gcn-20-32.sdsc.edu:5000/update_db')
		res = r.json()
		if res['result'] == 0:
			self.statusBar().showMessage('Landmark database updated')


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


	def load_local_proposals(self):

		sys.stderr.write('loading local proposals ...\n')
		self.statusBar().showMessage('loading local proposals ...')
		
		cluster_tuples = self.dm.load_pipeline_result('allSeedClusterScoreDedgeTuples')
		self.local_proposal_tuples = [(cl, ed, sig) for seed, cl, sig, ed in cluster_tuples]
		self.local_proposal_clusters = [m[0] for m in self.local_proposal_tuples]
		self.local_proposal_dedges = [m[1] for m in self.local_proposal_tuples]
		self.local_proposal_sigs = [m[2] for m in self.local_proposal_tuples]

		self.n_local_proposals = len(self.local_proposal_tuples)
		
		if not hasattr(self, 'local_proposal_pathPatches'):
			self.local_proposal_pathPatches = [None] * self.n_local_proposals
			self.local_proposal_vertexCircles = [None] * self.n_local_proposals

		self.local_proposal_indices_from_sp = defaultdict(list)
		for i, (seed, _, _, _) in enumerate(cluster_tuples):
			self.local_proposal_indices_from_sp[seed].append(i)
		self.local_proposal_indices_from_sp.default_factory = None

		sys.stderr.write('%d local proposals loaded.\n' % self.n_local_proposals)
		self.statusBar().showMessage('Local proposals loaded.')

		self.local_proposal_labels = [None] * self.n_local_proposals


	def load_global_proposals(self):
		
		self.global_proposal_tuples =  self.dm.load_pipeline_result('proposals')
		self.global_proposal_clusters = [m[0] for m in self.global_proposal_tuples]
		self.global_proposal_dedges = [m[1] for m in self.global_proposal_tuples]
		self.global_proposal_sigs = [m[2] for m in self.global_proposal_tuples]

		self.n_global_proposals = len(self.global_proposal_tuples)

		if not hasattr(self, 'global_proposal_pathPatches'):
			self.global_proposal_pathPatches = [None] * self.n_global_proposals
			self.global_proposal_vertexCircles = [None] * self.n_global_proposals

		self.statusBar().showMessage('%d global proposals loaded' % self.n_global_proposals)

		self.sp_covered_by_proposals = self.dm.load_pipeline_result('spCoveredByProposals')
		self.sp_covered_by_proposals = dict([(s, list(props)) for s, props in self.sp_covered_by_proposals.iteritems()])

		self.global_proposal_labels = [None] * self.n_global_proposals

	def load_callback(self):

		fname = str(QFileDialog.getOpenFileName(self, 'Open file', self.dm.labelings_dir))
		stack, sec, username, timestamp, suffix = os.path.basename(fname[:-4]).split('_')

		# self.accepted_proposals = {}

		_, _, _, accepted_proposal_props = self.dm.load_proposal_review_result(username, timestamp, suffix)

		for props in accepted_proposal_props:
			
			curr_polygon_path = QPainterPath()

			for i, (x, y) in enumerate(props['vertices']):
				if i == 0:
					curr_polygon_path.moveTo(x,y)
				else:
					curr_polygon_path.lineTo(x,y)

			if props['subtype'] == PolygonType.CLOSED:
				curr_polygon_path.closeSubpath()

			polygon = QGraphicsPathItemModified(curr_polygon_path, gui=self)
			polygon.setPen(self.red_pen)
			polygon.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)

			polygon.signal_emitter.clicked.connect(self.polygon_pressed)
			polygon.signal_emitter.moved.connect(self.polygon_moved)
			polygon.signal_emitter.released.connect(self.polygon_released)

			polygon.setZValue(50)

			polygon.setPath(curr_polygon_path)

			self.section1_gscene.addItem(polygon)

			# elif props['subtype'] == PolygonType.OPEN:
			# else:
			# 	raise 'unknown polygon type'

			props['vertexCircles'] = []
			for x, y in props['vertices']:

				ellipse = QGraphicsEllipseItemModified(-VERTEX_CIRCLE_RADIUS, -VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS, 2*VERTEX_CIRCLE_RADIUS)
				ellipse.setPos(x,y)
				
				ellipse.setPen(Qt.blue)
				ellipse.setBrush(Qt.blue)

				ellipse.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)
				ellipse.signal_emitter.moved.connect(self.vertex_moved)
				ellipse.signal_emitter.clicked.connect(self.vertex_clicked)
				ellipse.signal_emitter.released.connect(self.vertex_released)

				self.section1_gscene.addItem(ellipse)

				ellipse.setZValue(99)

				props['vertexCircles'].append(ellipse)

			textItem = QGraphicsSimpleTextItem(QString(props['label']))

			if 'labelPos' not in props:
				centroid = np.mean([(v.scenePos().x(), v.scenePos().y()) for v in props['vertexCircles']], axis=0)
				textItem.setPos(centroid[0], centroid[1])
			else:
				textItem.setPos(props['labelPos'][0], props['labelPos'][1])

			textItem.setScale(1.5)

			textItem.setFlags(QGraphicsItem.ItemIgnoresTransformations | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)

			textItem.setZValue(99)
			props['labelTextArtist'] = textItem

			self.section1_gscene.addItem(textItem)

			props.pop('vertices')

			self.accepted_proposals[polygon] = props


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

		if 'label' in self.accepted_proposals[self.selected_polygon]:
			self.label_selection_dialog.comboBox.setEditText(self.accepted_proposals[self.selected_polygon]['label']+' ('+self.structure_names[self.accepted_proposals[self.selected_polygon]['label']]+')')
		else:
			self.accepted_proposals[self.selected_polygon]['label'] = ''

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

		self.accepted_proposals[self.selected_polygon]['label'] = abbr

		if 'labelTextArtist' in self.accepted_proposals[self.selected_polygon] and self.accepted_proposals[self.selected_polygon]['labelTextArtist'] is not None:
			self.accepted_proposals[self.selected_polygon]['labelTextArtist'].setText(abbr)
		else:
			textItem = QGraphicsSimpleTextItem(QString(abbr))
			self.section1_gscene.addItem(textItem)

			print self.selected_polygon
			centroid = np.mean([(v.scenePos().x(), v.scenePos().y()) for v in self.accepted_proposals[self.selected_polygon]['vertexCircles']], axis=0)
			print centroid
			textItem.setPos(centroid[0], centroid[1])
			textItem.setScale(1.5)

			textItem.setFlags(QGraphicsItem.ItemIgnoresTransformations | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemClipsToShape | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemSendsScenePositionChanges)

			self.accepted_proposals[self.selected_polygon]['labelTextArtist'] = textItem

			textItem.setZValue(99)

			# vertices = self.vertices_from_vertexPatches(self.selected_proposal_polygon)
			# centroid = np.mean(vertices, axis=0)
			# text_artist = Text(centroid[0], centroid[1], abbr, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
			# self.accepted_proposals[self.selected_proposal_polygon]['labelTextArtist'] = text_artist
			# self.axis.add_artist(text_artist)
			# text_artist.set_picker(True)

		self.recent_labels.insert(0, abbr)
		# self.invalid_labelname = None

		self.label_selection_dialog.accept()

	def shuffle_proposal_from_pool(self, sp_ind):

		if self.shuffle_global_proposals:   
			if not hasattr(self, 'sp_covered_by_proposals'):
				return
		else:
			if not hasattr(self, 'local_proposal_indices_from_sp'):
				return

		if self.shuffle_global_proposals:

			if sp_ind not in self.sp_covered_by_proposals or sp_ind == -1:
				self.statusBar().showMessage('No proposal covers superpixel %d' % sp_ind)
				return 
		else:
			if sp_ind == -1:
				return
		
		if self.object_picked:
			return

		self.cancel_current_selection()

		if self.shuffle_global_proposals:
			self.selected_proposal_type = ProposalType.GLOBAL

			self.alternative_global_proposal_ind = (self.alternative_global_proposal_ind + 1) % len(self.sp_covered_by_proposals[sp_ind])
			self.selected_proposal_id = self.sp_covered_by_proposals[sp_ind][self.alternative_global_proposal_ind]

			dedges = self.global_proposal_dedges[self.selected_proposal_id]
		else:

			self.selected_proposal_type = ProposalType.LOCAL

			self.alternative_local_proposal_ind = (self.alternative_local_proposal_ind + 1) % len(self.local_proposal_indices_from_sp[sp_ind])
			self.selected_proposal_id = self.local_proposal_indices_from_sp[sp_ind][self.alternative_local_proposal_ind]

			cl, dedges, sig = self.local_proposal_tuples[self.selected_proposal_id]


		if self.shuffle_global_proposals:
			proposal_pathPatches = self.global_proposal_pathPatches
			proposal_vertexCircles = self.global_proposal_vertexCircles
		else:
			proposal_pathPatches = self.local_proposal_pathPatches
			proposal_vertexCircles = self.local_proposal_vertexCircles

		if proposal_pathPatches[self.selected_proposal_id] is None:  
			vertices = self.dm.vertices_from_dedges(dedges)

			proposal_pathPatches[self.selected_proposal_id] = Polygon(vertices, closed=True, 
									edgecolor=self.boundary_colors[0], fill=False, linewidth=UNSELECTED_POLYGON_LINEWIDTH)
			proposal_vertexCircles[self.selected_proposal_id] = [plt.Circle(v, radius=UNSELECTED_CIRCLE_SIZE, color=self.boundary_colors[0], alpha=.8) for v in vertices]

		if self.shuffle_global_proposals:
			self.selected_proposal_polygon = self.global_proposal_pathPatches[self.selected_proposal_id]
			self.selected_proposal_vertexCircles = self.global_proposal_vertexCircles[self.selected_proposal_id]
		else:
			self.selected_proposal_polygon = self.local_proposal_pathPatches[self.selected_proposal_id]
			self.selected_proposal_vertexCircles = self.local_proposal_vertexCircles[self.selected_proposal_id]            

		if self.selected_proposal_polygon not in self.axis.patches:
			self.axis.add_patch(self.selected_proposal_polygon)

			for vertex_circ in self.selected_proposal_vertexCircles:
				self.axis.add_patch(vertex_circ)

		self.selected_proposal_polygon.set_picker(None)
		for vertex_circ in self.selected_proposal_vertexCircles:
			vertex_circ.set_picker(None)

		if self.selected_proposal_polygon in self.accepted_proposals:
			self.selected_proposal_polygon.set_linewidth(UNSELECTED_POLYGON_LINEWIDTH)
			label =  self.accepted_proposals[self.selected_proposal_polygon]['label']
		else:
			label = ''

		if self.shuffle_global_proposals:
			self.statusBar().showMessage('global proposal (%s) covering seed %d, score %.4f' % (label, sp_ind, self.global_proposal_sigs[self.selected_proposal_id]))
		else:
			self.statusBar().showMessage('local proposal (%s) from seed %d, score %.4f' % (label, sp_ind, sig))

		self.canvas.draw()


	def section_changed(self, item):

		self.statusBar().showMessage('Loading ....')

		# if hasattr(self, 'global_proposal_tuples'):
		# 	del self.global_proposal_tuples
		# if hasattr(self, 'global_proposal_pathPatches'):
		# 	for p in self.global_proposal_pathPatches:
		# 		if p in self.axis.patches:
		# 			p.remove()
		# 	del self.global_proposal_pathPatches
		# if hasattr(self, 'local_proposal_tuples'):
		# 	del self.local_proposal_tuples
		# if hasattr(self, 'local_proposal_pathPatches'):
		# 	for p in self.local_proposal_pathPatches:
		# 		if p in self.axis.patches:
		# 			p.remove()
		# 	del self.local_proposal_pathPatches

		sec = int(str(item.text()))

		if hasattr(self, 'pixmaps'):
			# del self.pixmaps
			del self.dms
			del self.section1_pixmap
			del self.section2_pixmap
			del self.section3_pixmap
			del self.section1_gscene
			del self.section2_gscene
			del self.section3_gscene

		self.init_data(section=sec)
		self.reload_brain_labeling_gui()

		# self.mode_changed()
		# self.turn_superpixels_on()

		# self.pixmap = QPixmap("/home/yuncong/CSHL_data_processed/%(stack)s_thumbnail_aligned_cropped/%(stack)s_%(sec)04d_thumbnail_aligned_cropped.tif"%{'sec':sec, 'stack':self.stack})
		# self.pixmap_scaled = self.pixmap.scaledToHeight(self.bottom_panel.sizeHint().height())

		# self.graphicsScene_navMap = QGraphicsScene(self.graphicsView_navMap)
		# self.graphicsScene_navMap.addPixmap(self.pixmap_scaled)

		# self.navRect = self.graphicsScene_navMap.addRect(10,10,200,200, QPen(QColor(255,0,0), 1))

		# self.graphicsView_navMap.setScene(self.graphicsScene_navMap)
		# self.graphicsView_navMap.show()

		# self.navMap_scaling_x = self.pixmap_scaled.size().width()/float(self.dm.image_width)
		# print self.navMap_scaling_x
		# self.navMap_scaling_y = self.pixmap_scaled.size().height()/float(self.dm.image_height)

		# self.update_navMap()

		# self.graphicsScene_navMap.mousePressEvent = self.clicked_navMap

		# self.statusBar().showMessage('Loading complete.')

	def clicked_navMap(self, event):
		# print event.scenePos()

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

	def save_callback(self):

		timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

		if not hasattr(self, 'username') or self.username is None:
			username, okay = QInputDialog.getText(self, "Username", "Please enter your username:", QLineEdit.Normal, 'anon')
			if not okay: return
			self.username = str(username)

		accepted_proposal_props = []
		for polygon, props in self.accepted_proposals.iteritems():

			props_saved = props.copy()

			props_saved['vertices'] = [(v.scenePos().x(), v.scenePos().y()) for v in props['vertexCircles']]

			label_pos = props['labelTextArtist'].scenePos()
			props_saved['labelPos'] = (label_pos.x(), label_pos.y())

			props_saved.pop('vertexCircles')
			props_saved.pop('labelTextArtist')

			accepted_proposal_props.append(props_saved)

		self.dm.save_proposal_review_result(accepted_proposal_props, self.username, timestamp, suffix='consolidated')

		print self.new_labelnames
		self.dm.add_labelnames(self.new_labelnames, self.dm.repo_dir+'/visualization/newStructureNames.txt')

		self.statusBar().showMessage('Labelings saved to %s' % (self.username+'_'+timestamp))

	def labelbutton_callback(self):
		pass

	############################################
	# matplotlib canvas CALLBACKs
	############################################

	def remove_polygon(self, polygon):
		for circ in self.accepted_proposals[polygon]['vertexCircles']:
			self.section1_gscene.removeItem(circ)

		if 'labelTextArtist' in self.accepted_proposals[polygon]:
			self.section1_gscene.removeItem(self.accepted_proposals[polygon]['labelTextArtist'])

		self.section1_gscene.removeItem(polygon)

		self.accepted_proposals.pop(polygon)

	def undo(self):

		if len(self.history) == 0:
			return

		history_item = self.history.pop()

		if history_item['type'] == 'drag_polygon':

			polygon = history_item['polygon']
			moved_x, moved_y = history_item['mouse_moved']

			for circ in self.accepted_proposals[polygon]['vertexCircles']:
				curr_pos = circ.scenePos()
				circ.setPos(curr_pos.x() - moved_x, curr_pos.y() - moved_y)

			path = polygon.path()
			for i in range(polygon.path().elementCount()):
				elem = polygon.path().elementAt(i)
				scene_pos = polygon.mapToScene(elem.x, elem.y)
				path.setElementPositionAt(i, scene_pos.x() - moved_x, scene_pos.y() - moved_y)

			polygon.setPath(path)

			if 'labelTextArtist' in self.accepted_proposals[polygon]:
				curr_label_pos = self.accepted_proposals[polygon]['labelTextArtist'].scenePos()
				self.accepted_proposals[polygon]['labelTextArtist'].setPos(curr_label_pos.x() - moved_x, curr_label_pos.y() - moved_y)

			self.section1_gscene.update(0, 0, self.section1_gview.width(), self.section1_gview.height())

		elif history_item['type'] == 'drag_vertex':

			polygon = history_item['polygon']
			vertex = history_item['vertex']
			moved_x, moved_y = history_item['mouse_moved']

			curr_pos = vertex.scenePos()
			vertex.setPos(curr_pos.x() - moved_x, curr_pos.y() - moved_y)

			vertex_index = self.accepted_proposals[polygon]['vertexCircles'].index(vertex)

			path = polygon.path()
			elem_first = path.elementAt(0)
			elem_last = path.elementAt(path.elementCount()-1)
			is_closed = (elem_first.x == elem_last.x) & (elem_first.y == elem_last.y)

			if vertex_index == 0 and is_closed:
				path.setElementPositionAt(0, curr_pos.x() - moved_x, curr_pos.y() - moved_y)
				path.setElementPositionAt(len(self.accepted_proposals[polygon]['vertexCircles']), curr_pos.x() - moved_x, curr_pos.y() - moved_y)
			else:
				path.setElementPositionAt(vertex_index, curr_pos.x() - moved_x, curr_pos.y() - moved_y)

			polygon.setPath(path)

			self.section1_gscene.update(0, 0, self.section1_gview.width(), self.section1_gview.height())

		elif history_item['type'] == 'add_vertex':
			polygon = history_item['polygon']
			index = history_item['index']

			vertex = self.accepted_proposals[polygon]['vertexCircles'][index]
			# vertex = history_item['vertex']

			path = polygon.path()
			elem_first = path.elementAt(0)
			elem_last = path.elementAt(path.elementCount()-1)
			is_closed = (elem_first.x == elem_last.x) & (elem_first.y == elem_last.y)
			print 'is_closed', is_closed

			path = QPainterPath()

			n = len(self.accepted_proposals[polygon]['vertexCircles'])

			if n == 1:
				# if only one vertex in polygon, then undo removes the entire polygon
				self.section1_gscene.removeItem(polygon)
				if 'labelTextArtist' in self.accepted_proposals[polygon]:
					self.section1_gscene.removeItem(self.accepted_proposals[polygon]['labelTextArtist'])
				self.accepted_proposals.pop(polygon)
				self.section1_gscene.removeItem(vertex)

				self.set_mode(Mode.IDLE)
			else:

				if not is_closed:
					# if it is open, then undo removes the last vertex
					self.accepted_proposals[polygon]['vertexCircles'].remove(vertex)
					self.section1_gscene.removeItem(vertex)

					for i in range(n-1):
						elem = polygon.path().elementAt(i)
						if i == 0:
							path.moveTo(elem.x, elem.y)
						else:
							path.lineTo(elem.x, elem.y)
				else:
					# if it is closed, then undo opens it, without removing any vertex
					for i in range(n):
						elem = polygon.path().elementAt(i)
						if i == 0:
							path.moveTo(elem.x, elem.y)
						else:
							path.lineTo(elem.x, elem.y)

				polygon.setPath(path)

		elif history_item['type'] == 'set_uncertain_segment':

			old_polygon = history_item['old_polygon']
			new_certain_polygon = history_item['new_certain_polygon']
			new_uncertain_polygon = history_item['new_uncertain_polygon']

			label = self.accepted_proposals[new_certain_polygon]['label']

			self.remove_polygon(new_certain_polygon)
			self.remove_polygon(new_uncertain_polygon)

			self.section1_gscene.addItem(old_polygon)
			overlap_polygons = self.add_vertices_to_polygon(old_polygon)
			self.restack_polygons(old_polygon, overlap_polygons)
			self.add_label_to_polygon(old_polygon, label=label)


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
			self.section1_gview.setDragMode(QGraphicsView.RubberBandDrag)
		else:
			self.section1_gview.setDragMode(QGraphicsView.NoDrag)

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


	def find_vertex_insert_position(self, xys, pos, closed=True):

		n = len(xys)
		if n == 1:
			return 1

		xys_homo = np.column_stack([xys, np.ones(n,)])

		if closed:
			edges = np.array([np.cross(xys_homo[i], xys_homo[(i+1)%n]) for i in range(n)])
		else:
			edges = np.array([np.cross(xys_homo[i], xys_homo[i+1]) for i in range(n-1)])

		edges_normalized = edges/np.sqrt(np.sum(edges[:,:2]**2, axis=1))[:, np.newaxis]

		signed_dists = np.dot(edges_normalized, np.r_[pos,1])
		dists = np.abs(signed_dists)
		# sides = np.sign(signed_dists)

		projections = pos - signed_dists[:, np.newaxis] * edges_normalized[:,:2]

		endpoint = [None for _ in projections]
		for i, (px, py) in enumerate(projections):
			if (px > xys[i][0] and px > xys[(i+1)%n][0]) or (px < xys[i][0] and px < xys[(i+1)%n][0]):
				endpoint[i] = [i, (i+1)%n][np.argmin(np.squeeze(cdist([pos], [xys[i], xys[(i+1)%n]])))]
				dists[i] = np.min(np.squeeze(cdist([pos], [xys[i], xys[(i+1)%n]])))

		# print edges_normalized[:,:2]
		# print projections                
		# print dists
		# print endpoint
		nearest_edge_begins_at = np.argsort(dists)[0]

		if nearest_edge_begins_at == 0 and not closed and endpoint[0] == 0:
			new_vertex_ind = 0
		elif nearest_edge_begins_at == n-2 and not closed and endpoint[-1] == n-1:
			new_vertex_ind = n
		else:
			new_vertex_ind = nearest_edge_begins_at + 1  

		print 'nearest_edge_begins_at', nearest_edge_begins_at, 'new_vertex_ind', new_vertex_ind

		return new_vertex_ind



	# def connect_two_vertices(self, polygon1, polygon2=None, index1=None, index2=None):

	# 	vertices1 = self.vertices_from_vertexPatches(polygon1)
	# 	n1 = len(vertices1)

	# 	if polygon2 is None: # connect two ends of a single polygon   
			
	# 		print 'index1', index1, 'index2', index2
	# 		assert not polygon1.get_closed()
	# 		 # and ((index1 == 0 and index2 == n2-1) or (index1 == n1-1 and index2 == 0))
	# 		polygon1.set_closed(True)
	# 		if 'label' not in self.accepted_proposals[polygon1]:
	# 			self.acceptProposal_callback()
		
	# 	else:

	# 		vertices2 = self.vertices_from_vertexPatches(polygon2)
	# 		n2 = len(vertices2)
	# 		print 'index1', index1, 'index2', index2
	# 		assert not polygon1.get_closed() and index1 in [0, n1-1] and index2 in [0, n2-1]

	# 		if index1 == 0 and index2 == 0:
	# 			new_vertices = np.vstack([vertices1[::-1], vertices2])
	# 		elif index1 != 0 and index2 != 0:
	# 			new_vertices = np.vstack([vertices1, vertices2[::-1]])
	# 		elif index1 != 0 and index2 == 0:
	# 			new_vertices = np.vstack([vertices1, vertices2])
	# 		elif index1 == 0 and index2 != 0:
	# 			new_vertices = np.vstack([vertices1[::-1], vertices2[::-1]])

	# 		props = {}

	# 		patch = Polygon(new_vertices, closed=False, edgecolor=self.boundary_colors[1], fill=False, linewidth=UNSELECTED_POLYGON_LINEWIDTH)
	# 		patch.set_picker(True)

	# 		self.axis.add_patch(patch)

	# 		props['vertexPatches'] = []
	# 		for x,y in new_vertices:
	# 			vertex_circle = plt.Circle((x, y), radius=UNSELECTED_CIRCLE_SIZE, color=self.boundary_colors[1], alpha=.8)
	# 			vertex_circle.set_picker(CIRCLE_PICK_THRESH)
	# 			props['vertexPatches'].append(vertex_circle)
	# 			self.axis.add_patch(vertex_circle)
	# 			vertex_circle.set_picker(True)

	# 		if self.accepted_proposals[polygon1]['label'] == self.accepted_proposals[polygon2]['label']:
	# 			props['label'] = self.accepted_proposals[polygon1]['label']
	# 		else:
	# 			props['label'] = self.accepted_proposals[polygon1]['label']
	# 		# else:
	# 			# self.acceptProposal_callback()

	# 		props['type'] = self.accepted_proposals[polygon1]['type']

	# 		centroid = np.mean(new_vertices, axis=0)
	# 		props['labelTextArtist'] = Text(centroid[0], centroid[1], props['label'], style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

	# 		self.axis.add_artist(props['labelTextArtist'])
	# 		props['labelTextArtist'].set_picker(True)

	# 		self.accepted_proposals[patch] = props
			
	# 		for circ in self.accepted_proposals[polygon1]['vertexPatches']:
	# 			circ.remove()
			
	# 		for circ in self.accepted_proposals[polygon2]['vertexPatches']:
	# 			circ.remove()

	# 		self.accepted_proposals[polygon1]['labelTextArtist'].remove()
	# 		self.accepted_proposals[polygon2]['labelTextArtist'].remove()
			
	# 		polygon1.remove()
	# 		polygon2.remove()

	# 		self.accepted_proposals.pop(polygon1)
	# 		self.accepted_proposals.pop(polygon2)

	# 	self.cancel_current_selection()

	# 	self.canvas.draw()

	def split_path(self, path, vertex_indices):

		is_closed = self.is_path_closed(path)
		n = path.elementCount() - 1 if is_closed else path.elementCount()
		segs_in, segs_out = self.split_array(vertex_indices, n, is_closed)

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


	def is_path_closed(self, path):

		elem_first = path.elementAt(0)
		elem_last = path.elementAt(path.elementCount()-1)
		is_closed = (elem_first.x == elem_last.x) & (elem_first.y == elem_last.y)

		return is_closed

	def delete_vertices(self, polygon, indices_to_remove, merge=False):

		if merge:
			self.delete_vertices_merge(polygon, indices_to_remove)
		else:
			paths_to_remove, paths_to_keep = self.split_path(polygon.path(), indices_to_remove)

			for path in paths_to_keep:
				self.add_polygon_vertices_label(path, pen=self.red_pen, label=self.accepted_proposals[polygon]['label'])

			self.remove_polygon(polygon)


	def delete_between(self, polygon, first_index, second_index):

		print first_index, second_index

		if second_index < first_index:	# ensure first_index is smaller than second_index
			temp = first_index
			first_index = second_index
			second_index = temp

		path = polygon.path()
		is_closed = self.is_path_closed(path) 
		n = path.elementCount() - 1 if is_closed else path.elementCount()

		if (second_index - first_index > first_index + n - second_index):
			indices_to_remove = range(second_index, n+1) + range(0, first_index+1)
		else:
			indices_to_remove = range(first_index, second_index+1)

		print indices_to_remove

		paths_to_remove, paths_to_keep = self.split_path(path, indices_to_remove)

		for new_path in paths_to_keep:

			self.add_polygon_vertices_label(new_path, pen=self.red_pen, label=self.accepted_proposals[polygon]['label'])	

		self.remove_polygon(polygon)


	def delete_vertices_merge(self, polygon, indices_to_remove):

		path = polygon.path()
		is_closed = self.is_path_closed(path) 
		n = path.elementCount() - 1 if is_closed else path.elementCount()

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
				
		self.add_polygon_vertices_label(new_path, pen=self.red_pen, label=self.accepted_proposals[polygon]['label'])
		
		self.remove_polygon(polygon)

			
	def add_polygon_vertices_label(self, path, pen, label):
		new_polygon = self.add_polygon(path, pen)
		overlap_polygons = self.add_vertices_to_polygon(new_polygon)
		print overlap_polygons
		self.restack_polygons(new_polygon, overlap_polygons)
		self.add_label_to_polygon(new_polygon, label=label)

		return new_polygon

	def auto_extend_view(self, x, y):
		# always make just placed vertex at the center of the view

		viewport_scene_rect = self.section1_gview.viewport().rect()	# NOT UPDATING!!! WEIRD!!!
		cur_xmin = viewport_scene_rect.x()
		cur_ymin = viewport_scene_rect.y()
		cur_xmax = cur_xmin + viewport_scene_rect.width()
		cur_ymax = cur_ymin + viewport_scene_rect.height()

		print cur_xmin, cur_ymin, cur_xmax, cur_ymax

		if abs(x - cur_xmin) < AUTO_EXTEND_VIEW_TOLERANCE or abs(x - cur_xmax) < AUTO_EXTEND_VIEW_TOLERANCE:
			cur_xcenter = cur_xmin * .6 + cur_xmax * .4 if abs(x - cur_xmin) < AUTO_EXTEND_VIEW_TOLERANCE else cur_xmin * .4 + cur_xmax * .6
			translation_x = cur_xcenter - x

			self.section1_gview.translate(-translation_x, 0)
			self.section2_gview.translate(-translation_x, 0)
			self.section3_gview.translate(-translation_x, 0)

		if abs(y - cur_ymin) < AUTO_EXTEND_VIEW_TOLERANCE or abs(y - cur_ymax) < AUTO_EXTEND_VIEW_TOLERANCE:
			cur_ycenter = cur_ymin * .6 + cur_ymax * .4 if abs(y - cur_ymin) < AUTO_EXTEND_VIEW_TOLERANCE else cur_ymin * .4 + cur_ymax * .6
			translation_y = cur_ycenter - y

			self.section1_gview.translate(0, -translation_y)
			self.section2_gview.translate(0, -translation_y)
			self.section3_gview.translate(0, -translation_y)


	# def place_vertex(self, x, y):
	#     self.selected_proposal_vertices.append([x, y])

	#     # curr_vertex_circle = plt.Circle((x, y), radius=10, color=self.colors[self.curr_label + 1], alpha=.8)
	#     curr_vertex_circle = plt.Circle((x, y), radius=UNSELECTED_CIRCLE_SIZE, color=self.boundary_colors[1], alpha=.8)
	#     self.axis.add_patch(curr_vertex_circle)
	#     self.selected_proposal_vertexCircles.append(curr_vertex_circle)

	#     curr_vertex_circle.set_picker(CIRCLE_PICK_THRESH)

	#     self.auto_extend_view(x, y)

	#     self.canvas.draw()
	#     self.canvas2.draw()

	#     self.history.append({'type': 'add_vertex', 'selected_proposal_vertexCircles': self.selected_proposal_vertexCircles,
	#         'selected_proposal_vertices': self.selected_proposal_vertices})

		# print self.history


	def load_segmentation(self):
		sys.stderr.write('loading segmentation...\n')
		self.statusBar().showMessage('loading segmentation...')

		self.dm.load_multiple_results(results=[
		  'segmentation', 
		  'edgeEndpoints', 'edgeMidpoints'])
		self.segmentation = self.dm.load_pipeline_result('segmentation')
		self.n_superpixels = self.dm.segmentation.max() + 1

		self.seg_loaded = True
		sys.stderr.write('segmentation loaded.\n')

		sys.stderr.write('loading sp props...\n')
		self.statusBar().showMessage('loading sp properties..')
		# self.sp_centroids = self.dm.load_pipeline_result('spCentroids')
		# self.sp_bboxes = self.dm.load_pipeline_result('spBbox')
		sys.stderr.write('sp properties loaded.\n')

		self.statusBar().showMessage('')

		# self.sp_rectlist = [None for _ in range(self.dm.n_superpixels)]


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
				if self.selected_proposal_polygon not in self.accepted_proposals:
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

	def display_option_changed(self):
		if self.sender() == self.buttonSpOnOff:

			if not self.superpixels_on:
				self.turn_superpixels_on()
			else:
				self.turn_superpixels_off()
		else:
			print 'not implemented'
			return

			# if self.under_img is not None:
			#   self.under_img.remove()

			self.axis.clear()

			if self.sender() == self.img_radioButton:

				# self.axis.clear()
				# self.axis.axis('off')

				# self.under_img = self.axis.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
				self.axis.imshow(self.dm.image_rgb_jpg, aspect='equal', cmap=plt.cm.Greys_r)
				# self.superpixels_on = False

			elif self.sender() == self.textonmap_radioButton:

				# self.axis.clear()
				# self.axis.axis('off')

				if self.textonmap_vis is None:
					self.textonmap_vis = self.dm.load_pipeline_result('texMapViz')

				# if self.under_img is not None:
				#   self.under_img.remove()

				# self.under_img = self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
				self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
				# self.superpixels_on = False

			elif self.sender() == self.dirmap_radioButton:

				# self.axis.clear()
				# self.axis.axis('off')

				if self.dirmap_vis is None:
					self.dirmap_vis = self.dm.load_pipeline_result('dirMap', 'jpg')
					self.dirmap_vis[~self.dm.mask] = 0


				# self.under_img = self.axis.imshow(self.dirmap_vis, aspect='equal')
				self.axis.imshow(self.dirmap_vis, aspect='equal')

				# if not self.seg_loaded:
				#   self.load_segmentation()

				# self.superpixels_on = False

			# elif self.sender() == self.labeling_radioButton:
			#   pass

		self.axis.axis('off')
		# self.axis.set_xlim([self.newxmin, self.newxmax])
		# self.axis.set_ylim([self.newymin, self.newymax])
		# self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
		self.canvas.draw()

		self.axis2.axis('off')
		# self.axis2.set_xlim([self.newxmin, self.newxmax])
		# self.axis2.set_ylim([self.newymin, self.newymax])
		# self.fig2.subplots_adjust(left=0, bottom=0, right=1, top=1)
		self.canvas2.draw()

			   
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
