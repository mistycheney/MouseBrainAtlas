#! /usr/bin/env python

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

from matplotlib.backend_bases import key_press_handler
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

from ui_BrainLabelingGui_v10 import Ui_BrainLabelingGui

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import ListedColormap, NoNorm, ColorConverter
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from skimage.color import label2rgb

from visualization_utilities import *

sys.path.append(os.environ['LOCAL_REPO_DIR'] + '/utilities')
from utilities2015 import *

from collections import defaultdict

IGNORE_EXISTING_LABELNAMES = False

from enum import Enum
class Mode(Enum):
    PLACING_VERTICES = 'placing vertices'
    POLYGON_SELECTED = 'polygon selected'
    REVIEW_PROPOSAL = 'review proposal'

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
		# 	item.setCheckState(False)
			self.selected = self.selected - {str(item.text())}
		# 	print self.selected
		else:
		# 	item.setCheckState(True)
			self.selected.add(str(item.text()))

		print self.selected


	def OnOk(self):
		self.close()

	def OnCancel(self):
		self.selected = set([])
		self.close()



class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
	# def __init__(self, parent=None, parent_labeling_name=None, stack=None, section=None):
	def __init__(self, parent=None, stack=None, section=None):
		"""
		Initialization of BrainLabelingGUI.
		"""

		self.params_dir = '../params'

		# self.app = QApplication(sys.argv)
		QMainWindow.__init__(self, parent)

		# self.parent_labeling_name = parent_labeling_name

		self.init_data(stack, section)
		self.initialize_brain_labeling_gui()

	def download_result(self, results):
		for result_name in results:
			filename = self.dm._get_result_filename(result_name, include_path=False)
			cmd = "rsync -az yuncong@gcn-20-33.sdsc.edu:%(gordon_result_dir)s/%(stack)s/%(section)s/%(filename)s %(local_result_dir)s/%(stack)s/%(section)s/ " % {'gordon_result_dir':os.environ['GORDON_RESULT_DIR'],
																				'local_result_dir':os.environ['LOCAL_RESULT_DIR'],
																				'stack': self.stack,
																				'section': self.dm.slice_str,
																				'filename': filename
																				}
			# print cmd
			os.system(cmd)


	def init_data(self, stack, section):

		self.stack = stack
		self.section = section

		self.dm = DataManager(
			data_dir=os.environ['LOCAL_DATA_DIR'], 
                 repo_dir=os.environ['LOCAL_REPO_DIR'], 
                 result_dir=os.environ['LOCAL_RESULT_DIR'], 
                 labeling_dir=os.environ['LOCAL_LABELING_DIR'],
			stack=stack, section=section, segm_params_id='tSLIC200')

		print self.dm.slice_ind

		# if (stack is None or section is None) and self.parent_labeling_name is not None:
		# 	stack, section_str, user, timestamp = self.parent_labeling_name[:-4].split('_')
		# 	section = int(section_str)

		self.dm._load_image(versions=['rgb-jpg'])

		required_results = ['segmentationTransparent', 
					'segmentation',
		'segmentationWithText',
		'allSeedClusterScoreDedgeTuples',
		'proposals',
		'spCoveredByProposals',
		'edgeMidpoints',
		'edgeEndpoints',
		# 'spAreas',
		# 'spBbox',
		# 'spCentroids'
		]

		self.download_result(required_results)
		self.dm.load_multiple_results(required_results)

		self.selected_circle = None
		self.selected_polygon = None

		self.curr_polygon_vertices = []
		self.curr_polygon_vertex_circles = []

		# self.freeform_polygons = []
		self.polygon_types = []

		# self.all_polygons_vertex_circles = []
		# self.existing_polygons_vertex_circles = []

		self.highlight_polygon = None

		self.segm_transparent = None
		self.under_img = None
		self.textonmap_vis = None
		self.dirmap_vis = None

		self.mode = Mode.REVIEW_PROPOSAL
		self.click_on_object = False

		self.seg_loaded = False
		self.superpixels_on = False

		self.boundary_colors = [(0,1,1), (0,1,0), (1,0,0),(0,0,1)] # unknown, accepted, rejected

		self.accepted_proposals = defaultdict(dict)

		self.curr_proposal_pathPatch = None
		self.alternative_global_proposal_ind = 0
		self.alternative_local_proposal_ind = 0

		# self.user_approved_local_proposals = []
		# self.user_approved_local_pathPatches = []

		# self.user_approved_global_proposals = []
		# self.user_approved_global_pathPatches = []

		# self.user_defined_proposals = []
		# self.user_defined_pathPatches = []


		# self.alg_proposal_pathPatches = {}
		# self.alg_proposals = {}

		# self.global_proposal_pathPatches = []
		# self.global_proposal_labels = []
		# self.local_proposal_pathPatches = []
		# self.local_proposal_labels = []


		self.new_labelnames = []
		
		# self.freeform_proposal_labels = []

		self.proposal_picked = False

		self.shuffle_global_proposals = True # instead of local proposals

		self.paint_label = -1        # color of pen
		self.pick_mode = False       # True while you hold ctrl; used to pick a color from the image
		self.pressed = False           # related to pan (press and drag) vs. select (click)
		self.base_scale = 1.2       # multiplication factor for zoom using scroll wheel
		self.moved = False           # indicates whether mouse has moved while left button is pressed

	

	def paramSettings_clicked(self):
		pass


	# def add_polygon(self, vertices, polygon_type):

	# 	self.all_polygons_vertex_circles.append(self.curr_polygon_vertex_circles)

	# 	if polygon_type == PolygonType.CLOSED:
	# 		# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
	# 		polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=2)
	# 	elif polygon_type == PolygonType.OPEN:
	# 		# polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
	# 		polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.boundary_colors[1], linewidth=2)
	# 	elif polygon_type == PolygonType.TEXTURE:
	# 		# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, hatch='/')
	# 		# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
	# 		polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=2)
			
	# 	elif polygon_type == PolygonType.TEXTURE_WITH_CONTOUR:
	# 		# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, hatch='x')
	# 		polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=2)
	# 	elif polygon_type == PolygonType.DIRECTION:
	# 		# polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, linestyle='dashed')
	# 		polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=2)
	# 	else:
	# 		raise 'polygon_type must be one of enum closed, open, texture'

	# 	# xys = polygon.get_xy()
	# 	# x0_y0_x1_y1 = np.r_[xys.min(axis=0), xys.max(axis=0)]

	# 	self.axis.add_patch(polygon)
	# 	polygon.set_picker(True)

	# 	self.accepted_proposals.

		# self.freeform_polygons.append(polygon)
		# self.polygon_bbox_list.append(x0_y0_x1_y1)
		# self.polygon_labels.append(self.curr_label)
		# self.polygon_types.append(polygon_type)


	def openMenu(self, canvas_pos):

		self.endDrawClosed_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.endDrawOpen_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.confirmTexture_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.confirmTextureWithContour_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.confirmDirectionality_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.deletePolygon_Action.setVisible(self.selected_polygon is not None)
		self.deleteVertex_Action.setVisible(self.selected_circle is not None)
		self.addVertex_Action.setVisible(self.selected_polygon is not None)

		# if self.proposal_mode:

		print self.proposal_picked
		print self.curr_proposal_pathPatch not in self.accepted_proposals

		self.accProp_Action.setVisible(self.curr_proposal_pathPatch is not None and self.curr_proposal_pathPatch not in self.accepted_proposals)
		self.rejProp_Action.setVisible(self.curr_proposal_pathPatch is not None and self.curr_proposal_pathPatch in self.accepted_proposals)
		self.changeLabel_Action.setVisible(self.curr_proposal_pathPatch is not None and self.curr_proposal_pathPatch in self.accepted_proposals)

		action = self.menu.exec_(self.cursor().pos())

		if action == self.endDrawClosed_Action:

			# self.all_polygons_vertex_circles.append(self.curr_polygon_vertex_circles)

			# self.add_polygon(self.curr_polygon_vertices, PolygonType.CLOSED)

			polygon = Polygon(self.curr_polygon_vertices, closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=2)
			self.axis.add_patch(polygon)
			polygon.set_picker(True)

			self.curr_proposal_type = ProposalType.FREEFORM

			self.curr_proposal_pathPatch = polygon

			self.accepted_proposals[self.curr_proposal_pathPatch] = {'type': self.curr_proposal_type,
																	'subtype': PolygonType.CLOSED,
																	'vertices': self.curr_polygon_vertices,
																	'vertexPatches': self.curr_polygon_vertex_circles
																	}

			# self.curr_proposal_id = len(self.freeform_proposal_labels)
			# self.curr_proposal_pathPatch = self.freeform_polygons[self.curr_proposal_id]

			# self.freeform_proposal_labels.append('')

			self.mode = Mode.REVIEW_PROPOSAL
			# self.curr_freeform_polygon_id = len(self.freeform_proposal_labels)
			# self.freeform_proposal_labels.append('')

			self.curr_polygon_vertices = []
			self.curr_polygon_vertex_circles = []

			self.accProp_callback()
			
		elif action == self.endDrawOpen_Action:

			self.add_polygon(self.curr_polygon_vertices, PolygonType.OPEN)
			# self.statusBar().showMessage('Done drawing edge segment using label %d (%s)' % (self.curr_label,
			# 											self.free_proposal_labels[self.curr_label]))

			self.mode = Mode.REVIEW_PROPOSAL

			self.curr_proposal_type = ProposalType.FREEFORM
			self.curr_proposal_id = len(self.freeform_proposal_labels)
			self.curr_proposal_pathPatch = self.freeform_polygons[self.curr_proposal_id]

			self.freeform_proposal_labels.append('')

			self.accProp_callback()

		elif action == self.confirmTexture_Action:
			self.add_polygon(self.curr_polygon_vertices, PolygonType.TEXTURE)
			# self.statusBar().showMessage('Done drawing textured regions using label %d (%s)' % (self.curr_label,
			# 											self.free_proposal_labels[self.curr_label]))
			self.mode = Mode.REVIEW_PROPOSAL

			self.curr_proposal_type = ProposalType.FREEFORM
			self.curr_proposal_id = len(self.freeform_proposal_labels)
			self.curr_proposal_pathPatch = self.freeform_polygons[self.curr_proposal_id]

			self.freeform_proposal_labels.append('')

			self.accProp_callback()

		elif action == self.confirmTextureWithContour_Action:
			self.add_polygon(self.curr_polygon_vertices, PolygonType.TEXTURE_WITH_CONTOUR)
			# self.statusBar().showMessage('Done drawing textured regions using label %d (%s)' % (self.curr_label,
			# 											self.free_proposal_labels[self.curr_label]))
			self.mode = Mode.REVIEW_PROPOSAL

			self.curr_proposal_type = ProposalType.FREEFORM
			self.curr_proposal_id = len(self.freeform_proposal_labels)
			self.curr_proposal_pathPatch = self.freeform_polygons[self.curr_proposal_id]

			self.freeform_proposal_labels.append('')

			self.accProp_callback()

		# elif action == self.confirmDirectionality_Action:
		# 	self.add_polygon(self.curr_polygon_vertices, PolygonType.DIRECTION)
		# 	# self.statusBar().showMessage('Done drawing striated regions using label %d (%s)' % (self.curr_label,
		# 	# 											self.free_proposal_labels[self.curr_label]))
		# 	self.mode = Mode.REVIEW_GLOBAL_PROPOSAL
		# 	self.curr_freeform_polygon_id = len(self.freeform_proposal_labels)
		# 	self.freeform_proposal_labels.append('')
		# 	self.accProp_callback()
	
		elif action == self.deletePolygon_Action:
			self.remove_polygon()

		elif action == self.deleteVertex_Action:
			self.remove_selected_vertex()

		elif action == self.addVertex_Action:
			self.add_vertex_to_existing_polygon(canvas_pos)

		# elif action == self.crossReference_Action:
		# 	self.parent().refresh_data()
		# 	self.parent().comboBoxBrowseMode.setCurrentIndex(self.curr_label + 1)
		# 	self.parent().set_labelnameFilter(self.curr_label)
		# 	self.parent().switch_to_labeling()

		elif action == self.accProp_Action:
			self.accProp_callback()

		elif action == self.rejProp_Action:
			self.rejProp_callback()

		elif action == self.newPolygon_Action:
			self.statusBar().showMessage('Left click to place vertices')
			self.mode = Mode.PLACING_VERTICES

		elif action == self.changeLabel_Action:
			self.open_label_selection_dialog()

		else:
			# raise 'do not know how to deal with action %s' % action
			pass

	def initialize_brain_labeling_gui(self):

		self.menu = QMenu()
		self.endDrawClosed_Action = self.menu.addAction("Confirm closed contour")
		self.endDrawOpen_Action = self.menu.addAction("Confirm open boundary")
		self.confirmTexture_Action = self.menu.addAction("Confirm textured region without contour")
		self.confirmTextureWithContour_Action = self.menu.addAction("Confirm textured region with contour")
		self.confirmDirectionality_Action = self.menu.addAction("Confirm striated region")

		self.deletePolygon_Action = self.menu.addAction("Delete polygon")
		self.deleteVertex_Action = self.menu.addAction("Delete vertex")
		self.addVertex_Action = self.menu.addAction("Add vertex")

		self.newPolygon_Action = self.menu.addAction("New polygon")

		self.crossReference_Action = self.menu.addAction("Cross reference")

		self.accProp_Action = self.menu.addAction("Accept")
		self.rejProp_Action = self.menu.addAction("Reject")

		self.changeLabel_Action = self.menu.addAction('Change label')

		# A set of high-contrast colors proposed by Green-Armytage
		self.colors = np.loadtxt('100colors.txt', skiprows=1)
		self.label_cmap = ListedColormap(self.colors, name='label_cmap')

		self.curr_label = -1

		self.setupUi(self)

		self.fig = self.canvaswidget.fig
		self.canvas = self.canvaswidget.canvas

		self.canvas.mpl_connect('scroll_event', self.on_zoom)
		self.bpe_id = self.canvas.mpl_connect('button_press_event', self.on_press)
		self.bre_id = self.canvas.mpl_connect('button_release_event', self.on_release)
		self.canvas.mpl_connect('motion_notify_event', self.on_motion)

		self.canvas.mpl_connect('pick_event', self.on_pick)
		
		######################################

		self.setWindowTitle(self.windowTitle() + ', stack %s'%self.stack + ', section %d' %self.section)

		######################################

		self.button_autoDetect.clicked.connect(self.autoDetect_callback)
		self.button_loadLabeling.clicked.connect(self.load_callback)
		self.button_saveLabeling.clicked.connect(self.save_callback)
		self.button_quit.clicked.connect(self.close)
		self.buttonParams.clicked.connect(self.paramSettings_clicked)
		self.button_next.clicked.connect(self.next_callback)
		self.button_prev.clicked.connect(self.prev_callback)

		self.fig.clear()
		self.fig.set_facecolor('white')

		self.axis = self.fig.add_subplot(111)
		self.axis.axis('off')

		self.orig_image_handle = self.axis.imshow(self.dm.image_rgb_jpg, cmap=plt.cm.Greys_r,aspect='equal')

		self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

		self.newxmin, self.newxmax = self.axis.get_xlim()
		self.newymin, self.newymax = self.axis.get_ylim()


		##########################################
		self.display_buttons = [self.img_radioButton, self.textonmap_radioButton, self.dirmap_radioButton, self.labeling_radioButton]
		self.img_radioButton.setChecked(True)

		for b in self.display_buttons:
			b.toggled.connect(self.display_option_changed)

		self.radioButton_globalProposal.toggled.connect(self.mode_changed)
		self.radioButton_localProposal.toggled.connect(self.mode_changed)
		self.radioButton_globalProposal.setChecked(True)

		self.buttonSpOnOff.clicked.connect(self.display_option_changed)
		##########################################

		self.canvas.draw()
		self.show()

		self.sp_rectlist = []


	def detect_landmark(self, labels):
		import requests
		payload = {'labels': labels, 'section': self.dm.slice_ind}
		r = requests.get('http://gcn-20-32.sdsc.edu:5000/top_down_detect', params=payload)
		print r.url
		return r.json()

	def autoDetect_callback(self):
		self.labelsToDetect = ListSelection(self.dm.labelnames)
		self.labelsToDetect.exec_()

		if len(self.labelsToDetect.selected) > 0:
		
			returned_alg_proposal_dict = self.detect_landmark(list(self.labelsToDetect.selected)) # list of tuples (sps, dedges, sig)

			for label, (sps, dedges, sig) in returned_alg_proposal_dict.iteritems():
				pp = self.pathPatch_from_dedges(dedges, color=self.boundary_colors[1])
				pp.set_picker(True)
				self.accepted_proposals[pp] = {'sps': sps, 'dedges': dedges, 'sig': sig, 'type':ProposalType.ALGORITHM,
											'label': label}

				self.axis.add_patch(pp)
		
		self.canvas.draw()


	def on_pick(self, event):

		self.cancel_current_proposal()

		patch_vertexInd_tuple = [(patch, props['vertexPatches'].index(event.artist)) for patch, props in self.accepted_proposals.iteritems() 
					if 'vertexPatches' in props and event.artist in props['vertexPatches']]
		
		if len(patch_vertexInd_tuple) == 1:
			print 'clicked on a vertex circle'
			self.curr_proposal_pathPatch = patch_vertexInd_tuple[0][0]
			self.selected_vertex_index = patch_vertexInd_tuple[0][1]

			self.selected_circle = event.artist
			self.selected_circle.set_radius(20.)
			
			self.selected_polygon = self.curr_proposal_pathPatch

			self.curr_proposal_pathPatch.set_linewidth(5.)

			self.statusBar().showMessage('picked %s proposal (%s), vertex %d' % (self.accepted_proposals[self.curr_proposal_pathPatch]['type'].value,
																	 self.accepted_proposals[self.curr_proposal_pathPatch]['label'],
																	 self.selected_vertex_index))

		elif len(patch_vertexInd_tuple) == 0:
			print 'clicked on a polygon'

			self.click_on_object = True

			if event.artist in self.accepted_proposals:

				self.curr_proposal_pathPatch = event.artist
				self.curr_proposal_pathPatch.set_linewidth(5)

				if self.accepted_proposals[self.curr_proposal_pathPatch]['type'] == ProposalType.FREEFORM:
					self.selected_polygon = self.curr_proposal_pathPatch

					self.selected_polygon_xy_before_drag = self.selected_polygon.get_xy()
					self.selected_polygon_circle_centers_before_drag = [circ.center 
										for circ in self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches']]

				self.statusBar().showMessage('picked %s proposal (%s)' % (self.accepted_proposals[self.curr_proposal_pathPatch]['type'].value,
																		 self.accepted_proposals[self.curr_proposal_pathPatch]['label']))

		else:
			raise 'unknown situation'

		self.proposal_picked = True

		self.canvas.draw()

	def pathPatch_from_dedges(self, dedges, color):

	 	vertices = []
 		for de_ind, de in enumerate(dedges):
 			midpt = self.dm.edge_midpoints[frozenset(de)]
 			endpts = self.dm.edge_endpoints[frozenset(de)]
 			endpts_next_dedge = self.dm.edge_endpoints[frozenset(dedges[(de_ind+1)%len(dedges)])]

			dij = cdist([endpts[0], endpts[-1]], [endpts_next_dedge[0], endpts_next_dedge[-1]])
			i,j = np.unravel_index(np.argmin(dij), (2,2))
			if i == 0:
				vertices += [endpts[-1], midpt, endpts[0]]
			else:
				vertices += [endpts[0], midpt, endpts[-1]]

		path_patch = PathPatch(Path(vertices=vertices, closed=True), color=color, fill=False, linewidth=3)

		return path_patch

	def load_local_proposals(self):

		sys.stderr.write('loading local proposals ...\n')
		self.statusBar().showMessage('loading local proposals ...')
		
		cluster_tuples = self.dm.load_pipeline_result('allSeedClusterScoreDedgeTuples')
		self.local_proposal_tuples = [(cl, ed, sig) for seed, cl, sig, ed in cluster_tuples]
		self.local_proposal_clusters = [m[0] for m in self.local_proposal_tuples]
		self.local_proposal_dedges = [m[1] for m in self.local_proposal_tuples]
		self.local_proposal_sigs = [m[2] for m in self.local_proposal_tuples]

		self.n_local_proposals = len(self.local_proposal_tuples)
		
		# if not hasattr(self, 'local_proposal_review_results'):
		# 	self.local_proposal_review_results = [0] * self.n_local_proposals
		if not hasattr(self, 'local_proposal_pathPatches'):
			self.local_proposal_pathPatches = [None] * self.n_local_proposals

		self.local_proposal_indices_from_sp = defaultdict(list)
		for i, (seed, _, _, _) in enumerate(cluster_tuples):
			self.local_proposal_indices_from_sp[seed].append(i)
		self.local_proposal_indices_from_sp.default_factory = None

		sys.stderr.write('%d local proposals loaded.\n' % self.n_local_proposals)
		self.statusBar().showMessage('Local proposals loaded.')

		self.local_proposal_labels = [None] * self.n_local_proposals


	def load_global_proposals(self):
		
		self.global_proposal_tuples = self.dm.load_pipeline_result('proposals')
		self.global_proposal_clusters = [m[0] for m in self.global_proposal_tuples]
		self.global_proposal_dedges = [m[1] for m in self.global_proposal_tuples]
		self.global_proposal_sigs = [m[2] for m in self.global_proposal_tuples]

		self.n_global_proposals = len(self.global_proposal_tuples)

		# if not hasattr(self, 'global_proposal_review_results'):
			# self.global_proposal_review_results = [0] * self.n_global_proposals

		if not hasattr(self, 'global_proposal_pathPatches'):
			self.global_proposal_pathPatches = [None] * self.n_global_proposals

		self.statusBar().showMessage('%d global proposals loaded' % self.n_global_proposals)

		self.sp_covered_by_proposals = self.dm.load_pipeline_result('spCoveredByProposals')
		self.sp_covered_by_proposals = dict([(s, list(props)) for s, props in self.sp_covered_by_proposals.iteritems()])

		self.global_proposal_labels = [None] * self.n_global_proposals

	def load_callback(self):

		fname = str(QFileDialog.getOpenFileName(self, 'Open file', self.dm.labelings_dir))
		stack, sec, username, timestamp, suffix = os.path.basename(fname[:-4]).split('_')

		if suffix == 'consolidated':

			self.accepted_proposals = {}

			accepted_proposal_props = self.dm.load_proposal_review_result(username, timestamp, suffix)

			for props in accepted_proposal_props:
				if props['type'] == ProposalType.GLOBAL or props['type'] == ProposalType.LOCAL or props['type'] == ProposalType.ALGORITHM:
					patch = self.pathPatch_from_dedges(props['dedges'], color=self.boundary_colors[1])
				elif props['type'] == ProposalType.FREEFORM:
					patch = Polygon(vertices, closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=2)
					props['vertexPatches'] = []
					for x,y in props['vertices']:
						vertex_circle = plt.Circle((x, y), radius=10, color=self.boundary_colors[1], alpha=.8)
						vertex_circle.set_picker(True)
						props['vertexPatches'].append(vertex_circle)
						self.axis.add_patch(vertex_circle)

				self.axis.add_patch(patch)
				patch.set_picker(True)

				self.accepted_proposals[patch] = props

		self.canvas.draw()


	def open_label_selection_dialog(self):

		self.label_selection_dialog = QInputDialog(self)
		self.label_selection_dialog.setLabelText('Select landmark label')
		self.label_selection_dialog.setComboBoxItems(['New label'] + sorted(self.dm.labelnames + self.new_labelnames))

		self.label_selection_dialog.textValueSelected.connect(self.label_dialog_text_changed)

		self.label_selection_dialog.exec_()

	def set_selected_proposal_label(self, label):
		self.accepted_proposals[self.curr_proposal_pathPatch]['label'] = label

	def label_dialog_text_changed(self, text):
		
		if str(text) == 'New label':
			label, okay = QInputDialog.getText(self, "New label", 
									"Enter label:", QLineEdit.Normal, 'landmark')
			if not okay:
				return
			else:
				label = str(label)
				self.set_selected_proposal_label(label)

				if label not in self.new_labelnames:
					self.new_labelnames.append(label)
		else:
			self.set_selected_proposal_label(str(text))
		
		self.label_selection_dialog.done(0)


	def accProp_callback(self):

		if self.curr_proposal_type == ProposalType.GLOBAL:
			self.accepted_proposals[self.curr_proposal_pathPatch] = {'sps': self.global_proposal_clusters[self.curr_proposal_id],
																	'dedges': self.global_proposal_dedges[self.curr_proposal_id],
																	'sig': self.global_proposal_sigs[self.curr_proposal_id],
																	'type': self.curr_proposal_type,
																	'id': self.curr_proposal_id}

		elif self.curr_proposal_type == ProposalType.LOCAL:
			self.accepted_proposals[self.curr_proposal_pathPatch] = {'sps': self.local_proposal_clusters[self.curr_proposal_id],
																	'dedges': self.local_proposal_dedges[self.curr_proposal_id],
																	'sig': self.local_proposal_sigs[self.curr_proposal_id],
																	'type': self.curr_proposal_type,
																	'id': self.curr_proposal_id}

		self.curr_proposal_pathPatch.set_color(self.boundary_colors[1])
		self.curr_proposal_pathPatch.set_picker(True)
			
		self.canvas.draw()

		self.open_label_selection_dialog()

	def rejProp_callback(self):

		self.accepted_proposals.pop(self.curr_proposal_pathPatch)

		self.curr_proposal_pathPatch.remove()
		self.curr_proposal_pathPatch.set_color(self.boundary_colors[0])
		self.curr_proposal_pathPatch.set_picker(None)

		self.canvas.draw()
	
	def show_global_proposal_covering_sp(self, sp_ind):

		if sp_ind not in self.sp_covered_by_proposals:
			self.statusBar().showMessage('No proposal covers superpixel %d' % sp_ind)
			return 

		self.cancel_current_proposal()

		self.curr_proposal_type = ProposalType.GLOBAL
		self.alternative_global_proposal_ind = (self.alternative_global_proposal_ind + 1) % len(self.sp_covered_by_proposals[sp_ind])
		self.curr_proposal_id = self.sp_covered_by_proposals[sp_ind][self.alternative_global_proposal_ind]

		if self.global_proposal_pathPatches[self.curr_proposal_id] is None:
			self.global_proposal_pathPatches[self.curr_proposal_id] = self.pathPatch_from_dedges(self.global_proposal_dedges[self.curr_proposal_id], 
										color=self.boundary_colors[0])
		
		self.curr_proposal_pathPatch = self.global_proposal_pathPatches[self.curr_proposal_id]

		if self.curr_proposal_pathPatch not in self.axis.patches:
			self.axis.add_patch(self.curr_proposal_pathPatch)

		self.curr_proposal_pathPatch.set_picker(None)

		if self.curr_proposal_pathPatch in self.accepted_proposals:
			self.curr_proposal_pathPatch.set_linewidth(5.)
			label =  self.accepted_proposals[self.curr_proposal_pathPatch]['label']
		else:
			label = ''			

		self.statusBar().showMessage('global proposal (%s) covering seed %d, score %.4f' % (label, sp_ind, self.global_proposal_sigs[self.curr_proposal_id]))
		self.canvas.draw()

	def show_local_proposal_from_sp(self, sp_ind):

		self.cancel_current_proposal()

		self.curr_proposal_type = ProposalType.LOCAL
		self.alternative_local_proposal_ind = (self.alternative_local_proposal_ind + 1) % len(self.local_proposal_indices_from_sp[sp_ind])
		self.curr_proposal_id = self.local_proposal_indices_from_sp[sp_ind][self.alternative_local_proposal_ind]

		cl, dedges, sig = self.local_proposal_tuples[self.curr_proposal_id]

		if self.local_proposal_pathPatches[self.curr_proposal_id] is None:	
			self.local_proposal_pathPatches[self.curr_proposal_id] = self.pathPatch_from_dedges(dedges, 
																color=self.boundary_colors[0])

		self.curr_proposal_pathPatch = self.local_proposal_pathPatches[self.curr_proposal_id]

		if self.curr_proposal_pathPatch not in self.axis.patches:
			self.axis.add_patch(self.curr_proposal_pathPatch)

		self.curr_proposal_pathPatch.set_picker(None)

		if  self.curr_proposal_pathPatch in self.accepted_proposals:
			self.curr_proposal_pathPatch.set_linewidth(5.)
			label = self.accepted_proposals[self.curr_proposal_pathPatch]['label']
		else:
			label = ''

		self.statusBar().showMessage('local proposal (%s) from seed %d, score %.4f' % (label, sp_ind, sig))
		self.canvas.draw()


	def next_callback(self):

		if hasattr(self, 'global_proposal_tuples'):
			del self.global_proposal_tuples
		if hasattr(self, 'global_proposal_review_results'):
			del self.global_proposal_review_results
		if hasattr(self, 'global_proposal_pathPatches'):
			del self.global_proposal_pathPatches
		if hasattr(self, 'local_proposal_tuples'):
			del self.local_proposal_tuples
		if hasattr(self, 'local_proposal_review_results'):
			del self.local_proposal_review_results
		if hasattr(self, 'local_proposal_pathPatches'):
			del self.local_proposal_pathPatches

		self.init_data(self.stack, self.section+1)
		self.initialize_brain_labeling_gui()

		self.mode_changed()
		
	def prev_callback(self):
		if hasattr(self, 'global_proposal_tuples'):
			del self.global_proposal_tuples
		if hasattr(self, 'global_proposal_review_results'):
			del self.global_proposal_review_results
		if hasattr(self, 'global_proposal_pathPatches'):
			del self.global_proposal_pathPatches
		if hasattr(self, 'local_proposal_tuples'):
			del self.local_proposal_tuples
		if hasattr(self, 'local_proposal_review_results'):
			del self.local_proposal_review_results
		if hasattr(self, 'local_proposal_pathPatches'):
			del self.local_proposal_pathPatches

		self.init_data(self.stack, self.section-1)
		self.initialize_brain_labeling_gui()

		self.mode_changed()


	def save_callback(self):

		timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
		username, okay = QInputDialog.getText(self, "Username", "Please enter your username:", QLineEdit.Normal, 'anon')
		if not okay: return

		self.username = str(username)

		accepted_proposal_props = []
		for patch, props in self.accepted_proposals.iteritems():
			accepted_proposal_props.append(dict([(k,v) for k, v in props.iteritems() if k != 'vertexPatches']))

		print accepted_proposal_props

		self.dm.save_proposal_review_result(accepted_proposal_props, self.username, timestamp, suffix='consolidated')

		self.statusBar().showMessage('Labelings saved to %s' % (self.username+'_'+timestamp))

		# cur_xlim = self.axis.get_xlim()
		# cur_ylim = self.axis.get_ylim()

		# self.axis.set_xlim([0, self.dm.image_width])
		# self.axis.set_ylim([self.dm.image_height, 0])

		# self.fig.savefig('/tmp/preview.jpg', bbox_inches='tight')

		# self.axis.set_xlim(cur_xlim)
		# self.axis.set_ylim(cur_ylim)

		self.canvas.draw()

	def labelbutton_callback(self):
		pass

	############################################
	# matplotlib canvas CALLBACKs
	############################################

	def on_zoom(self, event):
		# get the current x and y limits and subplot position
		cur_pos = self.axis.get_position()
		cur_xlim = self.axis.get_xlim()
		cur_ylim = self.axis.get_ylim()
		
		xdata = event.xdata # get event x location
		ydata = event.ydata # get event y location

		if xdata is None or ydata is None: # mouse position outside data region
			return

		left = xdata - cur_xlim[0]
		right = cur_xlim[1] - xdata
		up = ydata - cur_ylim[0]
		down = cur_ylim[1] - ydata

		# print left, right, up, down

		if event.button == 'up':
			# deal with zoom in
			scale_factor = 1/self.base_scale
		elif event.button == 'down':
			# deal with zoom out
			scale_factor = self.base_scale
		
		self.newxmin = xdata - left*scale_factor
		self.newxmax = xdata + right*scale_factor
		self.newymin = ydata - up*scale_factor
		self.newymax = ydata + down*scale_factor

		self.axis.set_xlim([self.newxmin, self.newxmax])
		self.axis.set_ylim([self.newymin, self.newymax])

		self.canvas.draw() # force re-draw

	def on_press(self, event):
		self.press_x = event.xdata
		self.press_y = event.ydata

		self.pressed = True
		self.press_time = time.time()


	def on_motion(self, event):
		
		if self.selected_circle is not None and self.pressed: # drag vertex

			print 'dragging vertex'

			self.selected_circle.center = event.xdata, event.ydata

			xys = self.selected_polygon.get_xy()
			xys[self.selected_vertex_index] = self.selected_circle.center

			if self.selected_polygon.get_closed():
				self.selected_polygon.set_xy(xys[:-1])
			else:
				self.selected_polygon.set_xy(xys)
			
			self.canvas.draw()

		elif self.selected_polygon is not None and self.pressed and self.proposal_picked: # drag polygon

			print 'dragging polygon'

			offset_x = event.xdata - self.press_x
			offset_y = event.ydata - self.press_y

			for c, center0 in zip(self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches'], 
									self.selected_polygon_circle_centers_before_drag):
				c.center = (center0[0] + offset_x, center0[1] + offset_y)

			xys = self.selected_polygon_xy_before_drag + (offset_x, offset_y)
			
			if self.selected_polygon.get_closed():
				self.selected_polygon.set_xy(xys[:-1])
			else:
				self.selected_polygon.set_xy(xys)
			
			self.canvas.draw()


		elif self.pressed and time.time() - self.press_time > .5:
			# this is drag and move
			cur_xlim = self.axis.get_xlim()
			cur_ylim = self.axis.get_ylim()
			
			if (event.xdata==None) | (event.ydata==None):
				#print 'either event.xdata or event.ydata is None'
				return

			offset_x = self.press_x - event.xdata
			offset_y = self.press_y - event.ydata
			
			self.axis.set_xlim(cur_xlim + offset_x)
			self.axis.set_ylim(cur_ylim + offset_y)
			self.canvas.draw()


	def on_release(self, event):
		"""
		The release-button callback is responsible for picking a color or changing a color.
		"""

		self.pressed = False
		self.release_x = event.xdata
		self.release_y = event.ydata
		self.release_time = time.time()

		if not self.click_on_object:
			if self.selected_circle is not None:
				self.selected_circle.set_radius(10.)
				self.selected_circle = None

			if self.selected_polygon is not None:
				self.selected_polygon.set_linewidth(2.)
				self.selected_polygon = None

		self.click_on_object = False

		print self.mode, 

		# Fixed panning issues by using the time difference between the press and release event
		# Long times refer to a press and hold
		if (self.release_time - self.press_time) < .21 and self.release_x > 0 and self.release_y > 0:
			# fast click

			if event.button == 1: # left click
			
				if self.mode == Mode.PLACING_VERTICES:
					self.place_vertex(event.xdata, event.ydata)

				elif self.superpixels_on:
					if not self.proposal_picked:
						self.handle_sp_press(event.xdata, event.ydata)

			elif event.button == 3: # right click
				canvas_pos = (event.xdata, event.ydata)
				self.openMenu(canvas_pos)

		print self.mode

		self.canvas.draw() # force re-draw

		self.proposal_picked = False

	def remove_polygon(self):
		self.selected_polygon.remove()
		self.freeform_polygons.remove(self.selected_polygon)
		self.selected_polygon = None

		del self.polygon_types[self.curr_freeform_polygon_id]
		# del self.polygon_bbox_list[self.curr_freeform_polygon_id]
		del self.freeform_proposal_labels[self.curr_freeform_polygon_id]

		selected_vertex_circles = self.all_polygons_vertex_circles[self.curr_freeform_polygon_id]
		for circ in selected_vertex_circles:
			circ.remove()
		self.all_polygons_vertex_circles.remove(selected_vertex_circles)

	def add_vertex_to_existing_polygon(self, pos):
		from scipy.spatial.distance import cdist

		xys = self.selected_polygon.get_xy()
		xys = xys[:-1] if self.selected_polygon.get_closed() else xys
		dists = np.squeeze(cdist([pos], xys))
		two_neighbor_inds = np.argsort(dists)[:2]
		new_vertex_ind = max(two_neighbor_inds)
		print 1, xys
		xys = np.insert(xys, new_vertex_ind, pos, axis=0)
		print 2, xys
		self.selected_polygon.set_xy(xys)
		print 3, self.selected_polygon.get_xy()

		vertex_circle = plt.Circle(pos, radius=10, color=self.colors[self.freeform_proposal_labels[self.curr_freeform_polygon_id] + 1], alpha=.8)
		self.axis.add_patch(vertex_circle)

		# self.all_polygons_vertex_circles[self.curr_freeform_polygon_id].insert(new_vertex_ind, vertex_circle)

		vertex_circle.set_picker(True)

		self.canvas.draw()


	def remove_selected_vertex(self):
		self.selected_circle.remove()
		# self.all_polygons_vertex_circles[self.curr_freeform_polygon_id].remove(self.selected_circle)
		p = self.freeform_polygons[self.curr_freeform_polygon_id]
		xys = p.get_xy()
		xys = np.vstack([xys[:self.selected_vertex_index], xys[self.selected_vertex_index+1:]])
		self.freeform_polygons[self.curr_freeform_polygon_id].set_xy(xys[:-1] if p.get_closed() else xys)

		self.canvas.draw()


	def place_vertex(self, x,y):
		self.curr_polygon_vertices.append([x, y])

		# curr_vertex_circle = plt.Circle((x, y), radius=10, color=self.colors[self.curr_label + 1], alpha=.8)
		curr_vertex_circle = plt.Circle((x, y), radius=10, color=self.boundary_colors[1], alpha=.8)
		self.axis.add_patch(curr_vertex_circle)
		self.curr_polygon_vertex_circles.append(curr_vertex_circle)

		curr_vertex_circle.set_picker(True)



	############################################
	# other functions
	############################################

	def pick_color(self, selected_label):
		pass

	def handle_sp_press(self, x, y):
		print 'clicked'
		self.clicked_sp = self.dm.segmentation[int(y), int(x)]
		sys.stderr.write('clicked sp %d\n'%self.clicked_sp)

		self.cancel_current_proposal()

		# self.cancel_curr_global()
		# self.cancel_curr_local()

		if self.mode == Mode.REVIEW_PROPOSAL:
			if self.shuffle_global_proposals:
				self.show_global_proposal_covering_sp(self.clicked_sp)
			else:
				self.show_local_proposal_from_sp(self.clicked_sp)
			# self.show_region(self.clicked_sp)



	def load_segmentation(self):
		sys.stderr.write('loading segmentation...\n')
		self.statusBar().showMessage('loading segmentation...')
		self.dm.load_multiple_results(results=['segmentation', 'edgeEndpoints', 'edgeMidpoints'])
		# self.segmentation = self.dm.load_pipeline_result('segmentation')
		# self.n_superpixels = self.dm.segmentation.max() + 1
		self.seg_loaded = True
		sys.stderr.write('segmentation loaded.\n')

		sys.stderr.write('loading sp props...\n')
		self.statusBar().showMessage('loading sp properties..')
		# self.sp_centroids = self.dm.load_pipeline_result('spCentroids')
		# self.sp_bboxes = self.dm.load_pipeline_result('spBbox')
		sys.stderr.write('sp properties loaded.\n')

		self.statusBar().showMessage('')

		self.sp_rectlist = [None for _ in range(self.dm.n_superpixels)]


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
		
		# self.orig_image_handle.remove()

		self.segm_handle = self.axis.imshow(self.segm_transparent, aspect='equal', 
								cmap=self.my_cmap, alpha=1.)

		# self.axis.clear()
		
		# self.seg_vis = self.dm.load_pipeline_result('segmentationWithText')
		# # self.seg_vis[~self.dm.mask] = 0
		# self.seg_viz_handle = self.axis.imshow(self.seg_vis, aspect='equal')

		# self.canvas.draw()

	def cancel_current_proposal(self):
		if self.curr_proposal_pathPatch is not None:

			# restore line width from 5 to 3
			if self.curr_proposal_pathPatch.get_linewidth() != 3:
				self.curr_proposal_pathPatch.set_linewidth(3)

			if self.curr_proposal_pathPatch in self.axis.patches:
				if self.curr_proposal_pathPatch not in self.accepted_proposals:
					self.curr_proposal_pathPatch.remove()

		self.curr_proposal_pathPatch = None

	def mode_changed(self):

		self.cancel_current_proposal()

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
			# if self.under_img is not None:
			# 	self.under_img.remove()

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
				# 	self.under_img.remove()

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
				# 	self.load_segmentation()

				# self.superpixels_on = False

			elif self.sender() == self.labeling_radioButton:
				pass

		self.axis.axis('off')

		self.axis.set_xlim([self.newxmin, self.newxmax])
		self.axis.set_ylim([self.newymin, self.newymax])

		self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
		self.canvas.draw()

               
if __name__ == "__main__":
	from sys import argv, exit
	appl = QApplication(argv)

	# if len(sys.argv) == 2:
	stack = sys.argv[1]
	section = int(sys.argv[2])
	m = BrainLabelingGUI(stack=stack, section=section)
	# elif len(sys.argv) == 3:
	# 	section = int(sys.argv[1])
	# 	labeling_name = sys.argv[2]
	# 	m = BrainLabelingGUI(stack='MD593', section=section, parent_labeling_name='_'.join(labeling_name.split('_')[2:]))

	# m = BrainLabelingGUI(stack='RS141', section=1)
	# m.setWindowTitle("Brain Labeling")
	m.showMaximized()
	m.raise_()
	exit(appl.exec_())
