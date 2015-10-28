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

from skimage.color import label2rgb

from visualization_utilities import *

sys.path.append(os.environ['LOCAL_REPO_DIR'] + '/utilities')
from utilities2015 import *

IGNORE_EXISTING_LABELNAMES = False

from enum import Enum
class Mode(Enum):
    PLACING_VERTICES = 'placing vertices'
    POLYGON_SELECTED = 'polygon selected'
    IDLE = 'idle'

class PolygonType(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    TEXTURE = 'textured'
    TEXTURE_WITH_CONTOUR = 'texture with contour'
    DIRECTION = 'directionality'

class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
	def __init__(self, parent=None, parent_labeling_name=None, stack=None, section=None):
		"""
		Initialization of BrainLabelingGUI.
		"""

		self.params_dir = '../params'

		# self.app = QApplication(sys.argv)
		QMainWindow.__init__(self, parent)

		self.parent_labeling_name = parent_labeling_name

		self.dm = DataManager(data_dir=os.environ['LOCAL_DATA_DIR'], 
			repo_dir=os.environ['LOCAL_REPO_DIR'], 
			result_dir=os.environ['LOCAL_RESULT_DIR'], 
			labeling_dir=os.environ['LOCAL_LABELING_DIR'],
			stack=stack, section=section, segm_params_id='tSLIC200')

		if (stack is None or section is None) and self.parent_labeling_name is not None:
			stack, section_str, user, timestamp = self.parent_labeling_name[:-4].split('_')
			section = int(section_str)

		self.dm._load_image(versions=['rgb-jpg'])

		self.selected_circle = None
		self.selected_polygon = None

		self.curr_polygon_vertices = []
		self.polygon_list = []
		self.polygon_bbox_list = []
		self.polygon_labels = []
		self.polygon_types = []

		self.curr_polygon_vertex_circles = []
		self.all_polygons_vertex_circles = []
		# self.existing_polygons_vertex_circles = []

		self.highlight_polygon = None

		self.segm_transparent = None
		self.under_img = None
		self.textonmap_vis = None
		self.dirmap_vis = None

		self.debug_mode = False
		self.mode = Mode.IDLE
		self.click_on_object = False

		self.load_labeling()

		# self.data_manager.close()
		self.initialize_brain_labeling_gui()

		self.seg_loaded = False
		self.seg_enabled = False

		self.proposal_mode = False

		self.show_all_accepted = False

	def paramSettings_clicked(self):
		pass

	def load_labeling(self):

		# self.masked_img = self.dm.image_rgb.copy()
		self.masked_img = self.dm.image_rgb_jpg
		# self.masked_img[~self.dm.mask, :] = 0

		if self.parent_labeling_name is None:

			print 'No labeling is given. Initialize labeling.'

			self.parent_labeling = None

			self.curr_labeling = {
				'username' : None,
				'parent_labeling_name' : None,
				'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
				'initial_polygons': None,
				'final_polygons': None,
				'labelnames' : [],  # will be filled by _add_buttons()
			}

		else:

			print self.parent_labeling_name
			self.parent_labeling = self.dm.load_labeling(labeling_name=self.parent_labeling_name)

			print 'Load saved labeling'

			self.curr_labeling = {
				'username' : None,
				'parent_labeling_name' : self.parent_labeling_name,
				'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
				'initial_polygons': self.parent_labeling['final_polygons'],
				'final_polygons': None,
				'labelnames' : [],  # will be filled by _add_buttons()
			}
		
		# self.n_labels = len(self.labeling['labelnames'])

		# initialize GUI variables
		self.paint_label = -1        # color of pen
		self.pick_mode = False       # True while you hold ctrl; used to pick a color from the image
		self.pressed = False           # related to pan (press and drag) vs. select (click)
		self.base_scale = 1.2       # multiplication factor for zoom using scroll wheel
		self.moved = False           # indicates whether mouse has moved while left button is pressed


	def add_polygon(self, vertices, polygon_type):

		self.all_polygons_vertex_circles.append(self.curr_polygon_vertex_circles)

		if polygon_type == PolygonType.CLOSED:
			polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
		elif polygon_type == PolygonType.OPEN:
			polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
		elif polygon_type == PolygonType.TEXTURE:
			# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, hatch='/')
			polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
		elif polygon_type == PolygonType.TEXTURE_WITH_CONTOUR:
			# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, hatch='x')
			polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
		elif polygon_type == PolygonType.DIRECTION:
			# polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, linestyle='dashed')
			polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2)
		else:
			raise 'polygon_type must be one of enum closed, open, texture'

		xys = polygon.get_xy()
		x0_y0_x1_y1 = np.r_[xys.min(axis=0), xys.max(axis=0)]

		self.axis.add_patch(polygon)

		polygon.set_picker(True)

		self.polygon_list.append(polygon)
		self.polygon_bbox_list.append(x0_y0_x1_y1)
		self.polygon_labels.append(self.curr_label)
		self.polygon_types.append(polygon_type)

		self.curr_polygon_vertices = []
		self.curr_polygon_vertex_circles = []

		# for v in self.curr_polygon_vertex_circles:
		# 	v.remove()

	def openMenu(self, canvas_pos):

		self.endDraw_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.endDrawOpen_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.confirmTexture_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.confirmTextureWithContour_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.confirmDirectionality_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
		self.deletePolygon_Action.setVisible(self.selected_polygon is not None)
		self.deleteVertex_Action.setVisible(self.selected_circle is not None)
		self.addVertex_Action.setVisible(self.selected_polygon is not None)

		# if self.proposal_mode:
		self.accProp_Action.setVisible(self.proposal_mode)
		self.rejProp_Action.setVisible(self.proposal_mode)

		action = self.menu.exec_(self.cursor().pos())

		if action == self.endDraw_Action:

			self.add_polygon(self.curr_polygon_vertices, PolygonType.CLOSED)
			self.statusBar().showMessage('Done drawing closed region using label %d (%s)' % (self.curr_label,
														self.curr_labeling['labelnames'][self.curr_label]))
			self.mode = Mode.IDLE

		elif action == self.endDrawOpen_Action:

			self.add_polygon(self.curr_polygon_vertices, PolygonType.OPEN)
			self.statusBar().showMessage('Done drawing edge segment using label %d (%s)' % (self.curr_label,
														self.curr_labeling['labelnames'][self.curr_label]))
			self.mode = Mode.IDLE

		elif action == self.confirmTexture_Action:
			self.add_polygon(self.curr_polygon_vertices, PolygonType.TEXTURE)
			self.statusBar().showMessage('Done drawing textured regions using label %d (%s)' % (self.curr_label,
														self.curr_labeling['labelnames'][self.curr_label]))

			self.mode = Mode.IDLE

		elif action == self.confirmTextureWithContour_Action:
			self.add_polygon(self.curr_polygon_vertices, PolygonType.TEXTURE_WITH_CONTOUR)
			self.statusBar().showMessage('Done drawing textured regions using label %d (%s)' % (self.curr_label,
														self.curr_labeling['labelnames'][self.curr_label]))
			self.mode = Mode.IDLE


		elif action == self.confirmDirectionality_Action:
			self.add_polygon(self.curr_polygon_vertices, PolygonType.DIRECTION)
			self.statusBar().showMessage('Done drawing striated regions using label %d (%s)' % (self.curr_label,
														self.curr_labeling['labelnames'][self.curr_label]))
			self.mode = Mode.IDLE

		elif action == self.deletePolygon_Action:
			self.remove_polygon()

		elif action == self.deleteVertex_Action:
			self.remove_selected_vertex()

		elif action == self.addVertex_Action:
			self.add_vertex_to_existing_polygon(canvas_pos)

		elif action == self.crossReference_Action:
			self.parent().refresh_data()
			self.parent().comboBoxBrowseMode.setCurrentIndex(self.curr_label + 1)
			self.parent().set_labelnameFilter(self.curr_label)
			self.parent().switch_to_labeling()

		elif action == self.accProp_Action:
			self.accProp_callback()

		elif action == self.rejProp_Action:
			self.rejProp_callback()

		else:
			# raise 'do not know how to deal with action %s' % action
			pass

	def initialize_brain_labeling_gui(self):

		self.menu = QMenu()
		self.endDraw_Action = self.menu.addAction("Confirm closed contour")
		self.endDrawOpen_Action = self.menu.addAction("Confirm open boundary")
		self.confirmTexture_Action = self.menu.addAction("Confirm textured region without contour")
		self.confirmTextureWithContour_Action = self.menu.addAction("Confirm textured region with contour")
		self.confirmDirectionality_Action = self.menu.addAction("Confirm striated region")

		self.deletePolygon_Action = self.menu.addAction("Delete polygon")
		self.deleteVertex_Action = self.menu.addAction("Delete vertex")
		self.addVertex_Action = self.menu.addAction("Add vertex")

		self.crossReference_Action = self.menu.addAction("Cross reference")

		self.accProp_Action = self.menu.addAction("Accept")
		self.rejProp_Action = self.menu.addAction("Reject")

		# A set of high-contrast colors proposed by Green-Armytage
		self.colors = np.loadtxt('100colors.txt', skiprows=1)
		self.label_cmap = ListedColormap(self.colors, name='label_cmap')

		self.curr_label = -1

		self.setupUi(self)

		self.fig = self.canvaswidget.fig
		self.canvas = self.canvaswidget.canvas

		self.canvas.mpl_connect('scroll_event', self.zoom_fun)
		self.bpe_id = self.canvas.mpl_connect('button_press_event', self.press_fun)
		self.bre_id = self.canvas.mpl_connect('button_release_event', self.release_fun)
		self.canvas.mpl_connect('motion_notify_event', self.motion_fun)

		self.display_buttons = [self.img_radioButton, self.textonmap_radioButton, self.dirmap_radioButton, self.labeling_radioButton]
		self.img_radioButton.setChecked(True)

		for b in self.display_buttons:
			b.toggled.connect(self.display_option_changed)

		self.buttonSpOnOff.clicked.connect(self.display_option_changed)


		self.canvas.mpl_connect('pick_event', self.on_pick)


		########## Label buttons #############

		self.n_labelbuttons = 0
		
		self.labelbuttons = []
		self.labeledits = []

		if self.parent_labeling is not None:
			for n in self.parent_labeling['labelnames']:
				self._add_labelbutton(desc=n)
		else:
			for n in self.dm.labelnames:
				self._add_labelbutton(desc=n)

		# self.loadButton.clicked.connect(self.load_callback)
		self.saveButton.clicked.connect(self.save_callback)
		self.newLabelButton.clicked.connect(self.newlabel_callback)
		# self.newLabelButton.clicked.connect(self.sigboost_callback)
		self.quitButton.clicked.connect(self.close)
		self.buttonParams.clicked.connect(self.paramSettings_clicked)
		self.buttonGenProposals.clicked.connect(self.reviewProposals_callback)
		# self.buttonPrevProp.clicked.connect(self.prevProp_callback)
		# self.buttonNextProp.clicked.connect(self.nextProp_callback)
		# self.buttonAccProp.clicked.connect(self.accProp_callback)
		# self.buttonRejProp.clicked.connect(self.rejProp_callback)

		self.buttonShowAllAcc.clicked.connect(self.showAllAcc_callback)

		self.setWindowTitle(self.windowTitle() + ', parent_labeling = %s' %(self.parent_labeling_name))

		# self.statusBar().showMessage()       

		self.fig.clear()
		self.fig.set_facecolor('white')

		self.axis = self.fig.add_subplot(111)
		self.axis.axis('off')

		self.axis.imshow(self.masked_img, cmap=plt.cm.Greys_r,aspect='equal')

		if self.curr_labeling['initial_polygons'] is not None:
			for label, typed_polygons in self.curr_labeling['initial_polygons'].iteritems():
				for polygon_type, vertices in typed_polygons:
					if polygon_type == PolygonType.CLOSED:
						polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2)
					elif polygon_type == PolygonType.OPEN:
						polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[label + 1], linewidth=2)
					elif polygon_type == PolygonType.TEXTURE:
						# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2, hatch='/')
						polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2)
					elif polygon_type == PolygonType.TEXTURE_WITH_CONTOUR:
						# polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2, hatch='x')
						polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2)
					elif polygon_type == PolygonType.DIRECTION:
						# polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[label + 1], linewidth=2, linestyle='dashed')
						polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2)
					else:
						raise 'polygon_type must be one of enum closed, open, texture'

					xys = polygon.get_xy()
					x0_y0_x1_y1 = np.r_[xys.min(axis=0), xys.max(axis=0)]

					polygon.set_picker(10.)

					self.axis.add_patch(polygon)
					self.polygon_list.append(polygon)
					self.polygon_bbox_list.append(x0_y0_x1_y1)
					self.polygon_labels.append(label)
					self.polygon_types.append(polygon_type)

					curr_polygon_vertex_circles = []
					for x,y in vertices:
						curr_vertex_circle = plt.Circle((x, y), radius=10, color=self.colors[label + 1], alpha=.8)
						curr_vertex_circle.set_picker(True)
						self.axis.add_patch(curr_vertex_circle)
						curr_polygon_vertex_circles.append(curr_vertex_circle)

					self.all_polygons_vertex_circles.append(curr_polygon_vertex_circles)

		# self.curr_polygon_vertices = []

		self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

		self.newxmin, self.newxmax = self.axis.get_xlim()
		self.newymin, self.newymax = self.axis.get_ylim()

		self.canvas.draw()
		self.show()

		self.sp_rectlist = []

	

	def on_pick(self, event):

		import matplotlib.path as mplPath
			
		polygon_ids = np.where([event.artist in circs for circs in self.all_polygons_vertex_circles])[0]
		# polygon_ids contains either 0 or 1 item
		if len(polygon_ids) > 0: # picked on a vertex circle

			if self.selected_circle is not None:
				self.selected_circle.set_radius(10.)

			if self.selected_polygon is not None:
				self.selected_polygon.set_linewidth(2.)
				
			self.selected_polygon_id = polygon_ids[0]
			self.selected_vertex_index = self.all_polygons_vertex_circles[self.selected_polygon_id].index(event.artist)
			print 'picked polygon', self.selected_polygon_id, 'vertex', self.selected_vertex_index, 'label', self.polygon_labels[self.selected_polygon_id]
			self.selected_circle = event.artist
			self.selected_circle.set_radius(20.)
			self.selected_polygon = self.polygon_list[self.selected_polygon_id]
			self.selected_polygon.set_linewidth(5.)	
			
			self.statusBar().showMessage('picked polygon %d, vertex %d, label %d' % (self.selected_polygon_id, self.selected_vertex_index, self.polygon_labels[self.selected_polygon_id]))

		elif event.artist in self.polygon_list: # picked on a polygon

			if self.selected_circle is None or not mplPath.Path(event.artist.get_xy()).contains_point(self.selected_circle.center):

				if self.selected_circle is not None:
					self.selected_circle.set_radius(10.)
							
				if self.selected_polygon is not None:
					self.selected_polygon.set_linewidth(2.)

				self.selected_polygon = event.artist
				self.selected_polygon_id = self.polygon_list.index(self.selected_polygon)
				print 'picked polygon', self.selected_polygon_id, 'label', self.polygon_labels[self.selected_polygon_id]
				self.selected_polygon.set_linewidth(5.)
			
				self.statusBar().showMessage('picked polygon %d, label %d' % (self.selected_polygon_id, self.polygon_labels[self.selected_polygon_id]))

		self.canvas.draw()

		self.click_on_object = True
		print polygon_ids, 'set', self.click_on_object


	def _add_labelbutton(self, desc=None):
		self.n_labelbuttons += 1

		index = self.n_labelbuttons - 1

		row = (index) % 5
		col = (index) / 5

		btn = QPushButton('%d' % index, self)

		if desc is None:
			labelname = 'label %d'%index			
		else:
			labelname = desc

		edt = QLineEdit(QString(labelname))
		self.curr_labeling['labelnames'].append(labelname)

		self.labelbuttons.append(btn)
		self.labeledits.append(edt)

		btn.clicked.connect(self.labelbutton_callback)
		edt.editingFinished.connect(self.labelNameChanged)

		r, g, b, a = self.label_cmap(index + 1)

		btn.setStyleSheet("background-color: rgb(%d, %d, %d)"%(int(r*255),int(g*255),int(b*255)))
		btn.setFixedSize(20, 20)

		self.labelsLayout.addWidget(btn, row, 2*col)
		self.labelsLayout.addWidget(edt, row, 2*col+1)

	def newlabel_callback(self):
		# self.n_labels += 1
		self._add_labelbutton()

	# def sigboost_callback(self):
	# 	self._save_labeling()

	# def load_callback(self):
	# 	self.initialize_data_manager()

	def load_proposals(self):
		if not self.seg_loaded:
			self.load_segmentation()
		
		self.turn_superpixels_on()

		proposal_models = self.dm.load_pipeline_result('boundaryModels')
		self.proposal_clusters = [m[0] for m in proposal_models]
		self.proposal_dedges = [m[1] for m in proposal_models]
		self.proposal_sigs = [m[2] for m in proposal_models]
		self.n_proposals = len(proposal_models)
		self.proposal_review_results = [None for _ in range(self.n_proposals)]

		self.curr_prop_id = 0
		self.statusBar().showMessage('%d proposals loaded' % (self.n_proposals))

		self.sp_covered_by_proposals = self.dm.load_pipeline_result('spCoveredByProposals')
		self.sp_covered_by_proposals = dict([(s, list(props)) for s, props in self.sp_covered_by_proposals.iteritems()])

		self.proposal_mode = True

		from matplotlib.path import Path
		from matplotlib.patches import PathPatch

		self.proposal_pathPatches = []
		for dedges in self.proposal_dedges:
 		
 			vertices = []
	 		for de_ind, de in enumerate(dedges):
	 			midpt = self.dm.edge_midpoints[frozenset(de)]
	 			pts = self.dm.edge_coords[frozenset(de)]
	 			pts_next_dedge = self.dm.edge_coords[frozenset(dedges[(de_ind+1)%len(dedges)])]

				dij = cdist([pts[0], pts[-1]], [pts_next_dedge[0], pts_next_dedge[-1]])
				i,j = np.unravel_index(np.argmin(dij), (2,2))
				if i == 0:
					vertices += [pts[-1], midpt, pts[0]]
				else:
					vertices += [pts[0], midpt, pts[-1]]

			path_patch = PathPatch(Path(vertices=vertices, closed=True), color=(0,1,1), fill=False, linewidth=3)
			self.proposal_pathPatches.append(path_patch)

		# self.proposal_pathPatches = [PathPatch(Path(vertices=[self.dm.edge_midpoints[frozenset(de)] for de in dedges], closed=True), color=(0,1,1), fill=False)
		# 						for dedges in self.proposal_dedges]

		self.canvas.draw()

	def reviewProposals_callback(self):
		self.review_mode = True
		self.load_proposals()

	def accProp_callback(self):
		self.proposal_review_results[self.curr_prop_id] = True
		# self.buttonAccProp.setDown(True)
		# self.buttonRejProp.setDown(False)
		
		# self.remove_all_sp_highlightBoxes()
		# self.remove_all_proposal_pathpatches()

		# if self.show_all_accepted:
		# 	self.show_all_accepted_proposals()

		self.proposal_pathPatches[self.curr_prop_id].set_color((0,1,0))
		self.canvas.draw()
		# self.show_proposal(self.curr_prop_id)

		self.statusBar().showMessage('Accept proposal %d' % (self.curr_prop_id))

	def rejProp_callback(self):
		self.proposal_review_results[self.curr_prop_id] = False
		# self.buttonRejProp.setDown(True)
		# self.buttonAccProp.setDown(False)

		# self.remove_all_sp_highlightBoxes()
		# self.remove_all_proposal_pathpatches()

		# if self.show_all_accepted:
		# 	self.show_all_accepted_proposals()
		
		# self.show_proposal(self.curr_prop_id)

		self.proposal_pathPatches[self.curr_prop_id].set_color((1,0,0))
		self.canvas.draw()
		
		self.statusBar().showMessage('Reject proposal %d' % (self.curr_prop_id))
	
	def showAllAcc_callback(self):
		if self.show_all_accepted:
			self.buttonShowAllAcc.setText('Show All Accepted')
			# self.remove_all_sp_highlightBoxes()
			# self.remove_all_proposal_pathpatches()
			self.hide_all_accepted_proposals()
			self.show_all_accepted = False
		else:
			self.buttonShowAllAcc.setText('Hide All Accepted')
			self.show_all_accepted_proposals()
			self.show_all_accepted = True

	def show_all_accepted_proposals(self):
		proposal_ids = np.where(self.proposal_review_results)[0]
		self.show_proposals(proposal_ids)

	def hide_all_accepted_proposals(self):
		proposal_ids = np.where(self.proposal_review_results)[0]
		for prop_id in proposal_ids:
			self.proposal_pathPatches[prop_id].remove()
		self.canvas.draw()

	def remove_all_proposal_pathpatches(self):

		for p in self.proposal_pathPatches:
			try:
				p.remove()
			except:
				pass

		# for i, p in enumerate(self.shown_pathpatch_list):
		# 	p.remove()

		self.canvas.draw()

	def remove_all_sp_highlightBoxes(self):
		for i, r in enumerate(self.sp_rectlist):
			if r is not None:
				r.remove()
				self.sp_rectlist[i] = None

		self.canvas.draw()

	def show_proposals(self, proposal_ids):

 		for proposal_id in proposal_ids:

 			# cl = self.proposal_clusters[proposal_id]
			# sc = self.proposal_sigs[proposal_id]
			# self.paint_sps(cl, color=(0,1,0))
			self.axis.add_patch(self.proposal_pathPatches[proposal_id])

		self.canvas.draw()


	def show_proposal(self, proposal_id):

		# cl = self.proposal_clusters[proposal_id]
		sc = self.proposal_sigs[proposal_id]
		decision = self.proposal_review_results[proposal_id]

		if decision is None:
			# self.buttonAccProp.setDown(False)
			# self.buttonRejProp.setDown(False)
			decision_str = ''
			paint_color = (0,1,1)
		
		elif decision:
			# self.buttonAccProp.setDown(True)
			# self.buttonRejProp.setDown(False)
			decision_str = 'accepted'
			paint_color = (0,1,0)
			
		else:
			# self.buttonAccProp.setDown(False)
			# self.buttonRejProp.setDown(True)
			decision_str = 'rejected'
			paint_color = (1,0,0)

		self.proposal_pathPatches[proposal_id].set_color(paint_color)
		self.axis.add_patch(self.proposal_pathPatches[proposal_id])

		# self.paint_sps(cl, color=paint_color)

		self.statusBar().showMessage('proposal %d, score %.4f, %s' % (proposal_id, sc, decision_str))
		self.canvas.draw()


	def show_proposals_covering_sp(self, sp_ind):

		if sp_ind not in self.sp_covered_by_proposals:
			self.statusBar().showMessage('No proposal covers superpixel %d' % sp_ind)
			return 

		if not hasattr(self, 'alternative_proposal_ind'):
			self.alternative_proposal_ind = 0
		else:
			self.alternative_proposal_ind = (self.alternative_proposal_ind + 1) % len(self.sp_covered_by_proposals[sp_ind])

		self.curr_prop_id = self.sp_covered_by_proposals[sp_ind][self.alternative_proposal_ind]

		# self.remove_all_sp_highlightBoxes()
		self.remove_all_proposal_pathpatches()
		
		if self.show_all_accepted:
			self.show_all_accepted_proposals()

		self.show_proposal(self.curr_prop_id)


	def labelNameChanged(self):
		edt = self.sender()
		ind_onscreen = self.labeledits.index(edt) # the first one is "no label"
		self.curr_labeling['labelnames'][ind_onscreen] = str(edt.text())

	def _save_labeling(self, ):

		username, okay = QInputDialog.getText(self, "Username", 
							"Please enter your username:", QLineEdit.Normal, 'anon')
		if not okay: return

		self.username = str(username)

		# each row is (label, type, N-by-2 vertices)
		# self.labeling['final_polygons'] = [(l, t, p.get_xy()/[self.dm.image_width, self.dm.image_height]) for l,p,t in zip(self.polygon_labels, self.polygon_list, self.polygon_types)]

		typed_polygons = [(l, t, p.get_xy()) if t in [PolygonType.OPEN, PolygonType.DIRECTION] else (l, t, p.get_xy()[:-1]) for l,p,t in zip(self.polygon_labels, self.polygon_list, self.polygon_types)]
		self.curr_labeling['final_polygons'] = dict([(label, [(t,p) for l,t,p in group]) for label, group in groupby(sorted(typed_polygons, key=itemgetter(0)), itemgetter(0))])


		self.curr_labeling['logout_time'] = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
		self.curr_labeling['username'] = self.username

		new_labeling_name = self.username + '_' + self.curr_labeling['logout_time']

		# self.labelmap = self.generate_labelmap(self.polygon_list, self.polygon_labels)
		# labelmap_vis = label2rgb(self.labelmap, image=self.masked_img, colors=self.colors[1:], 
		# 				bg_label=-1, bg_color=(1,1,1), alpha=0.3, image_alpha=1.)

		labelmap_vis = np.zeros((self.dm.image_height, self.dm.image_width))

		new_labeling_fn = self.dm.save_labeling(self.curr_labeling, new_labeling_name, labelmap_vis)

		print 'Curr labelnames', self.curr_labeling['labelnames']

		new_labelnames = []
		q = [n.lower() for n in self.dm.labelnames]
		for n in self.curr_labeling['labelnames']:
			if n.lower() not in q:
				new_labelnames.append(n)

		print 'Global Labelnames', self.dm.labelnames + new_labelnames

		self.dm.set_labelnames(self.dm.labelnames + new_labelnames)
		
		self.statusBar().showMessage('Labeling saved to %s' % new_labeling_fn )


	def save_callback(self):
		if self.proposal_mode:
			self._save_proposal_review_results()
		else:
			self._save_labeling()

	def _save_proposal_review_results(self):
		timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
		self.dm.save_proposal_review_result(self.proposal_review_results, self.username, timestamp)

	def labelbutton_callback(self):
		self.statusBar().showMessage('Left click to drop vertices')
		self.mode = Mode.PLACING_VERTICES
		self.pick_color(int(self.sender().text()))

	############################################
	# matplotlib canvas CALLBACKs
	############################################

	def zoom_fun(self, event):
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

	def press_fun(self, event):
		self.press_x = event.xdata
		self.press_y = event.ydata

		if self.selected_polygon is not None:
			self.selected_polygon_xy0 = self.selected_polygon.get_xy()
			self.selected_polygon_circle_centers0 = [circ.center for circ in self.all_polygons_vertex_circles[self.selected_polygon_id]]

		self.pressed = True
		self.press_time = time.time()

	def motion_fun(self, event):
		
		if self.selected_circle is not None and self.pressed and self.click_on_object: # drag vertex

			print 'dragging vertex'

			self.selected_circle.center = event.xdata, event.ydata

			xys = self.selected_polygon.get_xy()
			xys[self.selected_vertex_index] = self.selected_circle.center
			if self.polygon_list[self.selected_polygon_id].get_closed():
				self.selected_polygon.set_xy(xys[:-1])
			else:
				self.selected_polygon.set_xy(xys)
			
			self.canvas.draw()

		elif self.selected_polygon is not None and self.pressed:

			print 'dragging polygon'

			offset_x = event.xdata - self.press_x
			offset_y = event.ydata - self.press_y

			for c, center0 in zip(self.all_polygons_vertex_circles[self.selected_polygon_id], self.selected_polygon_circle_centers0):
				c.center = (center0[0] + offset_x, center0[1] + offset_y)

			xys = self.selected_polygon_xy0 + (offset_x, offset_y)
			if self.polygon_list[self.selected_polygon_id].get_closed():
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



	def remove_polygon(self):
		self.selected_polygon.remove()
		self.polygon_list.remove(self.selected_polygon)
		del self.polygon_types[self.selected_polygon_id]
		del self.polygon_bbox_list[self.selected_polygon_id]
		del self.polygon_labels[self.selected_polygon_id]

		selected_vertex_circles = self.all_polygons_vertex_circles[self.selected_polygon_id]
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

		vertex_circle = plt.Circle(pos, radius=10, color=self.colors[self.polygon_labels[self.selected_polygon_id] + 1], alpha=.8)
		self.axis.add_patch(vertex_circle)

		self.all_polygons_vertex_circles[self.selected_polygon_id].insert(new_vertex_ind, vertex_circle)

		vertex_circle.set_picker(True)

		self.canvas.draw()


	def remove_selected_vertex(self):
		self.selected_circle.remove()
		self.all_polygons_vertex_circles[self.selected_polygon_id].remove(self.selected_circle)
		p = self.polygon_list[self.selected_polygon_id]
		xys = p.get_xy()
		xys = np.vstack([xys[:self.selected_vertex_index], xys[self.selected_vertex_index+1:]])
		self.polygon_list[self.selected_polygon_id].set_xy(xys[:-1] if p.get_closed() else xys)

		self.canvas.draw()


	def place_vertex(self, x,y):
		self.curr_polygon_vertices.append([x, y])

		curr_vertex_circle = plt.Circle((x, y), radius=10, color=self.colors[self.curr_label + 1], alpha=.8)
		self.axis.add_patch(curr_vertex_circle)
		self.curr_polygon_vertex_circles.append(curr_vertex_circle)

		curr_vertex_circle.set_picker(True)


	def release_fun(self, event):
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

			if event.button == 1: # left click: place a control point
			
				if self.mode == Mode.PLACING_VERTICES:

					self.place_vertex(event.xdata, event.ydata)

					# polygon_vertex_circle = plt.Circle((event.xdata, event.ydata), radius=10, color=self.colors[self.curr_label + 1], alpha=.8)
					
					# self.axis.add_patch(polygon_vertex_circle)
					# self.existing_polygons_vertex_circles.append(polygon_vertex_circle)

					self.statusBar().showMessage('... in the process of labeling region using label %d (%s)' % (self.curr_label, self.curr_labeling['labelnames'][self.curr_label]))

				elif self.seg_enabled:
					self.handle_sp_press(event.xdata, event.ydata)


			elif event.button == 3: # right click: open context menu
				canvas_pos = (event.xdata, event.ydata)
				self.openMenu(canvas_pos)

			#     self.statusBar().showMessage('Erase %d (%s)' % (self.curr_label, self.labeling['labelnames'][self.curr_label + 1]))
			#     self.erase_circles_near(event.xdata, event.ydata)

		print self.mode

		self.canvas.draw() # force re-draw

	############################################
	# other functions
	############################################

	def pick_color(self, selected_label):

		self.curr_label = selected_label
		self.statusBar().showMessage('Picked label %d (%s)' % (self.curr_label, self.curr_labeling['labelnames'][self.curr_label]))

	def handle_sp_press(self, x, y):
		self.clicked_sp = self.dm.segmentation[int(y), int(x)]
		sys.stderr.write('clicked sp %d\n'%self.clicked_sp)
		if self.proposal_mode:
			self.show_proposals_covering_sp(self.clicked_sp)
		else:
			self.show_region(self.clicked_sp)


	def load_segmentation(self):
		sys.stderr.write('loading segmentation...\n')
		self.statusBar().showMessage('loading segmentation...')

		self.dm.load_multiple_results(results=['segmentation', 'neighbors', 
                                  'edgeCoords', 'spCentroids', 'edgeNeighbors', 'dedgeNeighbors', 'edgeMidpoints'])

		# self.segmentation = self.dm.load_pipeline_result('segmentation')
		self.n_superpixels = self.dm.segmentation.max() + 1
		self.seg_loaded = True
		sys.stderr.write('segmentation loaded.\n')

		sys.stderr.write('loading clusters...\n')
		self.statusBar().showMessage('loading clusters..')
		cluster_tuples = self.dm.load_pipeline_result('allSeedClusterScoreDedgeTuples')
		from collections import defaultdict
		self.region_score_tuples = defaultdict(list)
		for s, cl, sc, ed in cluster_tuples:
			self.region_score_tuples[s].append((sc, cl))
		sys.stderr.write('regions loaded.\n')

		sys.stderr.write('loading sp props...\n')
		self.statusBar().showMessage('loading sp properties..')
		# self.sp_centroids = self.dm.load_pipeline_result('spCentroids')
		# self.sp_bboxes = self.dm.load_pipeline_result('spBbox')
		sys.stderr.write('sp properties loaded.\n')

		self.statusBar().showMessage('')

		self.sp_rectlist = [None for _ in range(self.n_superpixels)]

	def turn_superpixels_on(self):
		self.statusBar().showMessage('Turning supepixels on...')

		self.buttonSpOnOff.setText('Turn Superpixels OFF')

		if self.segm_transparent is None:
			self.segm_transparent = self.dm.load_pipeline_result('segmentationTransparent')
			self.my_cmap = plt.cm.Reds
			self.my_cmap.set_under(color="white", alpha="0")

		if not self.seg_loaded:
			self.load_segmentation()

		self.seg_enabled = True

		self.axis.clear()
		self.axis.axis('off')
		
		self.seg_vis = self.dm.load_pipeline_result('segmentationWithText')
		# self.seg_vis[~self.dm.mask] = 0
		self.axis.imshow(self.seg_vis, aspect='equal')
		self.canvas.draw()


	def display_option_changed(self):
		if self.sender() == self.buttonSpOnOff:

			if not self.seg_enabled:
				self.turn_superpixels_on()

			else:
				self.buttonSpOnOff.setText('Turn Superpixels ON')

				print 'superpixels off'
				self.segm_handle.remove()
				self.seg_enabled = False

				self.axis.clear()
				self.axis.axis('off')
				
				# self.axis.imshow(self.masked_img, cmap=plt.cm.Greys_r,aspect='equal')
				self.axis.imshow(self.masked_img, aspect='equal')

				# self.canvas.draw()

		else:
			# if self.under_img is not None:
			# 	self.under_img.remove()

			self.axis.clear()

			if self.sender() == self.img_radioButton:

				# self.axis.clear()
				# self.axis.axis('off')

				# self.under_img = self.axis.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
				self.axis.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
				# self.seg_enabled = False

			elif self.sender() == self.textonmap_radioButton:

				# self.axis.clear()
				# self.axis.axis('off')

				if self.textonmap_vis is None:
					self.textonmap_vis = self.dm.load_pipeline_result('texMapViz')

				# if self.under_img is not None:
				# 	self.under_img.remove()

				# self.under_img = self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
				self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
				# self.seg_enabled = False

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

				# self.seg_enabled = False

			elif self.sender() == self.labeling_radioButton:


				self.axis.clear()
				self.axis.axis('off')

				if not self.seg_loaded:
					self.load_segmentation()

				# if not self.groups_loaded:
				# 	self.load_groups()
				# else:
				for rect in self.sp_rectlist:
					if rect is not None:
						self.axis.add_patch(rect)

				self.seg_vis = self.dm.load_pipeline_result('segmentationWithText')
				self.seg_vis[~self.dm.mask] = 0
				self.axis.imshow(self.seg_vis, aspect='equal')

			# self.seg_enabled = True

		if self.seg_enabled:
			self.segm_handle = self.axis.imshow(self.segm_transparent, aspect='equal', 
									cmap=self.my_cmap, alpha=1.)

			for i in range(len(self.sp_rectlist)):
				self.sp_rectlist[i] = None

		self.axis.axis('off')

		self.axis.set_xlim([self.newxmin, self.newxmax])
		self.axis.set_ylim([self.newymin, self.newymax])

		self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
		self.canvas.draw()

	def show_region(self, sp_ind):

		for i, r in enumerate(self.sp_rectlist):
			if r is not None:
				r.remove()
				self.sp_rectlist[i] = None

		if not hasattr(self, 'alt_ind'):
			self.alt_ind = 0

		self.alt_ind = (self.alt_ind + 1) % len(self.region_score_tuples[sp_ind])

		sc, cl = self.region_score_tuples[sp_ind][self.alt_ind]

		self.paint_sps(cl, self.colors[self.curr_label + 1])

		# for i in set(cl):

		# 	ymin, xmin, ymax, xmax = self.dm.sp_bboxes[i]
		# 	width = xmax - xmin
		# 	height = ymax - ymin

		# 	rect = Rectangle((xmin, ymin), width, height, ec="none", 
		# 					alpha=.3, color=self.colors[self.curr_label + 1])

		# 	self.sp_rectlist[i] = rect
		# 	self.axis.add_patch(rect)

		self.statusBar().showMessage('Sp %d, cluster score %.4f' % (sp_ind, sc))


	def paint_sps(self, sp_inds, color):
		for sp_ind in sp_inds:
			self.paint_sp(sp_ind, color)

	def paint_sp(self, sp_ind, color):

		if self.sp_rectlist[sp_ind] is not None:
			self.sp_rectlist[sp_ind].remove()
			self.sp_rectlist[sp_ind] = None

		ymin, xmin, ymax, xmax = self.dm.sp_bboxes[sp_ind]
		width = xmax - xmin
		height = ymax - ymin

		rect = Rectangle((xmin, ymin), width, height, ec="none", alpha=.3, color=color)

		self.sp_rectlist[sp_ind] = rect
		self.axis.add_patch(rect)


	# def paint_superpixel(self, sp_ind):

	# 	if self.curr_label == self.sp_labellist[sp_ind]:

	# 		self.statusBar().showMessage('Superpixel already has the selected label')

	# 	elif self.curr_label != -1:

	# 		self.sp_labellist[sp_ind] = self.curr_label
	# 		# self.labelmap = self.sp_labellist[self.segmentation]

	# 		### Removes previous color to prevent a blending of two or more patches ###
	# 		if self.sp_rectlist[sp_ind] is not None:
	# 			self.sp_rectlist[sp_ind].remove()

	# 		# approximate the superpixel area with a square

	# 		ymin, xmin, ymax, xmax = self.sp_bboxes[sp_ind]
	# 		width = xmax - xmin
	# 		height = ymax - ymin

	# 		rect = Rectangle((xmin, ymin), width, height, 
	# 			ec="none", alpha=.3, color=self.colors[self.curr_label + 1])

	# 		self.sp_rectlist[sp_ind] = rect
	# 		self.axis.add_patch(rect)

	# 	else:
	# 		self.statusBar().showMessage("Remove label of superpixel %d" % sp_ind)
	# 		self.sp_labellist[sp_ind] = -1

	# 		self.sp_rectlist[sp_ind].remove()
	# 		self.sp_rectlist[sp_ind] = None

               
if __name__ == "__main__":
	from sys import argv, exit
	a = QApplication(argv)

	# labeling_name = sys.argv[1]
	# section = int(labeling_name.split('_')[1])

	if len(sys.argv) == 2:
		section = int(sys.argv[1])
		m = BrainLabelingGUI(stack='MD593', section=section)
	elif len(sys.argv) == 3:
		section = int(sys.argv[1])
		labeling_name = sys.argv[2]
		m = BrainLabelingGUI(stack='MD593', section=section, parent_labeling_name='_'.join(labeling_name.split('_')[2:]))

	# m = BrainLabelingGUI(stack='RS141', section=1)
	m.setWindowTitle("Brain Labeling")
	m.showMaximized()
	m.raise_()
	exit(a.exec_())
