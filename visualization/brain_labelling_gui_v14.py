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

sys.path.append(os.environ['LOCAL_REPO_DIR'] + '/notebooks')
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
			labeling_dir=os.environ['LOCAL_LABELING_DIR'])

		self.gabor_params_id='blueNisslWide'
		self.segm_params_id='blueNisslRegular'
		self.vq_params_id='blueNissl'

		self.dm.set_gabor_params(gabor_params_id=self.gabor_params_id)
		self.dm.set_segmentation_params(segm_params_id=self.segm_params_id)
		self.dm.set_vq_params(vq_params_id=self.vq_params_id)

		if (stack is None or section is None) and self.parent_labeling_name is not None:
			stack, section_str, user, timestamp = self.parent_labeling_name[:-4].split('_')
			section = int(section_str)

		self.dm.set_image(stack, 'x5', section)

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

	def paramSettings_clicked(self):
		pass

	def load_labeling(self):

		# self.masked_img = self.dm.image_rgb.copy()
		self.masked_img = gray2rgb(self.dm.image)
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
			polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, hatch='/')
		elif polygon_type == PolygonType.TEXTURE_WITH_CONTOUR:
			polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, hatch='x')
		elif polygon_type == PolygonType.DIRECTION:
			polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[self.curr_label + 1], linewidth=2, linestyle='dashed')
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

		self.spOnOffSlider.valueChanged.connect(self.display_option_changed)


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
						polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2, hatch='/')
					elif polygon_type == PolygonType.TEXTURE_WITH_CONTOUR:
						polygon = Polygon(vertices, closed=True, fill=False, edgecolor=self.colors[label + 1], linewidth=2, hatch='x')
					elif polygon_type == PolygonType.DIRECTION:
						polygon = Polygon(vertices, closed=False, fill=False, edgecolor=self.colors[label + 1], linewidth=2, linestyle='dashed')
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
			print 'picked polygon', self.selected_polygon_id, 'vertex', self.selected_vertex_index
			self.selected_circle = event.artist
			self.selected_circle.set_radius(20.)
			self.selected_polygon = self.polygon_list[self.selected_polygon_id]
			self.selected_polygon.set_linewidth(5.)	
		elif event.artist in self.polygon_list: # picked on a polygon

			if self.selected_circle is None or not mplPath.Path(event.artist.get_xy()).contains_point(self.selected_circle.center):

				if self.selected_circle is not None:
					self.selected_circle.set_radius(10.)
							
				if self.selected_polygon is not None:
					self.selected_polygon.set_linewidth(2.)

				self.selected_polygon = event.artist
				self.selected_polygon_id = self.polygon_list.index(self.selected_polygon)
				print 'picked polygon', self.selected_polygon_id
				self.selected_polygon.set_linewidth(5.)
		
		self.canvas.draw()

		self.click_on_object = True
		print polygon_ids, 'set', self.click_on_object



	def display_option_changed(self):
		if self.sender() == self.spOnOffSlider:

			if self.spOnOffSlider.value() == 1:

				if self.segm_transparent is None:
					self.segm_transparent = self.dm.load_pipeline_result('segmentationTransparent', 'png')
					self.my_cmap = plt.cm.Reds
					self.my_cmap.set_under(color="white", alpha="0")

				if not self.seg_loaded:
					self.load_segmentation()

				self.seg_enabled = True
			else:
				self.segm_handle.remove()
				self.seg_enabled = False

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
					self.textonmap_vis = self.dm.load_pipeline_result('texMap', 'png')

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

				self.seg_vis = self.dm.load_pipeline_result('segmentationWithText', 'jpg')
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

		labelmap_vis = np.zeros_like(self.dm.image_rgb)

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
		self._save_labeling()

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


		print self.click_on_object

		if not self.click_on_object:
			if self.selected_circle is not None:
				self.selected_circle.set_radius(10.)
				self.selected_circle = None

			if self.selected_polygon is not None:
				self.selected_polygon.set_linewidth(2.)
				self.selected_polygon = None

		self.click_on_object = False

		print  self.mode, 

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


	def show_region(self, sp_ind):

		for i, r in enumerate(self.sp_rectlist):
			if r is not None:
				r.remove()
				self.sp_rectlist[i] = None

		for i in self.cluster_sps[sp_ind]:

			ymin, xmin, ymax, xmax = self.sp_props[i, 4:]
			width = xmax - xmin
			height = ymax - ymin

			rect = Rectangle((xmin, ymin), width, height, 
				ec="none", alpha=.3, color=self.colors[self.curr_label + 1])

			self.sp_rectlist[i] = rect
			self.axis.add_patch(rect)

		self.statusBar().showMessage('Sp %d, cluster score %.4f' % (sp_ind, self.curr_cluster_score_sps[sp_ind]))


	def paint_superpixel(self, sp_ind):

		if self.curr_label == self.sp_labellist[sp_ind]:

			self.statusBar().showMessage('Superpixel already has the selected label')

		elif self.curr_label != -1:

			self.sp_labellist[sp_ind] = self.curr_label
			# self.labelmap = self.sp_labellist[self.segmentation]

			### Removes previous color to prevent a blending of two or more patches ###
			if self.sp_rectlist[sp_ind] is not None:
				self.sp_rectlist[sp_ind].remove()

			# approximate the superpixel area with a square

			ymin, xmin, ymax, xmax = self.sp_props[sp_ind, 4:]
			width = xmax - xmin
			height = ymax - ymin

			rect = Rectangle((xmin, ymin), width, height, 
				ec="none", alpha=.3, color=self.colors[self.curr_label + 1])

			self.sp_rectlist[sp_ind] = rect
			self.axis.add_patch(rect)

		else:
			self.statusBar().showMessage("Remove label of superpixel %d" % sp_ind)
			self.sp_labellist[sp_ind] = -1

			self.sp_rectlist[sp_ind].remove()
			self.sp_rectlist[sp_ind] = None

               
if __name__ == "__main__":
	from sys import argv, exit
	a = QApplication(argv)
	labeling_name = sys.argv[1]
	section = int(labeling_name.split('_')[1])
	m = BrainLabelingGUI(stack='RS141', section=section, parent_labeling_name='_'.join(labeling_name.split('_')[2:]))
	
	# m = BrainLabelingGUI(stack='RS141', section=1)
	m.setWindowTitle("Brain Labeling")
	m.showMaximized()
	m.raise_()
	exit(a.exec_())
