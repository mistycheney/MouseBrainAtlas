import sys
import os
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt4agg import (
	FigureCanvasQTAgg as FigureCanvas,
	NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends import qt4_compat

from matplotlib.patches import Rectangle, Polygon

from skimage.color import label2rgb
from random import random

import subprocess

import time
import datetime
from visualization_utilities import *

sys.path.append(os.path.realpath('../notebooks'))
from utilities import *
import json

from pprint import pprint

import cPickle as pickle

from matplotlib.colors import ListedColormap, NoNorm, ColorConverter

use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE

if use_pyside:
	#print 'Using PySide'
	from PySide.QtCore import *
	from PySide.QtGui import *
else:
	#print 'Using PyQt4'
	from PyQt4.QtCore import *
	from PyQt4.QtGui import *

from ui_BrainLabelingGui_v8 import Ui_BrainLabelingGui

IGNORE_EXISTING_LABELNAMES = False

class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
	def __init__(self, parent=None, parent_labeling_name=None, stack=None, section=None):
		"""
		Initialization of BrainLabelingGUI.
		"""
		# Load data

		self.gordon_username = 'yuncong'
		self.gordon_hostname = 'gcn-20-32.sdsc.edu'
		self.temp_dir = '/home/yuncong/BrainLocal'
		self.remote_repo_dir = '/home/yuncong/Brain'
		self.params_dir = '../params'

		# self.image_name = None
		# self.instance_dir = None
		# self.instance_name = None

		# self.app = QApplication(sys.argv)
		QMainWindow.__init__(self, parent)

		self.parent_labeling_name = parent_labeling_name

		self.dm = DataManager(data_dir=os.environ['LOCAL_DATA_DIR'], 
			repo_dir=os.environ['LOCAL_REPO_DIR'], 
			result_dir=os.environ['LOCAL_RESULT_DIR'], labeling_dir=os.environ['LOCAL_LABELING_DIR'])

		self.gabor_params_id='blueNisslWide'
		self.segm_params_id='blueNisslRegular'
		self.vq_params_id='blueNissl'

		self.dm.set_gabor_params(gabor_params_id=self.gabor_params_id)
		self.dm.set_segmentation_params(segm_params_id=self.segm_params_id)
		self.dm.set_vq_params(vq_params_id=self.vq_params_id)

		if stack is None or section is None:
			stack, section_str, user, timestamp = self.parent_labeling_name[:-4].split('_')
			section = int(section_str)

		self.dm.set_image(stack, 'x5', section)

		self.labelnames = self.dm.labelnames

		self.is_placing_vertices = False
		self.curr_polygon_vertices = []
		self.polygon_list = []
		self.polygon_labels = []
		self.vertex_list = []
		self.highlight_polygon = None

		self.segm_transparent = None
		self.under_img = None
		self.textonmap_vis = None
		self.dirmap_vis = None

		self.load_labeling()

		# self.data_manager.close()
		self.initialize_brain_labeling_gui()

	def paramSettings_clicked(self):
		pass
        # self.paramsForm = ParamSettingsForm()
        # self.paramsForm.show()

        # self.gabor_params_id='blueNisslWide'
        # self.segm_params_id='blueNisslRegular'
        # self.vq_params_id='blueNissl'

	def load_labeling(self):

		self.masked_img = self.dm.image_rgb.copy()
		self.masked_img[~self.dm.mask, :] = 0

		try:
			parent_labeling = self.dm.load_labeling(labeling_name=self.parent_labeling_name)

			print 'Load saved labeling'
			
			# label_circles = parent_labeling['final_label_circles']

			self.labeling = {
				'username' : None,
				'parent_labeling_name' : self.parent_labeling_name,
				'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
				'initial_polygons': parent_labeling['final_polygons'],
				'final_polygons': None,
				# 'init_label_circles' : label_circles,
				# 'final_label_circles' : None,
				'labelnames' : parent_labeling['labelnames'],
			}

		except Exception as e:

			print 'error', e
			print 'No labeling is given. Initialize labeling.'

			self.labeling = {
				'username' : None,
				'parent_labeling_name' : None,
				'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
				'initial_polygons': None,
				'final_polygons': None,
				# 'init_label_circles' : [],
				# 'final_label_circles' : None,
				'labelnames' : self.labelnames,
			}

			# labelnames not including -1 ("no label")
			# labelnames_fn = os.path.join(self.dm.labelings_dir, 'labelnames.txt')
			# if os.path.isfile(labelnames_fn) and not IGNORE_EXISTING_LABELNAMES:
			# 	with open(labelnames_fn, 'r') as f:
			# 		labelnames = f.readlines()
				# labelnames = json.load(open(labelnames_fn, 'r'))
			# self.labeling['labelnames'] = 
			# else:
			# 	# n_models = 10
			# 	n_models = 0
		
		self.n_labels = len(self.labeling['labelnames'])

		# initialize GUI variables
		self.paint_label = -1        # color of pen
		self.pick_mode = False       # True while you hold ctrl; used to pick a color from the image
		self.press = False           # related to pan (press and drag) vs. select (click)
		self.base_scale = 1.2       # multiplication factor for zoom using scroll wheel
		self.moved = False           # indicates whether mouse has moved while left button is pressed


	def openMenu(self, canvas_pos):

		if self.seg_enabled:
			self.growRegion_Action.setEnabled(True)
			seed_sp = self.segmentation[int(canvas_pos[1]), int(canvas_pos[0])]
		else:
			self.growRegion_Action.setEnabled(False)

		self.endDraw_Action.setVisible(self.is_placing_vertices)
		self.deletePolygon_Action.setVisible(not self.is_placing_vertices)

		pos = self.cursor().pos()

		action = self.menu.exec_(pos)
		
		if action == self.growRegion_Action:
					
			self.statusBar().showMessage('Grow region from superpixel %d' % seed_sp )
			self.curr_label = self.sp_labellist[seed_sp]
			for sp in self.cluster_sps[seed_sp]:
				self.paint_superpixel(sp)

		elif action == self.pickColor_Action:

			self.pick_color(self.sp_labellist[seed_sp])

		elif action == self.eraseColor_Action:

			self.pick_color(-1)
			self.paint_superpixel(seed_sp)

		elif action == self.eraseAllSpsCurrColor_Action:

			self.pick_color(-1)
			for sp in self.cluster_sps[seed_sp]:
				self.paint_superpixel(sp)

		elif action == self.endDraw_Action:

			self.is_placing_vertices = False

			polygon = Polygon(self.curr_polygon_vertices, closed=True, fill=False,
									edgecolor=self.colors[self.curr_label + 1], linewidth=2)
			self.axes.add_patch(polygon)
			self.polygon_list.append(polygon)
			self.polygon_labels.append(self.curr_label)

			self.curr_polygon_vertices = []

			for v in self.vertex_list:
				v.remove()
			self.vertex_list = []

			self.statusBar().showMessage('Done labeling region using label %d (%s)' % (self.curr_label,
				self.labelnames[self.curr_label]))

		elif action == self.deletePolygon_Action:

			polygon_index = self.polygon_list.index(self.selected_polygon)
			self.selected_polygon.remove()
			del self.polygon_list[polygon_index]
			del self.polygon_labels[polygon_index]
			self.remove_highlight_polygon()

		elif action == self.crossReference_Action:
			pass

			# self.crossRefGallery = CrossReferenceGui()

			# reference_labeling_preview_path_captions = []

			# for labeling in self.dm.inv_labeing_index[self.curr_label]:				
			# 	reference_labeling_preview_path_captions.append((labeling['previewpath'],
			# 	 												labeling['filename']))

			# self.crossRefGallery.set_images(reference_labeling_preview_path_captions,
			# 								callback=self.cross_ref_gallery_callback)


	# def cross_ref_gallery_callback(self):


	def initialize_brain_labeling_gui(self):

		self.menu = QMenu()
		self.growRegion_Action = self.menu.addAction("Label similar neighbors")
		self.pickColor_Action = self.menu.addAction("Pick this label")
		self.eraseColor_Action = self.menu.addAction("Remove label of this superpixel")
		self.eraseAllSpsCurrColor_Action = self.menu.addAction("Clear similar neighbors")
		self.endDraw_Action = self.menu.addAction("End drawing region")
		self.deletePolygon_Action = self.menu.addAction("Delete polygon")
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

		# self.canvas.customContextMenuRequested.connect(self.openMenu)

		# self.display_buttons = [self.img_radioButton, self.imgSeg_radioButton, self.textonmap_radioButton, self.dirmap_radioButton, self.labeling_radioButton]
		self.display_buttons = [self.img_radioButton, self.textonmap_radioButton, self.dirmap_radioButton, self.labeling_radioButton]
		self.img_radioButton.setChecked(True)
		self.seg_enabled = False
		self.seg_loaded = False
		self.groups_loaded = False

		for b in self.display_buttons:
			b.toggled.connect(self.display_option_changed)

		self.spOnOffSlider.valueChanged.connect(self.display_option_changed)

		self.n_labelbuttons = 0
		
		self.labelbuttons = []
		self.labeldescs = []

		self._add_labelbutton(desc='no label')

		for i in range(self.n_labels):
		    self._add_labelbutton(desc=self.labeling['labelnames'][i])

		self.loadButton.clicked.connect(self.load_callback)
		self.saveButton.clicked.connect(self.save_callback)
		self.newLabelButton.clicked.connect(self.newlabel_callback)
		# self.newLabelButton.clicked.connect(self.sigboost_callback)
		self.quitButton.clicked.connect(self.close)
		self.buttonParams.clicked.connect(self.paramSettings_clicked)


		self.brushSizeSlider.valueChanged.connect(self.brushSizeSlider_valueChanged)
		self.brushSizeEdit.setText('%d' % self.brushSizeSlider.value())

		# help_message = 'Usage: right click to pick a color; left click to assign color to a superpixel; scroll to zoom, drag to move'
		# self.setWindowTitle('%s' %(help_message))

		self.setWindowTitle(self.windowTitle() + ', parent_labeling = %s' %(self.parent_labeling_name))

		# self.statusBar().showMessage()       

		self.fig.clear()
		self.fig.set_facecolor('white')

		self.axes = self.fig.add_subplot(111)
		self.axes.axis('off')

		self.axes.imshow(self.masked_img, cmap=plt.cm.Greys_r,aspect='equal')
		
		# labelmap_vis = label2rgb(self.labelmap, image=self.img, colors=self.colors, alpha=0.3, image_alpha=1)
		# self.axes.imshow(labelmap_vis)

		# self.circle_list = [plt.Circle((x,y), radius=r, color=self.colors[l+1], alpha=.3) for x,y,r,l in self.labeling['init_label_circles']]
		# self.labelmap = self.generate_labelmap(self.circle_list)

		if self.labeling['initial_polygons'] is not None:
			for l, xys_percent in self.labeling['initial_polygons']:
				p = Polygon(xys_percent * np.array([self.dm.image_width, self.dm.image_height])[np.newaxis,:],
									closed=True, fill=False, edgecolor=self.colors[l + 1], linewidth=2)
				self.polygon_list.append(p)
				self.polygon_labels.append(l)
				self.axes.add_patch(p)

			self.labelmap = self.generate_labelmap(self.polygon_list, self.polygon_labels)

		self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

		self.newxmin, self.newxmax = self.axes.get_xlim()
		self.newymin, self.newymax = self.axes.get_ylim()

		self.canvas.draw()
		self.show()

	############################################
	# QT button CALLBACKs
	############################################

	# def growThreshSlider_valueChanged(self):
	# 	self.growThreshEdit.setText('%.2f' % (self.growThreshSlider.value()/100.))

	def brushSizeSlider_valueChanged(self, new_val):
		self.brushSizeEdit.setText('%d' % new_val)

	def load_groups(self):

		self.groups = self.dm.load_pipeline_result('groups', 'pkl')

		self.groups_ranked, self.group_scores_ranked = zip(*self.groups)
		
		for i, g in enumerate(self.groups_ranked[:50]):
			self._add_labelbutton()
			self.pick_color(i)
			for sp in g:
				self.paint_superpixel(sp)

		self.groups_loaded = True

	def load_segmentation(self):

		self.segmentation = self.dm.load_pipeline_result('segmentation', 'npy')
		self.n_superpixels = len(np.unique(self.segmentation)) - 1

		self.sp_props = self.dm.load_pipeline_result('spProps', 'npy')
		
		self.sp_labellist = -1*np.ones((self.n_superpixels, ), dtype=np.int)
		self.sp_rectlist = [None for _ in range(self.n_superpixels)]

		# self.neighbors = self.dm.load_pipeline_result('neighbors', 'npy')
		# self.texton_hists = self.dm.load_pipeline_result('texHist', 'npy')
		# self.textonmap = self.dm.load_pipeline_result('texMap', 'npy')

		self.clusters = self.dm.load_pipeline_result('clusters', 'pkl')
		# self.cluster_sps, curr_cluster_score_sps, scores_sps, nulls_sps, models_sps, added_sps = zip(*self.clusters)
		self.cluster_sps, curr_cluster_score_sps = zip(*self.clusters)

		# self.n_texton = 100

		# from scipy.spatial.distance import cdist
		# overall_texton_hist = np.bincount(self.textonmap[self.mask].flat, minlength=self.n_texton)
		# self.overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
		# self.D_sp_null = np.squeeze(cdist(self.texton_hists, [self.overall_texton_hist_normalized], chi2))

		self.seg_loaded = True

	# def display_option_changed(self, checked):
	def display_option_changed(self):
		# if checked:

			# self.axes.clear()

		if self.sender() == self.spOnOffSlider:

			if self.spOnOffSlider.value() == 1:

				if self.segm_transparent is None:
					self.segm_transparent = self.dm.load_pipeline_result('segmentationTransparent', 'png')
					self.my_cmap = plt.cm.Reds
					self.my_cmap.set_under(color="white", alpha="0")

				# self.segm_handle = self.axes.imshow(self.segm_transparent, aspect='equal', 
				# 					cmap=self.my_cmap, alpha=1.)

				if not self.seg_loaded:
					self.load_segmentation()

				self.seg_enabled = True
			else:
				self.segm_handle.remove()
				self.seg_enabled = False

		# elif self.sender() == self.imgSeg_radioButton:

		# 	# self.seg_vis = self.dm.load_pipeline_result('segmentationWithText', 'jpg')
		# 	# self.seg_vis[~self.dm.mask] = 0

		# 	# self.axes.imshow(self.seg_vis, aspect='equal')

		# 	self.axes.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
		# 	self.axes.imshow(self.segm_transparent, aspect='equal', cmap=my_red_cmap, alpha=1.)

		# 	if not self.seg_loaded:
		# 		self.load_segmentation()

		# 	self.seg_enabled = True

		else:
			# if self.under_img is not None:
			# 	self.under_img.remove()

			self.axes.clear()

			if self.sender() == self.img_radioButton:

				# self.axes.clear()
				# self.axes.axis('off')

				# self.under_img = self.axes.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
				self.axes.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
				# self.seg_enabled = False

			elif self.sender() == self.textonmap_radioButton:

				# self.axes.clear()
				# self.axes.axis('off')

				if self.textonmap_vis is None:
					self.textonmap_vis = self.dm.load_pipeline_result('texMap', 'png')

				# if self.under_img is not None:
				# 	self.under_img.remove()

				# self.under_img = self.axes.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
				self.axes.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
				# self.seg_enabled = False

			elif self.sender() == self.dirmap_radioButton:

				# self.axes.clear()
				# self.axes.axis('off')

				if self.dirmap_vis is None:
					self.dirmap_vis = self.dm.load_pipeline_result('dirMap', 'jpg')
					self.dirmap_vis[~self.dm.mask] = 0


				# self.under_img = self.axes.imshow(self.dirmap_vis, aspect='equal')
				self.axes.imshow(self.dirmap_vis, aspect='equal')

				# if not self.seg_loaded:
				# 	self.load_segmentation()

				# self.seg_enabled = False

			elif self.sender() == self.labeling_radioButton:


				self.axes.clear()
				self.axes.axis('off')

				if not self.seg_loaded:
					self.load_segmentation()

				if not self.groups_loaded:
					self.load_groups()
				else:
					for rect in self.sp_rectlist:
						if rect is not None:
							self.axes.add_patch(rect)

				self.seg_vis = self.dm.load_pipeline_result('segmentationWithText', 'jpg')
				self.seg_vis[~self.dm.mask] = 0
				self.axes.imshow(self.seg_vis, aspect='equal')

			# self.seg_enabled = True

		if self.seg_enabled:
			self.segm_handle = self.axes.imshow(self.segm_transparent, aspect='equal', 
									cmap=self.my_cmap, alpha=1.)

		self.axes.axis('off')

		self.axes.set_xlim([self.newxmin, self.newxmax])
		self.axes.set_ylim([self.newymin, self.newymax])

		self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
		self.canvas.draw()


	def _add_labelbutton(self, desc=None):
		self.n_labelbuttons += 1

		index = self.n_labelbuttons - 2

		labelname = desc if desc is not None else 'label %d'%index

		self.labeling['labelnames'].append(labelname)

		row = (index + 1) % 5
		col = (index + 1) / 5

		btn = QPushButton('%d' % index, self)
		edt = QLineEdit(QString(desc if desc is not None else labelname))

		self.labelbuttons.append(btn)
		self.labeldescs.append(edt)

		btn.clicked.connect(self.labelbutton_callback)

		r, g, b, a = self.label_cmap(index + 1)

		btn.setStyleSheet("background-color: rgb(%d, %d, %d)"%(int(r*255),int(g*255),int(b*255)))
		btn.setFixedSize(20, 20)

		self.labelsLayout.addWidget(btn, row, 2*col)
		self.labelsLayout.addWidget(edt, row, 2*col+1)

	def newlabel_callback(self):
		self.n_labels += 1
		self._add_labelbutton()

	def sigboost_callback(self):
		self._save_labeling()


	def load_callback(self):
		self.initialize_data_manager()

	def _save_labeling(self, ):

		username, ok = QInputDialog.getText(self, "Username", 
							"Please enter your username:", QLineEdit.Normal, 'anon')
		if not ok: return

		self.username = str(username)

		# self.axes.imshow(labelmap_vis)
		# for c in self.circle_list:
		#     c.remove()

		# self.circle_list = []

		# self.canvas.draw()

		# self.labeling['final_label_circles'] = self.circle_list_to_labeling_field(self.circle_list)
		self.labeling['final_polygons'] = [(l, p.get_xy()/np.array([self.dm.image_width, self.dm.image_height])[np.newaxis, :]) for l,p in zip(self.polygon_labels, self.polygon_list)]
		self.labeling['logout_time'] = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
		self.labeling['labelnames'] = [str(edt.text()) for edt in self.labeldescs[1:]]

		# labelnames_fn = os.path.join(self.data_dir, 'labelnames.json')

		new_labeling_name = self.username + '_' + self.labeling['logout_time']

		# new_labeling_fn = os.path.join(self.dm.labelings_dir, self.dm.image_name + '_' + new_labeling_name + '.pkl')
		# pickle.dump(self.labeling, open(new_labeling_fn, 'w'))
		# print 'Labeling saved to', new_labeling_fn

		# self.labelmap = self.generate_labelmap(self.circle_list)

		# self.labelmap = self.generate_labelmap(self.circle_list)
		# labelmap_vis = self.colors[self.labelmap]

		self.labelmap = self.generate_labelmap(self.polygon_list, self.polygon_labels)

		labelmap_vis = label2rgb(self.labelmap, image=self.masked_img, colors=self.colors[1:], 
						bg_label=-1, bg_color=(1,1,1), alpha=0.3, image_alpha=1.)

		new_labeling_fn = self.dm.save_labeling(self.labeling, new_labeling_name, labelmap_vis)
		
		self.statusBar().showMessage('Labeling saved to %s' % new_labeling_fn )


	def save_callback(self):
		self._save_labeling()

	def labelbutton_callback(self):
		self.statusBar().showMessage('Left click to select vertices')
		self.is_placing_vertices = True
		self.pick_color(int(self.sender().text()))

	############################################
	# matplotlib canvas CALLBACKs
	############################################

	def zoom_fun(self, event):
		# get the current x and y limits and subplot position
		cur_pos = self.axes.get_position()
		cur_xlim = self.axes.get_xlim()
		cur_ylim = self.axes.get_ylim()
		
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

		# set new limits

		# inv = self.axes.transData.inverted()
		# lb_disp = inv.transform([self.newxmin, self.newymin])
		# rt_disp = inv.transform([self.newxmax, self.newymax])
		# print lb_disp, rt_disp

		self.axes.set_xlim([self.newxmin, self.newxmax])
		self.axes.set_ylim([self.newymin, self.newymax])

		self.canvas.draw() # force re-draw

	def press_fun(self, event):
		self.press_x = event.xdata
		self.press_y = event.ydata
		self.press = True
		self.press_time = time.time()

	def motion_fun(self, event):
			
		if self.press and time.time() - self.press_time > .5:
			# this is drag and move
			cur_xlim = self.axes.get_xlim()
			cur_ylim = self.axes.get_ylim()
			
			if (event.xdata==None) | (event.ydata==None):
				#print 'either event.xdata or event.ydata is None'
				return

			offset_x = self.press_x - event.xdata
			offset_y = self.press_y - event.ydata
			
			self.axes.set_xlim(cur_xlim + offset_x)
			self.axes.set_ylim(cur_ylim + offset_y)
			self.canvas.draw()


	def remove_highlight_polygon(self):
		# Remove the highlight polygon if it exists
		if self.highlight_polygon is not None:
			self.highlight_polygon.remove()
			self.highlight_polygon = None

	def draw_highlight_polygon(self, event):

		if not self.is_placing_vertices:
			# Check if the click is within a polygon. If so, 
			# construct the hightlight polygon over the selected polygon
			containing_polygons = []
			for i, (p,l) in enumerate(zip(self.polygon_list, self.polygon_labels)):
				contains, attrd = p.contains(event)
				if contains:
					containing_polygons.append((p,l))

			if len(containing_polygons) > 0:
				selected_polygon, selected_polygon_label = containing_polygons[0]

				self.selected_polygon = selected_polygon
				self.seletced_polygon_label = selected_polygon_label
				self.statusBar().showMessage('Polygon (%s) is selected' %
					self.labelnames[self.seletced_polygon_label])
				
				self.highlight_polygon = Polygon(selected_polygon.get_xy())
				self.highlight_polygon.update_from(selected_polygon)
				self.highlight_polygon.set_linewidth(5)
				self.axes.add_patch(self.highlight_polygon)

				self.pick_color(self.seletced_polygon_label)


	def release_fun(self, event):
		"""
		The release-button callback is responsible for picking a color or changing a color.
		"""

		self.press = False
		self.release_x = event.xdata
		self.release_y = event.ydata
		self.release_time = time.time()

		# Fixed panning issues by using the time difference between the press and release event
		# Long times refer to a press and hold
		if (self.release_time - self.press_time) < .21 and self.release_x > 0 and self.release_y > 0:

			self.remove_highlight_polygon()
			self.draw_highlight_polygon(event)

			if event.button == 1: # left click: draw
				if self.curr_label is None:
					self.statusBar().showMessage('want to paint, but no label is selected')
				else:
					if self.seg_enabled:

						sp_ind = self.segmentation[int(event.ydata), int(event.xdata)]

						if sp_ind == -1:
							self.statusBar().showMessage('This is the background')
						else:
							self.statusBar().showMessage('Labeled superpixel %d using label %d (%s)' % (sp_ind, self.curr_label, self.labeling['labelnames'][self.curr_label]))
							self.paint_superpixel(sp_ind)

					else:
						if self.is_placing_vertices:
							self.curr_polygon_vertices.append([event.xdata, event.ydata])

							vertex = plt.Circle((event.xdata, event.ydata), radius=10, 
											color=self.colors[self.curr_label + 1], alpha=.8)
							self.axes.add_patch(vertex)
							self.vertex_list.append(vertex)

							self.statusBar().showMessage('... in the process of labeling region using label %d (%s)' % (self.curr_label, self.labeling['labelnames'][self.curr_label]))

			elif event.button == 3: # right click: erase
				canvas_pos = (event.xdata, event.ydata)
				self.openMenu(canvas_pos)


			#     self.statusBar().showMessage('Erase %d (%s)' % (self.curr_label, self.labeling['labelnames'][self.curr_label + 1]))
			#     self.erase_circles_near(event.xdata, event.ydata)

			
		self.canvas.draw() # force re-draw

	############################################
	# other functions
	############################################

	def pick_color(self, selected_label):

		self.curr_label = selected_label
		self.statusBar().showMessage('Picked label %d (%s)' % (self.curr_label, self.labeling['labelnames'][self.curr_label]))

	# def paint_circle(self, x, y):

	# 	if self.curr_label is None:
	# 		self.statusBar().showMessage('No label is selected')
	# 	else:
	# 		pass

			# brush_radius = self.brushSizeSlider.value()
			# circ = plt.Circle((x, y), radius=brush_radius, color=self.colors[self.curr_label + 1], alpha=.3)
			# self.axes.add_patch(circ)
			# self.circle_list.append(circ)

	# def erase_circles_near(self, x, y):
	# 	to_remove = []
	# 	for c in self.circle_list:
	# 		if abs(c.center[0] - x) < 30 and abs(c.center[1] - y) < 30:
	# 			to_remove.append(c)

	# 	for c in to_remove:
	# 		self.circle_list.remove(c)
	# 		c.remove()

	# def circle_list_to_labeling_field(self, circle_list):
	# 	label_circles = []
	# 	for c in circle_list:
	# 		label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
	# 		label_circles.append((int(c.center[0]), int(c.center[1]), c.radius, label))
	# 	return label_circles


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
			self.axes.add_patch(rect)

		else:
			self.statusBar().showMessage("Remove label of superpixel %d" % sp_ind)
			self.sp_labellist[sp_ind] = -1

			self.sp_rectlist[sp_ind].remove()
			self.sp_rectlist[sp_ind] = None


	def generate_labelmap(self, polygon_list, polygon_labels):
		"""
		Generate labelmap from the list of polygons and the list of polygon labels
		"""

		labelmap_flat = -1 * np.ones((self.dm.image_height * self.dm.image_width, 1), dtype=np.int)

		X, Y = np.mgrid[0:self.dm.image_height, 0:self.dm.image_width]
		all_points = np.column_stack([Y.ravel(), X.ravel()]) # nx2
		for p, l in zip(self.polygon_list, self.polygon_labels):
			labelmap_flat[p.get_path().contains_points(all_points)] = l

		labelmap = labelmap_flat.reshape((self.dm.image_height, self.dm.image_width))

		# for c in zip(polygon_list):
		# 	cx, cy = c.center
		# 	for x in np.arange(cx-c.radius, cx+c.radius):
		# 		for y in np.arange(cy-c.radius, cy+c.radius):
		# 			if (cx-x)**2+(cy-y)**2 <= c.radius**2:
		# 				label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
		# 				labelmap[int(y),int(x)] = label

		return labelmap

	# def generate_labelmap(self, circle_list):
	# 	"""
	# 	Generate labelmap from the list of circles
	# 	"""

	# 	labelmap = -1*np.ones((self.dm.image_height, self.dm.image_width), dtype=np.int)

	# 	for c in circle_list:
	# 		cx, cy = c.center
	# 		for x in np.arange(cx-c.radius, cx+c.radius):
	# 			for y in np.arange(cy-c.radius, cy+c.radius):
	# 				if (cx-x)**2+(cy-y)**2 <= c.radius**2:
	# 					label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
	# 					labelmap[int(y),int(x)] = label

	# 	return labelmap


# if __name__ == '__main__':

# 	import sys
# 	import os
# 	sys.path.append('../notebooks')

# 	from utilities import *

# 	dm = DataManager(data_dir=os.environ['LOCAL_DATA_DIR'], 
# 	    repo_dir=os.environ['LOCAL_REPO_DIR'],
# 	    result_dir=os.environ['LOCAL_RESULT_DIR'], 
# 	    labeling_dir=os.environ['LOCAL_LABELING_DIR'])

# 	class args:
# 		stack_name = 'RS140'
# 		resolution = 'x5'
# 		# slice_ind = int(sys.argv[1])
# 		slice_ind = 0
# 		gabor_params_id = 'blueNisslWide'
# 		segm_params_id = 'blueNisslRegular'
# 		vq_params_id = 'blueNissl'
		
# 	dm.set_image(args.stack_name, args.resolution, args.slice_ind)
# 	dm.set_gabor_params(gabor_params_id=args.gabor_params_id)
# 	dm.set_segmentation_params(segm_params_id=args.segm_params_id)
# 	dm.set_vq_params(vq_params_id=args.vq_params_id)
	
# 	gui = BrainLabelingGUI(dm=dm)
#  #    # gui.show()
# 	gui.app.exec_()