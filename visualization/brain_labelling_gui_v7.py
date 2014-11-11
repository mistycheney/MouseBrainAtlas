
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
from matplotlib.patches import Rectangle

from skimage.color import label2rgb
from random import random

import subprocess

import time
import datetime
import cv2
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

from ui_DataManager import Ui_DataManager
from ui_BrainLabelingGui_v7 import Ui_BrainLabelingGui
# from ui_InputSelectionMultipleLists import Ui_InputSelectionDialog


class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
    def __init__(self, parent=None):
        """
        Initialization of BrainLabelingGUI.
        """
        # Load data

        self.data_dir = '/home/yuncong/BrainLocal/DavidData_v3'
        self.remote_data_dir = '/home/yuncong/project/DavidData2014v3'
        self.gordon_username = 'yuncong'
        self.gordon_hostname = 'gcn-20-32.sdsc.edu'
        self.temp_dir = '/home/yuncong/BrainLocal'
        self.remote_repo_dir = '/home/yuncong/Brain'
        self.params_dir = '../params'

        self.image_name = None
        self.instance_dir = None
        self.instance_name = None

        self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)


        self.parent_labeling_name = None

        self.stack_name = 'RS141'
        self.resolution = 'x5'
        self.slice_id = '0001'
        self.image_name = 'RS141_x5_0001'

        self.username = 'yuncong'

        self.load_data()
        # self.data_manager.close()
        self.initialize_brain_labeling_gui()


    def load_data(self):

        # img = cv2.imread(self._full_object_name('img', 'tif'), 0)
        # self.mask = np.load(self._full_object_name('cropMask', 'npy'))
        
        img_path = os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, self.image_name+'.tif')
        mask_path = os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, self.image_name + '_mask.png')


        self.img = cv2.imread('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/filterResults/RS141_x5_0001_gabor-blueNisslWide_cropImg.tif')

        self.mask = np.load('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/filterResults/RS141_x5_0001_gabor-blueNisslWide_cropMask.npy')

        # self.mask = cv2.imread(mask_path, 0) > 0
        # self.mask = np.ones_like(img, dtype=np.bool)
        # self.img = cv2.imread(img_path, 0)
        self.img[~self.mask] = 0

        try:
            labeling_fn = os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, 'labelings', self.image_name + '_' + self.parent_labeling_name + '.pkl')

            parent_labeling = pickle.load(open(labeling_fn, 'r'))

            print 'Load saved labeling'
            
            label_circles = parent_labeling['final_label_circles']

            self.labeling = {
                'username' : self.username,
                'parent_labeling_name' : self.parent_labeling_name,
                'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
                'init_label_circles' : label_circles,
                'final_label_circles' : None,
                'labelnames' : parent_labeling['labelnames'],
            }


        except Exception as e:
            print 'error', e

            print 'No labeling is given. Initialize labeling.'

            self.labeling = {
                'username' : self.username,
                'parent_labeling_name' : None,
                'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
                'init_label_circles' : [],
                'final_label_circles' : None,
                'labelnames' : [],
            }

            labelnames_fn = os.path.join(self.data_dir, 'labelnames.json')
            if os.path.isfile(labelnames_fn):
                labelnames = json.load(open(labelnames_fn, 'r'))
                self.labeling['labelnames'] = labelnames
            else:
                n_models = 10
                self.labeling['labelnames']=['No Label']+['Label %2d'%i for i in range(n_models+1)]                    
        
        
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
        else:
            self.growRegion_Action.setEnabled(False)

        pos = self.cursor().pos()

        action = self.menu.exec_(pos)
        
        if action == self.growRegion_Action:
            sp_ind = self.segmentation[int(canvas_pos[1]), int(canvas_pos[0])]
            self.grow_region(sp_ind)

    def initialize_brain_labeling_gui(self):


        self.menu = QMenu()
        self.growRegion_Action = self.menu.addAction("Grow region")

        # A set of high-contrast colors proposed by Green-Armytage
        self.colors = np.loadtxt('high_contrast_colors.txt', skiprows=1)/255.
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

        self.display_buttons = [self.img_radioButton, self.imgSeg_radioButton, self.textonmap_radioButton, self.dirmap_radioButton]
        self.img_radioButton.setChecked(True)
        self.seg_enabled = False

        for b in self.display_buttons:
            b.toggled.connect(self.display_option_changed)

        self.n_labelbuttons = 0
        
        self.labelbuttons = []
        self.labeldescs = []

        for i in range(self.n_labels):
            self._add_labelbutton(desc=self.labeling['labelnames'][i])

        self.loadButton.clicked.connect(self.load_callback)
        self.saveButton.clicked.connect(self.save_callback)
        # self.newLabelButton.clicked.connect(self.newlabel_callback)
        # self.newLabelButton.clicked.connect(self.sigboost_callback)
        self.quitButton.clicked.connect(self.close)

        self.brushSizeSlider.valueChanged.connect(self.brushSizeSlider_valueChanged)
        self.brushSizeEdit.setText('%d' % self.brushSizeSlider.value())

        self.growThreshSlider.valueChanged.connect(self.growThreshSlider_valueChanged)
        self.growThreshEdit.setText('%.2f' % (self.growThreshSlider.value()/100.))


        # help_message = 'Usage: right click to pick a color; left click to assign color to a superpixel; scroll to zoom, drag to move'
        # self.setWindowTitle('%s' %(help_message))

        self.setWindowTitle(self.windowTitle() + ', parent_labeling = %s' %(self.parent_labeling_name))

        # self.statusBar().showMessage()       

        self.fig.clear()
        self.fig.set_facecolor('white')

        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')

        self.axes.imshow(self.img, cmap=plt.cm.Greys_r,aspect='equal')
        
        # labelmap_vis = label2rgb(self.labelmap, image=self.img, colors=self.colors, alpha=0.3, image_alpha=1)
        # self.axes.imshow(labelmap_vis)

        self.circle_list = [plt.Circle((x,y), radius=r, color=self.colors[l+1], alpha=.3) for x,y,r,l in self.labeling['init_label_circles']]
        self.labelmap = self.generate_labelmap(self.circle_list)

        np.save('/tmp/labelmap.npy', self.labelmap)

        for c in self.circle_list:
            self.axes.add_patch(c)

        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)


        self.newxmin, self.newxmax = self.axes.get_xlim()
        self.newymin, self.newymax = self.axes.get_ylim()


        self.canvas.draw()
        self.show()


    ############################################
    # QT button CALLBACKs
    ############################################


    def growThreshSlider_valueChanged(self):
        self.growThreshEdit.setText('%.2f' % (self.growThreshSlider.value()/100.))

    def brushSizeSlider_valueChanged(self, new_val):
        self.brushSizeEdit.setText('%d' % new_val)

    def grow_region(self, seed_sp):
        from grow_regions_module import grow_cluster
        self.statusBar().showMessage('Grow region from superpixel %d' % seed_sp )

        self.curr_label = self.sp_labellist[seed_sp]

        self.model_fit_reduce_limit = self.growThreshSlider.value()/100.
        curr_cluster = grow_cluster(seed_sp, self.neighbors, self.texton_hists, self.D_sp_null, model_fit_reduce_limit=self.model_fit_reduce_limit)

        for sp in curr_cluster:
            self.paint_superpixel(sp)

        print curr_cluster


    def load_segmentation(self):

        self.segmentation = np.load('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/segmResults/RS141_x5_0001_gabor-blueNisslWide-segm-blueNissl_cropSegmentation.npy')
        self.n_superpixels = len(np.unique(self.segmentation)) - 1
        
        self.sp_labellist = -1*np.ones((self.n_superpixels, ), dtype=np.int)
        self.sp_rectlist = [None for _ in range(self.n_superpixels)]

        self.neighbors = np.load('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/segmResults/RS141_x5_0001_gabor-blueNisslWide-segm-blueNissl_neighbors.npy')
        self.texton_hists = np.load('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/histResults/RS141_x5_0001_gabor-blueNisslWide-segm-blueNissl-vq-blueNissl_texHist.npy')
        self.textonmap = np.load('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/vqResults/RS141_x5_0001_gabor-blueNisslWide-vq-blueNissl_texMap.npy')

        from scipy.spatial.distance import cdist

        self.n_texton = 100
        overall_texton_hist = np.bincount(self.textonmap[self.mask].flat, minlength=self.n_texton)
        self.overall_texton_hist_normalized = overall_texton_hist.astype(np.float) / overall_texton_hist.sum()
        self.D_sp_null = np.squeeze(cdist(self.texton_hists, [self.overall_texton_hist_normalized], chi2))

        self.seg_enabled = True

    def display_option_changed(self, checked):
        if checked:

            self.axes.clear()
            self.axes.axis('off')

            if self.sender() == self.imgSeg_radioButton:
                self.seg_vis = cv2.imread('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/segmResults/RS141_x5_0001_gabor-blueNisslWide-segm-blueNissl_cropSegmentation.tif')[:,:,::-1]
                self.seg_vis[~self.mask] = 0

                self.axes.imshow(self.seg_vis, aspect='equal')

                self.load_segmentation()

            elif self.sender() == self.img_radioButton:
                self.axes.imshow(self.img, aspect='equal')
                self.seg_enabled = False

            elif self.sender() == self.textonmap_radioButton:
                self.textonmap_vis = cv2.imread('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/vqResults/RS141_x5_0001_gabor-blueNisslWide-vq-blueNissl_texMap.tif')[:,:,::-1]
                self.textonmap_vis[~self.mask] = 0
                
                self.axes.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
                self.seg_enabled = False

            elif self.sender() == self.dirmap_radioButton:
                dirmap_vis = cv2.imread('/home/yuncong/BrainLocal/DavidData_v4/RS141/x5/0001/segmResults/RS141_x5_0001_gabor-blueNisslWide-segm-blueNissl_dirMap.tif')[:,:,::-1]

                dirmap_vis[~self.mask] = 0

                # from skimage import color, img_as_float

                # alpha = 0.6

                # img = img_as_float(self.img)
                # dirmap_vis = img_as_float(dirmap_vis)

                # img_hsv = color.rgb2hsv(img)
                # color_mask_hsv = color.rgb2hsv(dirmap_vis)
                # img_hsv[..., 0] = color_mask_hsv[..., 0]
                # img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
                
                # img_masked = color.hsv2rgb(img_hsv)



                self.axes.imshow(dirmap_vis, aspect='equal')

                self.load_segmentation()

            self.axes.set_xlim([self.newxmin, self.newxmax])
            self.axes.set_ylim([self.newymin, self.newymax])

            self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            self.canvas.draw()


    def _add_labelbutton(self, desc=None):
        self.n_labelbuttons += 1

        label = self.n_labelbuttons - 2

        row = (label + 1) % 5
        col = (label + 1) / 5

        btn = QPushButton('%d' % label, self)
        edt = QLineEdit(QString(desc if desc is not None else 'Label %d' % label))

        self.labelbuttons.append(btn)
        self.labeldescs.append(edt)

        btn.clicked.connect(self.labelbutton_callback)

        r, g, b, a = self.label_cmap(label + 1)

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

    def _save_labeling(self):

        # self.axes.imshow(labelmap_vis)
        # for c in self.circle_list:
        #     c.remove()

        # self.circle_list = []

        # self.canvas.draw()

        self.labeling['final_label_circles'] = self.circle_list_to_labeling_field(self.circle_list)
        self.labeling['logout_time'] = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
        self.labeling['labelnames'] = [str(edt.text()) for edt in self.labeldescs]

        labelnames_fn = os.path.join(self.data_dir, 'labelnames.json')

        json.dump(self.labeling['labelnames'], open(labelnames_fn, 'w'))

        new_labeling_name = self.username + '_' + self.labeling['logout_time']

        new_labeling_fn = os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, 'labelings', self.image_name + '_'+new_labeling_name+'.pkl')
        pickle.dump(self.labeling, open(new_labeling_fn, 'w'))
        print 'Labeling saved to', new_labeling_fn

        new_preview_fn = os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, 'labelings', self.image_name + '_'+new_labeling_name + '_preview.png')

        self.labelmap = self.generate_labelmap(self.circle_list)
        # labelmap_vis = self.colors[self.labelmap]

        labelmap_vis = label2rgb(self.labelmap, image=self.img, colors=self.colors[1:], bg_label=-1, bg_color=self.colors[0], alpha=0.3, image_alpha=1.)
        
        from skimage import img_as_ubyte
        cv2.imwrite(new_preview_fn, img_as_ubyte(labelmap_vis[:,:,::-1]))

        print 'Preview saved to', new_preview_fn

        self.statusBar().showMessage('Labeling saved to %s' % new_labeling_fn )


    def save_callback(self):
        self._save_labeling()


    def labelbutton_callback(self):
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

            if event.button == 1: # left click: draw
                if self.curr_label is None:
                    self.statusBar().showMessage('No label is selected')
                else:
                    
                    if self.seg_enabled:

                        sp_ind = self.segmentation[int(event.ydata), int(event.xdata)]

                        if sp_ind == -1:
                            self.statusBar().showMessage('This is the background')
                        else:
                            self.statusBar().showMessage('Labeling superpixel %d using %d (%s)' % (sp_ind, self.curr_label, self.labeling['labelnames'][self.curr_label + 1]))                        
                            self.paint_superpixel(sp_ind)

                    else:
                        self.paint_circle(event.xdata, event.ydata)
                        self.statusBar().showMessage('Labeling using %d (%s)' % (self.curr_label, self.labeling['labelnames'][self.curr_label + 1]))

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
        self.statusBar().showMessage('Picked label %d (%s)' % (self.curr_label, self.labeling['labelnames'][self.curr_label + 1]))

    def paint_circle(self, x, y):
        if self.curr_label is None:
            self.statusBar().showMessage('No label is selected')
        else:
            brush_radius = self.brushSizeSlider.value()
            circ = plt.Circle((x, y), radius=brush_radius, color=self.colors[self.curr_label + 1], alpha=.3)
            self.axes.add_patch(circ)
            self.circle_list.append(circ)

    def erase_circles_near(self, x, y):
        to_remove = []
        for c in self.circle_list:
            if abs(c.center[0] - x) < 30 and abs(c.center[1] - y) < 30:
                to_remove.append(c)

        for c in to_remove:
            self.circle_list.remove(c)
            c.remove()

    def circle_list_to_labeling_field(self, circle_list):
        label_circles = []
        for c in circle_list:
            label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
            label_circles.append((int(c.center[0]), int(c.center[1]), c.radius, label))
        return label_circles


    def paint_superpixel(self, sp_ind):

        if self.curr_label == self.sp_labellist[sp_ind]:

            self.statusBar().showMessage('Superpixel already has the selected label')

        elif self.curr_label != -1:

            self.sp_labellist[sp_ind] = self.curr_label
            # self.labelmap = self.sp_labellist[self.segmentation]

            ### Removes previous color to prevent a blending of two or more patches ###
            if self.sp_rectlist[sp_ind] is not None:
                self.sp_rectlist[sp_ind].remove()

            # approximate the superpixel polygon with a square
            ys, xs = np.nonzero(self.segmentation == sp_ind)
            xmin = xs.min()
            ymin = ys.min()

            height = ys.max() - ys.min()
            width = xs.max() - xs.min()

            rect = Rectangle((xmin, ymin), width, height, ec="none", alpha=.3, color=self.colors[self.curr_label + 1])

            self.sp_rectlist[sp_ind] = rect
            self.axes.add_patch(rect)

        else:
            self.statusBar().showMessage("Remove label of superpixel %d" % sp_ind)
            self.sp_labellist[sp_ind] = -1

            self.sp_rectlist[sp_ind].remove()
            self.sp_rectlist[sp_ind] = None


    def generate_labelmap(self, circle_list):
        """
        Generate labelmap from the list of circles
        """

        labelmap = -1*np.ones_like(self.img, dtype=np.int)

        for c in circle_list:
            cx, cy = c.center
            for x in np.arange(cx-c.radius, cx+c.radius):
                for y in np.arange(cy-c.radius, cy+c.radius):
                    if (cx-x)**2+(cy-y)**2 <= c.radius**2:
                        label = np.where(np.all(self.colors == c.get_facecolor()[:3], axis=1))[0][0] - 1
                        labelmap[int(y),int(x)] = label

        return labelmap


if __name__ == '__main__':
    gui = BrainLabelingGUI()
    # gui.show()
    gui.app.exec_()

