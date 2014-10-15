
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
import utilities
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
from ui_BrainLabelingGui import Ui_BrainLabelingGui
# from ui_InputSelectionMultipleLists import Ui_InputSelectionDialog

def get_vispaths(file_paths):
    """
    Return the paths for visualization purposes
    """

    vis_paths = file_paths[:]
    for i in range(len(file_paths)):
        p = file_paths[i]
        if 'stages' in p or 'pipelineResults' in p or p.endswith('.tif'):
            vis_paths[i] = None
        elif 'labeling' in p:
            elements = p.split('/')
            n_elem = len(elements)
            if n_elem == 5:
                vis_paths[i] = None
            elif n_elem == 6:
                if elements[5].endswith('.pkl'):
                    labeling_name = '_'.join(elements[5][:-4].split('_')[-2:])
                    vis_paths[i] = '/'.join(elements[:4] + [labeling_name])
                else:
                    vis_paths[i] = None
    return vis_paths


class Status(object):
    NO_LABEL, INSTANCE_NOT_PROCESSED, INSTANCE_PROCESSED_NOT_DOWNLOADED, INSTANCE_READY, \
    IMG_NOT_DOWNLOADED, IMG_READY, LABELING_READY, LABELING_NOT_DOWNLOADED, LABELING_READY_NOT_UPLOADED, ACTION = range(10)

status_text = ['NO_LABEL', 'INSTANCE_NOT_PROCESSED', 'INSTANCE_PROCESSED_NOT_DOWNLOADED', \
'INSTANCE_READY', 'IMG_NOT_DOWNLOADED', 'IMG_READY', 'LABELING_READY', 'LABELING_NOT_DOWNLOADED', \
'LABELING_READY_NOT_UPLOADED', 'ACTION']


class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
    def __init__(self, parent=None):
        """
        Initialization of BrainLabelingGUI.
        """
        # Load data

        self.data_dir = '/home/yuncong/BrainLocal/DavidData'
        self.remote_data_dir = '/home/yuncong/DavidData/'
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

        self.initialize_data_manager()


    def label_paths(self):

        self._download_remote_directory_structure()

        remote_dir_structure = pickle.load(open('%s/remote_directory_structure.pkl'%self.temp_dir, 'r')).values()[0]
        local_dir_structure = get_directory_structure(self.data_dir).values()[0]

        remote_paths = dict_to_paths(remote_dir_structure)
        remote_vispaths = get_vispaths(remote_paths)

        local_paths = dict_to_paths(local_dir_structure)
        local_vispaths = get_vispaths(local_paths)

        params_list = [fn[6:-5] for fn in os.listdir(self.params_dir)]

        complete_paths1 = pad_paths(remote_paths, level=3, key_list=params_list)
        complete_paths2 = pad_paths(local_paths, level=3, key_list=params_list)
        complete_paths = list(set(complete_paths1) | set(complete_paths2))

        # pprint(complete_paths)

        complete_vispaths = get_vispaths(complete_paths)

        labeled_complete_vispaths = dict([(p, Status.NO_LABEL) for p in complete_vispaths])

        for p in complete_vispaths:
            if p is None: continue
            nlevel = len(p.split('/'))
            if nlevel == 5: # labeling path
                if p in local_vispaths:
                    labeled_complete_vispaths[p] = Status.LABELING_READY
                    if p not in remote_vispaths:
                        labeled_complete_vispaths[p] = Status.LABELING_READY_NOT_UPLOADED
                elif p in remote_vispaths:
                    labeled_complete_vispaths[p] = Status.LABELING_NOT_DOWNLOADED
            if nlevel == 4: # params path
                if p in local_vispaths:
                    labeled_complete_vispaths[p] = Status.INSTANCE_READY
                elif p in remote_vispaths:
                    labeled_complete_vispaths[p] = Status.INSTANCE_PROCESSED_NOT_DOWNLOADED
                else:
                    labeled_complete_vispaths[p] = Status.INSTANCE_NOT_PROCESSED
            elif nlevel == 3: # image path
                if p in local_vispaths:
                    labeled_complete_vispaths[p] = Status.IMG_READY
                elif p in remote_vispaths:
                    labeled_complete_vispaths[p] = Status.IMG_NOT_DOWNLOADED

        status_labels = [labeled_complete_vispaths[p] for p in complete_vispaths]                

        for p in complete_vispaths[:]:
            if p is None: continue
            nlevel = len(p.split('/'))
            if nlevel == 4:
                if labeled_complete_vispaths[p] == Status.INSTANCE_READY:
                    complete_vispaths.append(p + '/' + 'empty_labeling')
                    complete_paths.append(None)
                    status_labels.append(Status.ACTION)

        return complete_paths, complete_vispaths, status_labels


    def _full_labeling_name(self, labeling_name, ext):
        return os.path.join(self.instance_dir, 'labelings', self.instance_name + '_' + labeling_name + '.' + ext)

    def _full_object_name(self, obj_name, ext):
       return os.path.join(self.instance_dir, 'pipelineResults', self.instance_name + '_' + obj_name + '.' + ext)


    def on_select_item(self, index):


        item = self.data_model.itemFromIndex(index)
        
        fullpath_list = []
        while item is not None:
            fullpath_list.append(str(item.text()))
            item = item.parent()
        fullpath_list = fullpath_list[::-1]

        self.full_vispath = '/'.join(fullpath_list)
        self.status = self.vispaths_status_dict[self.full_vispath]

        self.data_manager.statusBar().showMessage(status_text[self.status])

        self.data_manager_ui.inputLoadButton.setText('None')
        self.data_manager_ui.uploadButton.setText('None')

        if self.status == Status.LABELING_READY or self.status == Status.LABELING_READY_NOT_UPLOADED or self.status == Status.ACTION:
            self.data_manager_ui.inputLoadButton.setText('Load')
            if self.status == Status.LABELING_READY_NOT_UPLOADED:
                self.data_manager_ui.uploadButton.setText('Upload Labeling')

        elif self.status == Status.INSTANCE_PROCESSED_NOT_DOWNLOADED or self.status == Status.IMG_NOT_DOWNLOADED or self.status == Status.LABELING_NOT_DOWNLOADED:
            self.data_manager_ui.inputLoadButton.setText('Download')
        elif self.status == Status.INSTANCE_NOT_PROCESSED:
            self.data_manager_ui.inputLoadButton.setText('Process')

        if len(fullpath_list) == 3: # select slice number
                self.stack_name, self.resolution, self.slice_id = fullpath_list

                self.image_name = '_'.join([self.stack_name, self.resolution, self.slice_id])
                self.instance_dir = None
                self.instance_name = None

                preview_img = QPixmap(os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, self.image_name + '.tif'))

                self.data_manager_ui.preview_pic.setGeometry(0, 0, 500, 500)
                self.data_manager_ui.preview_pic.setPixmap(preview_img.scaled(self.data_manager_ui.preview_pic.size(), Qt.KeepAspectRatio))

        elif len(fullpath_list) == 5:
            self.stack_name, self.resolution, self.slice_id, self.params_name, self.parent_labeling_name = fullpath_list

            self.image_name = '_'.join([self.stack_name, self.resolution, self.slice_id])
            self.instance_name = self.image_name + '_' + self.params_name
            self.instance_dir = os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, self.params_name)

            self.username = str(self.data_manager_ui.usernameEdit.text())

            preview_fn = self._full_labeling_name(self.parent_labeling_name + '_preview', 'png')
            if os.path.isfile(preview_fn):

                preview_img = QPixmap(self._full_labeling_name(self.parent_labeling_name + '_preview', 'png'))

                self.data_manager_ui.preview_pic.setGeometry(0, 0, 500, 500)
                self.data_manager_ui.preview_pic.setPixmap(preview_img.scaled(self.data_manager_ui.preview_pic.size(), Qt.KeepAspectRatio))
            else:
                self.data_manager_ui.preview_pic.clear()
        else:
            self.data_manager_ui.preview_pic.clear()


    def on_inputLoadButton(self):

        if self.status == Status.LABELING_READY or self.status == Status.LABELING_READY_NOT_UPLOADED or self.status == Status.ACTION:
            # load brain labeling gui

            if self.status == Status.ACTION:
                self.parent_labeling_name = None

            self.load_data()
            # self.data_manager.close()
            self.initialize_brain_labeling_gui()

        elif self.status == Status.INSTANCE_PROCESSED_NOT_DOWNLOADED:
            filepath = self.vispaths_filepaths_dict[self.full_vispath]
            remote_dir = os.path.join(self.remote_data_dir, filepath)
            local_dir = os.path.dirname(os.path.join(self.data_dir, filepath))
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            cmd = "scp -r %s@%s:%s %s" % (self.gordon_username, self.gordon_hostname, remote_dir, local_dir)
            print cmd
            # subprocess.call(cmd, shell=True)
            # self.refresh_data_status()

        elif self.status == Status.IMG_NOT_DOWNLOADED:
            filepath = self.vispaths_filepaths_dict[self.full_vispath]
            remote_fn = os.path.join(self.remote_data_dir, filepath, '_'.join(filepath.split(os.sep))+'.tif')
            local_dir = os.path.join(self.data_dir, filepath)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            cmd = "scp %s@%s:%s %s" % (self.gordon_username, self.gordon_hostname, remote_fn, local_dir)
            print cmd
            subprocess.call(cmd, shell=True)
            self.refresh_data_status()

        elif self.status == Status.LABELING_NOT_DOWNLOADED:
            filepath = self.vispaths_filepaths_dict[self.full_vispath]
            remote_fn = os.path.join(self.remote_data_dir, filepath + '.pkl')
            local_dir = os.path.dirname(os.path.join(self.data_dir, filepath))
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            cmd = "scp %s@%s:%s %s" % (self.gordon_username, self.gordon_hostname, remote_fn, local_dir)
            print cmd
            subprocess.call(cmd, shell=True)
            self.refresh_data_status()


    def _download_remote_directory_structure(self):
        cmd = "ssh %s@%s 'python %s/utility_scripts/get_directory_structure.py %s'"%(self.gordon_username, self.gordon_hostname, self.remote_repo_dir, self.remote_data_dir)
        print cmd
        subprocess.call(cmd, shell=True)

        cmd = "scp -r %s@%s:%s/utility_scripts/remote_directory_structure.pkl %s"%(self.gordon_username, self.gordon_hostname, self.remote_repo_dir, self.temp_dir)
        print cmd
        subprocess.call(cmd, shell=True)


    def on_DataManager_getRemoteButton(self):
        self._download_remote_directory_structure()
        self.refresh_data_status()


    def on_DataManager_uploadButton(self):
        if self.status == Status.LABELING_READY_NOT_UPLOADED:

            labeling_filepath = self.vispaths_filepaths_dict[self.full_vispath]
            remote_dir = os.path.join(self.remote_data_dir, os.path.dirname(labeling_filepath))
            
            labeling_wildcard = os.path.join(self.data_dir, labeling_filepath+'*')

            cmd = "scp %s %s@%s:%s" %(labeling_wildcard, self.gordon_username, self.gordon_hostname, remote_dir)
            print cmd
            subprocess.call(cmd, shell=True)

            labelnames_fn = os.path.join(self.data_dir, os.path.dirname(labeling_filepath), '*labelnames*')
            
            cmd = "scp %s %s@%s:%s" %(labelnames_fn, self.gordon_username, self.gordon_hostname, remote_dir)
            print cmd
            subprocess.call(cmd, shell=True)


    def load_data(self):

        self.segmentation = np.load(self._full_object_name('segmentation', 'npy'))
        self.n_superpixels = max(self.segmentation.flatten()) + 1
        self.img = cv2.imread(self._full_object_name('segmentation', 'tif'), 0)

        try:

            labeling = pickle.load(open(self._full_labeling_name(self.parent_labeling_name, 'pkl'), 'r'))
            
            self.labellist = labeling['final_labellist']

            self.labeling = {
                'username' : self.username,
                'parent_labeling_name' : self.parent_labeling_name,
                'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
                'init_labellist' : self.labellist,
                'final_labellist' : None,
                'labelnames' : labeling['labelnames'],
                'history' : []
            }
            
            self.n_models = max(10,np.max(self.labellist)+1)
        
            print 'Loading saved labeling'
            self.n_labels = len(self.labeling['labelnames'])

        except:

            print 'No labeling is given. Initialize labeling.'

            self.labellist = -1 * np.ones(self.n_superpixels, dtype=np.int)
            self.labeling = {
                'username' : self.username,
                'parent_labeling_name' : None,
                'login_time' : datetime.datetime.now().strftime("%m%d%Y%H%M%S"),
                'init_labellist' : self.labellist,
                'final_labellist' : None,
                'labelnames' : [],
                'history' : []
            }

            self.n_models = 10

            # Retrieves previous label names

            labelnames_fn = self._full_labeling_name('labelnames', 'json')
            if os.path.isfile(labelnames_fn):
                labelnames = json.load(open(labelnames_fn, 'r'))
                self.labeling['labelnames'] = labelnames
            else:
                self.labeling['labelnames']=['No Label']+['Label %2d'%i for i in range(self.n_models+1)]                    
        
            self.n_labels = len(self.labeling['labelnames'])

        self.labelmap = -1 * np.ones_like(self.segmentation, dtype=np.int)

        if np.max(self.labellist) > 0:
            self.labelmap = self.labellist[self.segmentation]
            print "labelmap updated"

        # A set of high-contrast colors proposed by Green-Armytage
        self.colors = np.loadtxt('high_contrast_colors.txt', skiprows=1)
        
        for i in range(len(self.colors)):
            self.colors[i]=tuple([float(c)/255.0 for c in self.colors[i]])
        
        self.label_cmap = ListedColormap(self.colors, name='label_cmap')

        # initialize GUI variables
        self.paint_label = -1        # color of pen
        self.pick_mode = False       # True while you hold ctrl; used to pick a color from the image
        self.press = False           # related to pan (press and drag) vs. select (click)
        self.base_scale = 1.05       # multiplication factor for zoom using scroll wheel
        self.moved = False           # indicates whether mouse has moved while left button is pressed


    def refresh_data_status(self):
        paths, vis_paths, labels = self.label_paths()
        
        self.vispaths_status_dict = dict([(p, l) for p, l in zip(vis_paths, labels) if p is not None])
        self.vispaths_filepaths_dict = dict([(vp, fp) for vp, fp in zip(vis_paths, paths) if vp is not None])

        self.data_model = paths_to_QStandardModel([p for p in vis_paths if p is not None])
        self.data_manager_ui.StackSliceView.setModel(self.data_model)

    def initialize_data_manager(self):

        self.data_manager = QMainWindow()
        self.data_manager_ui = Ui_DataManager()
        self.data_manager_ui.setupUi(self.data_manager)

        self.refresh_data_status()

        self.data_manager_ui.StackSliceView.clicked.connect(self.on_select_item)

        self.data_manager_ui.StackSliceView.setColumnWidths([10,10,10,10])

        self.data_manager_ui.inputLoadButton.clicked.connect(self.on_inputLoadButton)
        self.data_manager_ui.uploadButton.clicked.connect(self.on_DataManager_uploadButton)
        self.data_manager_ui.getRemoteButton.clicked.connect(self.on_DataManager_getRemoteButton)

        self.data_manager.show()


    def initialize_brain_labeling_gui(self):

        self.setupUi(self)

        self.fig = self.canvaswidget.fig
        self.canvas = self.canvaswidget.canvas

        self.canvas.mpl_connect('scroll_event', self.zoom_fun)
        self.canvas.mpl_connect('button_press_event', self.press_fun)
        self.canvas.mpl_connect('button_release_event', self.release_fun)
        self.canvas.mpl_connect('motion_notify_event', self.motion_fun)
        
        self.n_labelbuttons = 0
        
        self.labelbuttons = []
        self.labeldescs = []

        for i in range(self.n_labels):
            self._add_labelbutton(desc=self.labeling['labelnames'][i])

        self.loadButton.clicked.connect(self.load_callback)
        self.saveButton.clicked.connect(self.save_callback)
        self.newLabelButton.clicked.connect(self.newlabel_callback)
        self.newLabelButton.clicked.connect(self.sigboost_callback)
        self.quitButton.clicked.connect(self.close)

        # help_message = 'Usage: right click to pick a color; left click to assign color to a superpixel; scroll to zoom, drag to move'
        # self.setWindowTitle('%s' %(help_message))

        self.setWindowTitle(self.windowTitle() + ', parent_labeling = %s' %(self.parent_labeling_name))

        # self.statusBar().showMessage()       

        self.fig.clear()
        self.fig.set_facecolor('white')

        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')

        self.axes.imshow(self.img, cmap=plt.cm.Greys_r,aspect='equal')
        self.label_layer=None  # to avoid removing layer when it is not yet there
        
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1) 
    

        self.rect_list = [None for _ in range(len(self.labellist))]

        for i,value in enumerate(self.labellist):
            if value != -1:
                ys, xs = np.nonzero(self.segmentation == i)
                xmin = xs.min()
                ymin = ys.min()

                height = ys.max() - ys.min()
                width = xs.max() - xs.min()
                rect = Rectangle((xmin, ymin), width, height, ec="none", alpha=.2, color=self.colors[int(self.labellist[i])+1])
                self.rect_list[i] = rect
                self.axes.add_patch(rect)

        self.canvas.draw()
        self.show()


    ############################################
    # QT button CALLBACKs
    ############################################

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
        # TODO: 
        # - upload this labeling
        # - run sigboost with this labeling
        # - download resulting labeling
        # - load the resulting labeling


    def load_callback(self):
        self.initialize_data_manager()

    def _save_labeling(self):
        self.labeling['final_labellist'] = self.labellist
        self.labeling['logout_time'] = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
        self.labeling['labelnames'] = [str(edt.text()) for edt in self.labeldescs]

        json.dump(self.labeling['labelnames'], open(self._full_labeling_name('labelnames','json'), 'w'))

        new_labeling_name = self.username + '_' + self.labeling['logout_time']
        new_labeling_fn = self._full_labeling_name(new_labeling_name, 'pkl')
        pickle.dump(self.labeling, open(new_labeling_fn, 'w'))
        print 'Labeling saved to', new_labeling_fn

        new_preview_fn = self._full_labeling_name(new_labeling_name + '_preview', 'png')

        # plt.sca(self.axes)
        # plt.tight_layout()
        # plt.savefig(new_preview_fn, bbox_inches='tight')

        img = utilities.load_image('cropImg', instance_name=self.instance_name, results_dir=os.path.join(self.instance_dir, 'pipelineResults'))

        labelmap_rgb = label2rgb(self.labelmap.astype(np.int), image=img, colors=self.colors[1:], alpha=.3, 
                         image_alpha=.8, bg_color=self.colors[0])
        labelmap_rgb = utilities.regulate_img(labelmap_rgb)
        cv2.imwrite(new_preview_fn, labelmap_rgb)

        print 'Preview saved to', new_preview_fn


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
        
	# This makes sure the subplot properly expands to figure window    
     #   	if cur_pos.x0 <= .01 and cur_pos.y0 <=.01: 
    	#     newxmin = xdata - left*scale_factor
    	#     newxmax = xdata + right*scale_factor
    	#     newymin = ydata - up*scale_factor
     #        newymax = ydata + down*scale_factor
    	# elif cur_pos.x0 >.05 and cur_pos.y0 >.05:
     #        newxmin = xdata - left
     #        newxmax = xdata + right
     #        newymin = ydata - up
     #        newymax = ydata + down
    	# elif cur_pos.y0 <=.01:
     #        newxmin = xdata - left
     #        newxmax = xdata + right
     #        newymin = ydata - up*scale_factor
     #        newymax = ydata + down*scale_factor
    	# elif cur_pos.x0 <=.01:
     #        newxmin = xdata - left*scale_factor
     #        newxmax = xdata + right*scale_factor
     #        newymin = ydata - up
     #        newymax = ydata + down
    	# else:
        newxmin = xdata - left*scale_factor
        newxmax = xdata + right*scale_factor
        newymin = ydata - up*scale_factor
        newymax = ydata + down*scale_factor

    	# set new limits
        self.axes.set_xlim([newxmin, newxmax])
        self.axes.set_ylim([newymin, newymax])

        self.canvas.draw() # force re-draw

    def press_fun(self, event):
        self.press_x = event.xdata
        self.press_y = event.ydata
        self.press = True
        self.press_time = time.time()

    def motion_fun(self, event):
        	
        if self.press and time.time() - self.press_time > .5:
            cur_xlim = self.axes.get_xlim()
            cur_ylim = self.axes.get_ylim()
            
            if (event.xdata==None) | (event.ydata==None):
                #print 'one of event.xdata or event.ydata is None'
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
            # self.axes.clear()
            try:
                selected_sp = self.segmentation[int(event.ydata), int(event.xdata)]
                selected_label = self.labelmap[int(event.ydata), int(event.xdata)]
            except:
                return

            if event.button == 3: # right click
                # Picking a color
                self.pick_color(selected_label)
            elif event.button == 1: # left click
                # Painting a color
                self.statusBar().showMessage('Labeled superpixel %d as %d (%s)' % (selected_sp, 
                                            self.paint_label, self.labeling['labelnames'][self.paint_label+1]))
                self.change_superpixel_color(selected_sp)

        self.canvas.draw() # force re-draw

    ############################################
    # other functions
    ############################################

    def pick_color(self, selected_label):

        self.paint_label = selected_label
        self.statusBar().showMessage('Picked label %d (%s)' % (self.paint_label, self.labeling['labelnames'][self.paint_label+1]))

    def change_superpixel_color(self, selected_sp):
        '''
        update the labelmap
        '''
        b = time.time()

        ## This updates the labelmap, labellist, and rect_list. Allows the ability to remove rectangle patches.
        ## Also prevents overlaying multiple or different patches over the same sp.

        # checks to see if you are removing a label or labeling trying to label twice
        if self.paint_label == self.labellist[selected_sp]:

            print "sp is already the selected paint label"

        elif self.paint_label != -1 :

            print 'Labeled sp %d as %d' % (selected_sp, self.paint_label)
            self.labellist[selected_sp] = self.paint_label
            self.labelmap = self.labellist[self.segmentation]

            ### Removes previous color to prevent a blending of two or more patches ###
            if self.rect_list[selected_sp] is not None:
                self.rect_list[selected_sp].remove()

            # approximate the superpixel polygon with a square
            ys, xs = np.nonzero(self.segmentation == selected_sp)
            xmin = xs.min()
            ymin = ys.min()

            height = ys.max() - ys.min()
            width = xs.max() - xs.min()

            rect = Rectangle((xmin, ymin), width, height, ec="none", alpha=.2, color=self.colors[self.paint_label+1])

            self.rect_list[selected_sp] = rect
            self.axes.add_patch(rect)

            self.labeling['history'].append((selected_sp, self.paint_label))

        else:
            print "Removing label of sp %d" % selected_sp
            self.labellist[selected_sp] = -1
            self.labelmap = self.labellist[self.segmentation]

            self.rect_list[selected_sp].remove()
            self.rect_list[selected_sp] = None
	    
            self.labeling['history'].append((selected_sp, self.paint_label))

        print 'update', time.time() - b


if __name__ == '__main__':
    gui = BrainLabelingGUI()
    # gui.show()
    gui.app.exec_()

