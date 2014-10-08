
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

import time
import datetime
import cv2
from utilities import *

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

from ui_InputSelectionColumnView import Ui_InputSelectionDialog
from ui_BrainLabelingGui import Ui_BrainLabelingGui
# from ui_InputSelectionMultipleLists import Ui_InputSelectionDialog

def get_local_data_structure(data_dir):
    local_structure = get_directory_structure(data_dir)
    
    local_data = {}
    for stack, stack_content in local_structure.values()[0].iteritems():
        local_data[stack] = {}
        for resol, resol_content in stack_content.iteritems():
            local_data[stack][resol] = {}
            for slice, slice_content in resol_content.iteritems():
                local_data[stack][resol][slice] = {}
                for param, param_content in slice_content.iteritems():
                    if param_content is not None:
                        labelings = ['_'.join(labeling[:-4].split('_')[-2:]) for labeling in param_content['labelings'].keys() 
                                if labeling.endswith('.pkl')]
                        local_data[stack][resol][slice][param] = labelings

    return local_data

class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
    def __init__(self, parent=None):
        """
        Initialization of BrainLabelingGUI.
        """
        # Load data

        self.data_dir = '/home/yuncong/BrainLocal/DavidData'
        self.instance_dir = None
        self.instance_name = None

        self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.initialize_input_selection_dialog()


    def load_data(self):

        self.segmentation = np.load(self._full_object_name('segmentation', 'npy'))
        self.n_superpixels = max(self.segmentation.flatten()) + 1
        self.img = cv2.imread(self._full_object_name('segmentation', 'tif'), 0)

        try:

            print self._full_labeling_name(self.parent_labeling_name, 'pkl')

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

            # # Retrieves previous label names
            # try:
            #     labelnames = json.load(open(self._fullname('labelnames', 'json'), 'r'))
            #     self.labeling['names'] = labelnames.values()
            # except:
            #     self.labeling['names']=['No Label']+['Label %2d'%i for i in range(self.n_models+1)]                    
        

        self.labelmap = -1 * np.ones_like(self.segmentation, dtype=np.int)

        if np.max(self.labellist) > 0:
            self.labelmap = self.labellist[self.segmentation]
            print "labelmap updated"

        # A set of high-contrast colors proposed by Green-Armytage
        self.colors = [(255,255,255),
                       (240,163,255),(0,117,220),(153,63,0),(76,0,92),(25,25,25),(0,92,49),(43,206,72),
                       (255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),(194,0,136),
                       (0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
                       (224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(255,80,5)]
        
        for i in range(len(self.colors)):
            self.colors[i]=tuple([float(c)/255.0 for c in self.colors[i]])
        
        self.label_cmap = ListedColormap(self.colors, name='label_cmap')

        # initialize GUI variables
        self.paint_label = -1        # color of pen
        self.pick_mode = False       # True while you hold ctrl; used to pick a color from the image
        self.press = False           # related to pan (press and drag) vs. select (click)
        self.base_scale = 1.05       # multiplication factor for zoom using scroll wheel
        self.moved = False           # indicates whether mouse has moved while left button is pressed
 

    def _full_labeling_name(self, labeling_name, ext):
        return os.path.join(self.instance_dir, 'labelings', self.instance_name + '_' + labeling_name + '.' + ext)

    def _full_object_name(self, obj_name, ext):
       return os.path.join(self.instance_dir, 'pipelineResults', self.instance_name + '_' + obj_name + '.' + ext)


    def on_select_item(self, index):

        # print self.ui.StackSliceView.columnWidths()

        item = self.local_data_model.itemFromIndex(index)
        
        fullpath = []
        while item is not None:
            fullpath.append(str(item.text()))
            item = item.parent()

        if len(fullpath) == 5:
            self.parent_labeling_name, self.params_name, self.slice_id, self.resolution, self.stack_name = fullpath

            self.instance_name = '_'.join([self.stack_name, self.resolution, self.slice_id, self.params_name])
            self.instance_dir = os.path.join(self.data_dir, self.stack_name, self.resolution, self.slice_id, self.params_name)

            self.username = self.ui.usernameEdit.text()

            preview_fn = self._full_labeling_name(self.parent_labeling_name + '_preview', 'png')
            if os.path.isfile(preview_fn):

                preview_img = QPixmap(self._full_labeling_name(self.parent_labeling_name + '_preview', 'png'))

                self.preview_pic.setGeometry(0, 0, 500, 500)
                self.preview_pic.setPixmap(preview_img.scaled(self.preview_pic.size(), Qt.KeepAspectRatio))
            else:
                self.preview_pic.clear()
        else:
            self.preview_pic.clear()


    def on_inputLoadButton(self):

        self.input_selection_dialog.close()

        self.load_data()

        self.initialize_gui()


    def initialize_input_selection_dialog(self):

        remote_data = None # load from dict file

        local_data = get_local_data_structure(self.data_dir)

        self.input_selection_dialog = QDialog()
        self.ui = Ui_InputSelectionDialog()
        self.ui.setupUi(self.input_selection_dialog)

        self.local_data_model = convert_to_QStandardItemModel(local_data)
        self.ui.StackSliceView.setModel(self.local_data_model)
        self.ui.StackSliceView.updatePreviewWidget.connect(self.on_select_item)

        self.preview_pic = QLabel(self.ui.StackSliceView)

        self.ui.StackSliceView.setPreviewWidget(self.preview_pic)

        self.ui.StackSliceView.setColumnWidths([10,10,10,10, 300, 600])

        self.ui.inputLoadButton.clicked.connect(self.on_inputLoadButton)

        self.input_selection_dialog.show()


    def initialize_gui(self):

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
        self.quitButton.clicked.connect(self.close)

        # help_message = 'Usage: right click to pick a color; left click to assign color to a superpixel; scroll to zoom, drag to move'
        # self.setWindowTitle('%s' %(help_message))

        self.setWindowTitle('parent_labeling = %s' %(self.parent_labeling_name))

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

    def load_callback(self):
        return

    def save_callback(self):
        
        self.labeling['final_labellist'] = self.labellist
        self.labeling['logout_time'] = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
        self.labeling['labelnames'] = [edt.text() for edt in self.labeldescs]

        new_labeling_name = self.username + '_' + self.labeling['logout_time']
        new_labeling_fn = self._full_labeling_name(new_labeling_name, 'pkl')
        pickle.dump(self.labeling, open(new_labeling_fn, 'w'))
        print 'Labeling saved to', new_labeling_fn

        new_preview_fn = self._full_labeling_name(new_labeling_name + '_preview', 'png')

        plt.sca(self.axes)
        plt.tight_layout()
        plt.savefig(new_preview_fn, bbox_inches='tight')

        print 'Preview saved to', new_preview_fn

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
        	
        if self.press:
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
