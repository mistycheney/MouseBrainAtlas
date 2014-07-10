# from __future__ import print_function

import sys
import os
from PIL import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends import qt4_compat
from skimage.color import label2rgb
from random import random

from matplotlib.colors import ListedColormap, NoNorm, ColorConverter

use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE

if use_pyside:
    print 'Using PySide'
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    print 'Using PyQt4'
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

import argparse
import pickle

class PickByColorsGUI(QMainWindow):
    def __init__(self, parent=None, recalc_callback=None, save_callback=None,\
                img=None, segmentation=None, labeling=None):
        """
        Initialization of PickByColorsGUI:
        parent:       the parent window within which this window is embedded.
        recalc_callback: A callback function which is launched when the user clicks on the "recalculate" button.
        save_callback: A callback function which is launched when the user clicks on the "save" button.
        img:          2D array containing the image.
        segmentation: 2D array containing the segment index for each pixel, generated using the super-pixel alg.
        labeling:    A dictionary containing the mapping of super-pixels to colors (labellist) 
                      and ascii names for the colors.
        """
        QMainWindow.__init__(self, parent)
        
        self.paint_label = None     # color of pen
        self.pick_mode  = False     # True while you hold ctrl; used to pick a color from the image
        self.press = False          # related to pan (press and drag) vs. select (click)
        self.base_scale = 1.3       # multiplication factor for zoom using scroll wheel

        self.segmentation = segmentation

        self.recalc_callback=recalc_callback
        self.save_callback=save_callback

        # convert labelmap (dict) to pixel values
        self.labellist=labeling['labellist']
        print 'shape of labellist',np.shape(self.labellist)
        self.buttonTexts=labeling['oldnames']

        self.labelmap_orig = -1*np.ones_like(segmentation, dtype=np.int)
        self.labelmap_orig = self.labellist[segmentation]
        self.labelmap_new = self.labelmap_orig.copy()

        self.n_models = max(self.labellist)+1
        self.n_labels = self.n_models + 1

        # Colors proposed by Green-Armytage 
        self.colors = [(255,255,255),
                       (240,163,255),(0,117,220),(153,63,0),(76,0,92),(25,25,25),(0,92,49),(43,206,72),
                       (255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),(194,0,136),
                       (0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),
                       (224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(255,80,5)]
        for i in range(len(self.colors)):
            self.colors[i]=tuple([float(c)/255.0 for c in self.colors[i]])
        self.label_cmap = ListedColormap(self.colors, name='label_cmap')
        self.img = img

        self.create_main_frame() # construct the layout
        self.initialize_canvas()  # initialize matplotlib plot


    def create_main_frame(self):
        """
        Declare the window layout and widgets
        """
        self.main_frame = QWidget()

        # matplotlib region (the brain image)
        # TODO: allow it to fill the whole available frame
        self.fig = Figure((10.0, 10.0), dpi=100)
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        # callbacks
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.canvas.mpl_connect('scroll_event', self.zoom_fun)
        self.canvas.mpl_connect('button_press_event', self.press_fun)
        self.canvas.mpl_connect('button_release_event', self.release_fun)
        self.canvas.mpl_connect('motion_notify_event', self.motion_fun)

        ############################################
        # QT widgets definitions and layout 
        ############################################
        # existing color button/decription region
        self.colorButtons = [QPushButton('%d'%(i), self) for i in range(-1, self.n_labels-1)]    # push buttons with colors
        self.descEdits = [QLineEdit(QString(self.buttonTexts[i])) for i in range(self.n_labels)] # corresponding decription fields
        self.newDescEdits=[]

        self.color_box = QGridLayout()
        for i, (btn, edt) in enumerate(zip(self.colorButtons, self.descEdits)):
            r, g, b, a = self.label_cmap(i)
            btn.setStyleSheet("background-color: rgba(%d, %d, %d, 20%%)"%(
                int(r*255), int(g*255), int(b*255)))
            btn.setFixedSize(20, 20)
            self.color_box.addWidget(btn, i, 0)
            self.color_box.addWidget(edt, i, 1)

        # new color button/decription region
        self.newcolor_box = QGridLayout()

        # right box = exisitng color region + new color region
        right_box = QHBoxLayout()
        right_box.addLayout(self.color_box)
        right_box.addLayout(self.newcolor_box)

        # left box = matplotlib canvas + buttons below
        left_box = QVBoxLayout()
        left_box.addWidget(self.canvas)

        buttons_box=QHBoxLayout()
        left_box.addLayout(buttons_box)

        # "add new color" button
        newcolor_button = QPushButton('add new color', self)
        buttons_box.addWidget(newcolor_button)
        # callback
        newcolor_button.clicked.connect(self.newcolor_button_clicked)

        # Recalculate button
        recalc_button = QPushButton('recalculate models', self)
        buttons_box.addWidget(recalc_button)
        recalc_button.clicked.connect(self.recalc_callback)

        # save button: save current labeling and current classifiers.
        save_button = QPushButton('save', self)
        buttons_box.addWidget(save_button)
        save_button.clicked.connect(self.save_callback)

        # quit button
        quit_button = QPushButton('quit', self)
        buttons_box.addWidget(quit_button)
        quit_button.clicked.connect(self.close)

        # overall box = left box + right box
        all_box = QHBoxLayout()
        all_box.addLayout(left_box)
        all_box.addLayout(right_box)

        self.main_frame.setLayout(all_box)
        self.setCentralWidget(self.main_frame)

        help_message = 'Usage: Ctrl + Left Click to pick a color; Left Click to assign color to a superpixel; Scroll to zoom, Left Click + drag to pan'
        self.setWindowTitle('PickByColorsGUI (%s)' % help_message)

        # self.statusBar().showMessage()

        self.show()

    def newcolor_button_clicked(self):
        """
        callback for "add new color"
        """

        self.n_labels += 1
        self.paint_label = self.n_labels - 2

        btn = QPushButton('%d'%self.paint_label, self)
        edt = QLineEdit()

        self.newDescEdits.append(edt)

        r, g, b, a = self.label_cmap(self.paint_label+1)

        btn.setStyleSheet("background-color: rgb(%d, %d, %d)"%(
            int(r*255),int(g*255),int(b*255)))
        btn.setFixedSize(20, 20)

        c = self.newcolor_box.rowCount()
        self.newcolor_box.addWidget(btn, c, 0)
        self.newcolor_box.addWidget(edt, c, 1)

    def initialize_canvas(self):
        """
        Initialize matplotlib canvas widget
        """
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')

        self.axes.imshow(self.img, cmap=plt.cm.Greys_r, aspect='equal')
        self.label_layer = self.axes.imshow(self.labelmap_orig + 1,\
                                            cmap=self.label_cmap, alpha=0.2,
                                            aspect='equal', norm=NoNorm())
        self.canvas.draw()


    # CALLBACKS 

    def on_key_press(self, event):
        if event.key == 'control':
            self.pick_mode = True
            print 'pick mode on'

    def on_key_release(self, event):
        if event.key == 'control':
            self.pick_mode = False
            print 'pick mode off'
        
    def zoom_fun(self, event):
        # get the current x and y limits
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        # set the range
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location

        left = xdata - cur_xlim[0]
        right = cur_xlim[1] - xdata
        up = ydata - cur_ylim[0]
        down = cur_ylim[1] - ydata

        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/self.base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = self.base_scale
            
        # set new limits
        self.axes.set_xlim([xdata - left*scale_factor,
                     xdata + right*scale_factor])
        self.axes.set_ylim([ydata - up*scale_factor,
                     ydata + down*scale_factor])
        self.axes.figure.canvas.draw() # force re-draw

    def press_fun(self, event):
        self.press_x = event.xdata
        self.press_y = event.ydata
        self.press = True

    def motion_fun(self, event):
        self.moved = True
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
            self.axes.figure.canvas.draw()

    def release_fun(self, event):
        """
        The release-button callback is responsible for picking a color or changing a color.
        """
        self.press = False
        
        # click without moving, means selection
        if self.moved: self.moved = False
        if event.xdata == self.press_x and event.ydata == self.press_y:
            # self.axes.clear()

            self.selected_sp = self.segmentation[int(event.ydata), int(event.xdata)]
            self.selected_label = self.labelmap_new[int(event.ydata), int(event.xdata)]
            
            if self.pick_mode:
                # Picking a color
                self.paint_label = self.labelmap_new[int(event.ydata), int(event.xdata)]
            else:
                #Painting a color
                if self.paint_label is None:
                    self.statusBar().showMessage('superpixel %d, selected label = %d'%(self.selected_sp,
                                self.selected_label))
                else:
                    self.labelmap_new[self.segmentation == self.selected_sp] = self.paint_label
                    self.labellist[self.selected_sp]=self.paint_label

            if self.paint_label is not None:
                self.statusBar().showMessage('superpixel %d, selected label = %d, paint label = %d'%(self.selected_sp,
                                self.selected_label,
                                self.paint_label))

            curr_xlim = self.axes.get_xlim()
            curr_ylim = self.axes.get_ylim()

            self.label_layer.remove()

            self.label_layer = self.axes.imshow(self.labelmap_new + 1, cmap=self.label_cmap, alpha=.2,
                    aspect='equal', norm=NoNorm())

            self.axes.set_xlim(curr_xlim)
            self.axes.set_ylim(curr_ylim)
            self.axes.figure.canvas.draw() # force re-draw
    
    def get_labels(self):
        A={}
        A['labellist']=self.labellist
        A['oldnames']=[str(q.text()) for q in self.descEdits]
        A['newnames']=[str(q.text()) for q in self.newDescEdits]
        return A

class Interactive_Labeling:
    def recalc_handler(self,event):
        print 'recalc_handler',event
        print self.main_window.get_labels()

    def save_handler(self,event):
        print 'save_handler'
        state = self.main_window.get_labels()
        print state
        pickle.dump(state,open(self.labellist_filename,'w'))

    def full_name(self,extension):
        return os.path.join(self.data_dir, self.result_name + extension)

    def main(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='GUI for coloring superpixels',
            epilog="""Example:
            python %s PMD1305_region0_reduce2_0244_param_nissl324
            """%(os.path.basename(sys.argv[0]), ))

        parser.add_argument("result_name", type=str, help="name of the result")
        parser.add_argument("-d", "--data_dir", type=str, help="result data directory (default: %(default)s)",\
                            default='.')
        args = parser.parse_args()
        self.result_name=args.result_name
        self.data_dir=args.data_dir
        data_dir = os.path.realpath(args.data_dir)

        # The brain image with superpixel boundaries drawn on it
        self.img_filename = self.full_name('_segmentation.png')

        # a matrix of labels indicating which superpixel a pixel belongs to 
        # each label is an integer from 0 to n_superpixels.
        # -1 means background, 0 to n_superpixel-1 corresponds to each superpixel
        self.seg_filename = self.full_name('_segmentation.npy')

        # a list of labels indicating which model a suerpixel is associated with. 
        # Each label is an integer from -1 to n_models-1.
        # -1 means background, 0 to n_models-1 corresponds to each model
        self.labellist_filename = self.full_name('_labeling.pkl') 

        img = np.array(Image.open(self.img_filename)).mean(axis=-1)
        segmentation = np.load(self.seg_filename)

        self.labeling = pickle.load(open(self.labellist_filename,'r'))

        app = QApplication(sys.argv)
        main_window = PickByColorsGUI(img=img, segmentation=segmentation,\
                                      recalc_callback=self.recalc_handler,\
                                      save_callback=self.save_handler,\
                                      labeling=self.labeling)
        main_window.show()
        self.main_window=main_window
        app.exec_()

if __name__ == "__main__":
    IL=Interactive_Labeling()
    IL.main()
