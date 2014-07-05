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
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

import argparse

class PickByColorsGUI(QMainWindow):
    def __init__(self, parent=None, img=None, segmentation=None, labellist=None, n_models=20):
        QMainWindow.__init__(self, parent)
        
        self.paint_label = None     # color of pen
        self.pick_mode  = False     # True while you hold ctrl; that let's you pick a color from image
        self.press = False          # related to pan (press and drag) vs. select (click)
        self.base_scale = 1.3       # multiplication factor for zoom using scroll wheel

        self.segmentation = segmentation

        # convert labelmap (dict) to pixel values
        self.labelmap_orig = -1*np.ones_like(segmentation, dtype=np.int)
        self.labelmap_orig = labellist[segmentation]
        self.labelmap_new = self.labelmap_orig.copy()

        self.n_models = n_models
        self.n_labels = self.n_models + 1

        self.colors = [(1,1,1)] + [(random(),random(),random()) for i in xrange(30)] # be interactive
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
        self.fig = Figure((10.0, 5.0), dpi=100)
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

        # QT widgets definitions below

        # existing color button/decription region
        self.colorButtons = [QPushButton('%d'%(i), self) for i in range(-1, self.n_labels-1)]    # push buttons with colors
        self.descEdits = [QLineEdit() for i in range(self.n_labels)]                            # corresponding decription fields

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

        # "add new color" button
        newcolor_button = QPushButton('add new color', self)
        left_box.addWidget(newcolor_button)
        # callback
        newcolor_button.clicked.connect(self.newcolor_button_clicked)

        # quit button
        quit_button = QPushButton('quit', self)
        left_box.addWidget(quit_button)
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

        self.axes.imshow(self.img, cmap=plt.cm.Greys_r,
                        aspect='equal')
        self.label_layer = self.axes.imshow(self.labelmap_orig + 1, cmap=self.label_cmap, alpha=0.2,
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
            
            offset_x = self.press_x - event.xdata
            offset_y = self.press_y - event.ydata
            
            self.axes.set_xlim(cur_xlim + offset_x)
            self.axes.set_ylim(cur_ylim + offset_y)
            self.axes.figure.canvas.draw()

    def release_fun(self, event):
        self.press = False
        
        # click without moving, means selection
        if self.moved: self.moved = False
        if event.xdata == self.press_x and event.ydata == self.press_y:
            # self.axes.clear()

            self.selected_sp = self.segmentation[int(event.ydata), int(event.xdata)]
            self.selected_label = self.labelmap_new[int(event.ydata), int(event.xdata)]
            
            if self.pick_mode:
                self.paint_label = self.labelmap_new[int(event.ydata), int(event.xdata)]
            else:
                if self.paint_label is None:
                    self.statusBar().showMessage('superpixel %d, selected label = %d'%(self.selected_sp,
                                self.selected_label))
                else:
                    self.labelmap_new[self.segmentation == self.selected_sp] = self.paint_label

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

def main():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='GUI for coloring superpixels',
    epilog="""Example:
    python %s PMD1305_region0_reduce2_0244_param_nissl324
    """%(os.path.basename(sys.argv[0]), ))

    parser.add_argument("result_name", type=str, help="name of the result")
    parser.add_argument("-d", "--data_dir", type=str, help="result data directory (default: %(default)s)", default='.')
    args = parser.parse_args()

    data_dir = os.path.realpath(args.data_dir)

    # The brain image with superpixel boundaries drawn on it
    img_filename = os.path.join(data_dir, args.result_name + '_segmentation.png')

    # a matrix of labels indicating which superpixel a pixel belongs to 
    # each label is an integer from 0 to n_superpixels.
    # -1 means background, 0 to n_superpixel-1 corresponds to each superpixel
    seg_filename = os.path.join(data_dir, args.result_name + '_segmentation.npy')
    
    # a list of labels indicating which model a suerpixel is associated with. 
    # Each label is an integer from -1 to n_models-1.
    # -1 means background, 0 to n_models-1 corresponds to each model
    labellist_filename = os.path.join(data_dir, args.result_name + '_labellist.npy') 

    img = np.array(Image.open(img_filename)).mean(axis=-1)
    segmentation = np.load(seg_filename)
    labellist = np.load(labellist_filename)

    app = QApplication(sys.argv)
    main_window = PickByColorsGUI(img=img, segmentation=segmentation, 
                                    labellist=labellist, n_models=20)
    main_window.show()
    app.exec_()

if __name__ == "__main__":
    main()
