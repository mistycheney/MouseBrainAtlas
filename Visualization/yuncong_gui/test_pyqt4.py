# from __future__ import print_function

import sys
import os
import cv2

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

output_dir = '/home/yuncong/BrainLocal/output'
img_name = 'PMD1305_region0_reduce2_0244'
result_name = '%s_param10'%img_name

class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        #self.x, self.y = self.get_data()
        self.get_data()
        self.create_main_frame()
        self.on_draw()
        self.paint_label = None
        self.pick_mode  =False
        self.new_labels = []

    def create_main_frame(self):
        self.main_frame = QWidget()

        self.fig = Figure((10.0, 5.0), dpi=100)
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

        # self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.colorButtons = [QPushButton('%d'%(i), self) for i in range(-1,self.n_labels-1)]
        self.descEdits = [QLineEdit() for i in range(self.n_labels)]

        self.color_box = QGridLayout()
        # self.color_box.setSizeConstraint(QLayout.SetFixedSize)

        for i, (btn, edt) in enumerate(zip(self.colorButtons, self.descEdits)):

            r, g, b, a = self.label_cmap(i)
            btn.setStyleSheet("background-color: rgba(%d, %d, %d, 20%%)"%(
                int(r*255), int(g*255), int(b*255)))
            btn.setFixedSize(20, 20)
            self.color_box.addWidget(btn, i, 0)
            self.color_box.addWidget(edt, i, 1)

        self.newcolor_box = QGridLayout()

        right_box = QHBoxLayout()
        right_box.addLayout(self.color_box)
        right_box.addLayout(self.newcolor_box)

        left_box = QVBoxLayout()
        left_box.addWidget(self.canvas)
        newcolor_button = QPushButton('add new color', self)
        left_box.addWidget(newcolor_button)
        newcolor_button.clicked.connect(self.newcolor_button_clicked)

        quit_button = QPushButton('quit', self)
        left_box.addWidget(quit_button)
        quit_button.clicked.connect(self.close)

        all_box = QHBoxLayout()
        all_box.addLayout(left_box)
        # vbox.addWidget(self.mpl_toolbar)
        all_box.addLayout(right_box)
        self.main_frame.setLayout(all_box)
        self.setCentralWidget(self.main_frame)

        # self.statusBar()

        self.setWindowTitle('Brain')

        self.statusBar().showMessage('Ctrl + Left Click to pick a color; Left Click to assign color to a superpixel; Scroll to zoom, Left Click + drag to pan')

        self.show()

    def newcolor_button_clicked(self):

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


    def get_data(self):
        data_dir = os.path.join(output_dir, result_name+'_data')
        # img_filename = os.path.join(data_dir, result_name + '_img_cropped.png')
        img_filename = os.path.join(data_dir, result_name + '_segmentation.tif')
        seg_filename = os.path.join(data_dir, result_name + '_segmentation.npy')
        labelmap_filename = os.path.join(data_dir, result_name + '_labelmap.npy')        

        self.segmentation = np.load(seg_filename)
        self.labelmap = np.load(labelmap_filename)
        self.n_models = 20
        self.n_labels = self.n_models + 1

        self.labelmap_new = self.labelmap.copy()

        self.colors = [(1,1,1)] + [(random(),random(),random()) for i in xrange(30)]
        self.label_cmap = ListedColormap(self.colors, name='label_cmap')
        # self.label_cmap = mpl.colors.LinearSegmentedColormap.from_list()
        self.img = cv2.imread(img_filename, 0)

    def on_draw(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')
        self.zoom_factory(self.axes, base_scale=1.5)
        #self.axes.plot(self.x, self.y, 'ro')
        self.axes.imshow(self.img, cmap=plt.cm.Greys_r,
                        aspect='equal')
        self.label_layer = self.axes.imshow(self.labelmap + 1, cmap=self.label_cmap, alpha=0.2,
                        aspect='equal', norm=NoNorm())
        self.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'control':
            self.pick_mode = True
            print 'pick mode on'

        # implement the default mpl key press events described at
        # http://matplotlib.org/users/navigation_toolbar.html#navigation-keyboard-shortcuts
        # key_press_handler(event, self.canvas, self.mpl_toolbar)

    def on_key_release(self, event):
        if event.key == 'control':
            self.pick_mode = False
            print 'pick mode off'

    def zoom_factory(self, ax, base_scale = 2.):
        self.press = False

        def zoom_fun(event):
            # get the current x and y limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
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
                scale_factor = 1/base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale            
                
            # set new limits
            ax.set_xlim([xdata - left*scale_factor,
                         xdata + right*scale_factor])
            ax.set_ylim([ydata - up*scale_factor,
                         ydata + down*scale_factor])
            ax.figure.canvas.draw() # force re-draw
     
        def press_fun(event):
            self.press_x = event.xdata
            self.press_y = event.ydata
            self.press = True

        def motion_fun(event):
            self.moved = True
            if self.press:
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                
                offset_x = self.press_x - event.xdata
                offset_y = self.press_y - event.ydata
                
                ax.set_xlim(cur_xlim + offset_x)
                ax.set_ylim(cur_ylim + offset_y)
                ax.figure.canvas.draw()

        def release_fun(event):
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

                curr_xlim = ax.get_xlim()
                curr_ylim = ax.get_ylim()

                self.label_layer.remove()

                self.label_layer = self.axes.imshow(self.labelmap_new + 1, cmap=self.label_cmap, alpha=.2,
                        aspect='equal', norm=NoNorm())

                self.axes.set_xlim(curr_xlim)
                self.axes.set_ylim(curr_ylim)
                self.axes.figure.canvas.draw() # force re-draw


            # # click and drag, means pan
            # else:
                 # force re-draw

        fig = ax.get_figure() # get the figure of interest
        # attach the call back
        fig.canvas.mpl_connect('scroll_event',zoom_fun)
        fig.canvas.mpl_connect('button_press_event',press_fun)
        fig.canvas.mpl_connect('button_release_event',release_fun)
        fig.canvas.mpl_connect('motion_notify_event',motion_fun)

def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()
