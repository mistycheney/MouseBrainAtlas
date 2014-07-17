import sys
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
from skimage.color import label2rgb
from random import random

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
        labeling:    A dictionary containing the mapping of super-pixels to colors (label_history) 
                      and ascii names for the colors.
        """
        self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.segmentation = segmentation
        self.n_superpixels=max(segmentation.flatten())+1
        self.img = img

        self.recalc_callback=recalc_callback
        self.save_callback=save_callback

        # Read in the labellist history. In labeling we store an array with the history of 
        # labelings that we can run through for a demo, perform undo, redo etc.
        # The fields in labeling are:
        # labellist: an array that maps segment (super-pixel) number to labels
        # names: an array that contains the textual name associated with the label.
        # label_history: The history of changes, together with full label-lists from time to time.
        #         The fields in each elements of the list are:
        #         name - name of user
        #         time
        #         full: Binary, true if all of the frame is saved, else is just the change
        #         data: if full: array of labels (labellist), if not: a pair: (segment_no, label)

        if labeling == None:
            print 'No labeling given'
            self.labeling={}
            self.labellist=-1*np.ones(self.n_superpixels)
            self.labeling['labellist']=self.labellist
            self.n_models=10
            self.labeling['names']=['No Label']+['Label %2d'%i for i in range(self.n_models+1)]
            self.labeling['label_history']=[]
        else:
            self.labeling=labeling
            self.labellist=labeling['labellist']
            self.n_models = max(10,np.max(self.labellist)+1)

        self.buttonTexts=self.labeling['names']
        self.n_labels = self.n_models + 1
        print 'n_labels',self.n_labels

        self.labelmap = -1*np.ones_like(self.segmentation, dtype=np.int)
        if self.n_labels>1:
            self.labelmap = self.labellist[self.segmentation]

        # A set of high-contrast Colors proposed by Green-Armytage 
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
        self.pick_mode  = False      # True while you hold ctrl; used to pick a color from the image
        self.press = False           # related to pan (press and drag) vs. select (click)
        self.base_scale = 1.05       # multiplication factor for zoom using scroll wheel
        self.moved=False             # Flag indicating whether mouse moved while button pressed

        self.create_main_frame()     # construct the layout
        self.initialize_canvas()     # initialize matplotlib plot

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
        self.colorButtons = []    # layout container for push buttons with colors
        self.buttonGroup = QButtonGroup(self) # event handling container for same buttons. 


        for i in range(-1, self.n_labels-1):
            button=QPushButton('%d'%(i), self)
            self.colorButtons.append(button)
            self.buttonGroup.addButton(button)

        self.buttonGroup.buttonClicked.connect(self.handleButtonClicked)
        

        self.descEdits = [QLineEdit(QString(self.buttonTexts[i])) for i in range(self.n_labels)] # corresponding decription fields
        self.newDescEdits=[]

        self.NameField=QLineEdit(QString('Your Name'))

        self.color_box = QGridLayout()
        self.color_box.addWidget(self.NameField, 0, 1)
        for i, (btn, edt) in enumerate(zip(self.colorButtons, self.descEdits)):
            r, g, b, a = self.label_cmap(i)
            btn.setStyleSheet("background-color: rgba(%d, %d, %d, 20%%)"%(
                int(r*255), int(g*255), int(b*255)))
            btn.setFixedSize(20, 20)
            self.color_box.addWidget(btn, i+1, 0)
            self.color_box.addWidget(edt, i+1, 1)

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

        self.label_layer=None  # to avoid removing layer when it is not yet there
        self.draw_colors()

        self.canvas.draw()


    # CALLBACKS 

    def handleButtonClicked(self, button):
        self.pick_color(int(button.text()))

    def on_key_press(self, event):
        if event.key == 'control':
            self.pick_mode = True

    def on_key_release(self, event):
        if event.key == 'control':
            self.pick_mode = False
        
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
            try:
                selected_sp = self.segmentation[int(event.ydata), int(event.xdata)]
                selected_label = self.labelmap[int(event.ydata), int(event.xdata)]
            except:
                return
            if self.pick_mode:
                # Picking a color
                self.pick_color(selected_label)
            else:
                #Painting a color
                self.statusBar().showMessage('Paint color: superpixel %d, selected label = %d' % (selected_sp, self.paint_label))
                self.change_seg_color(selected_sp)
            self.draw_colors()

    def pick_color(self,selected_label):
        self.paint_label = selected_label
        self.statusBar().showMessage('Choose label: selected label = %d' % self.paint_label)

    def change_seg_color(self,selected_sp):

        print 'self.paint_label',self.paint_label
        self.labellist[selected_sp]=self.paint_label
        self.labelmap = self.labellist[self.segmentation]

        timestamp=str(datetime.datetime.now())
        username=str(self.NameField.text())
        self.labeling['label_history'].append({'name':username,
                                               'time':timestamp,
                                               'Full':False,
                                               'data':(selected_sp,self.paint_label)
                                           })
        

    def draw_colors(self):
        if self.label_layer!=None:
            self.label_layer.remove()

        import collections

        print 'draw_colors: values in labelmap are',set(self.labelmap.flatten())
        self.label_layer = self.axes.imshow((self.labelmap + 1)/float(len(self.colors)), cmap=self.label_cmap, alpha=.2,
                aspect='equal', norm=NoNorm())

        #self.axes.set_xlim(cur_xlim)
        #self.axes.set_ylim(cur_ylim)
        self.axes.figure.canvas.draw() # force re-draw

    def get_labels(self):
        self.labeling['labeling']=self.labeling
        self.labeling['names']=[str(self.descEdits[i].text()) for i in range(len(self.descEdits))]
        timestamp=str(datetime.datetime.now())
        username=str(self.NameField.text())
        self.labeling['label_history'].append({'name':username,
                                               'time':timestamp,
                                               'Full':True,
                                               'data':self.labeling
                                           })
        
        return self.labeling

