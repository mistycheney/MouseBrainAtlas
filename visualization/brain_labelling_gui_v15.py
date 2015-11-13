#! /usr/bin/env python

import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

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

from matplotlib.backend_bases import key_press_handler, MouseEvent, KeyEvent
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
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib.colors import ListedColormap, NoNorm, ColorConverter
from matplotlib.path import Path
from matplotlib.text import Text

from skimage.color import label2rgb

from visualization_utilities import *

sys.path.append(os.environ['LOCAL_REPO_DIR'] + '/utilities')
from utilities2015 import *

from collections import defaultdict

import requests

from enum import Enum
class Mode(Enum):
    PLACING_VERTICES = 'placing vertices'
    POLYGON_SELECTED = 'polygon selected'
    REVIEW_PROPOSAL = 'review proposal'

class ProposalType(Enum):
    GLOBAL = 'global'
    LOCAL = 'local'
    FREEFORM = 'freeform'
    ALGORITHM = 'algorithm'

class PolygonType(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    TEXTURE = 'textured'
    TEXTURE_WITH_CONTOUR = 'texture with contour'
    DIRECTION = 'directionality'

SELECTED_POLYGON_LINEWIDTH = 5
UNSELECTED_POLYGON_LINEWIDTH = 3
SELECTED_CIRCLE_SIZE = 30
UNSELECTED_CIRCLE_SIZE = 5
CIRCLE_PICK_THRESH = 1000.

class ListSelection(QDialog):
    def __init__(self, item_ls, parent=None):
        super(ListSelection, self).__init__(parent)

        self.setWindowTitle('Detect which landmarks ?')

        self.selected = set([])

        self.listWidget = QListWidget()
        for item in item_ls:    
            w_item = QListWidgetItem(item)
            self.listWidget.addItem(w_item)

            w_item.setFlags(w_item.flags() | Qt.ItemIsUserCheckable)
            w_item.setCheckState(False)

        self.listWidget.itemChanged.connect(self.OnSingleClick)

        layout = QGridLayout()
        layout.addWidget(self.listWidget,0,0,1,3)

        self.but_ok = QPushButton("OK")
        layout.addWidget(self.but_ok ,1,1)
        self.but_ok.clicked.connect(self.OnOk)

        self.but_cancel = QPushButton("Cancel")
        layout.addWidget(self.but_cancel ,1,2)
        self.but_cancel.clicked.connect(self.OnCancel)

        self.setLayout(layout)
        self.setGeometry(300, 200, 460, 350)

    def OnSingleClick(self, item):
        if not item.checkState():
        #   item.setCheckState(False)
            self.selected = self.selected - {str(item.text())}
        #   print self.selected
        else:
        #   item.setCheckState(True)
            self.selected.add(str(item.text()))

        print self.selected

    def OnOk(self):
        self.close()

    def OnCancel(self):
        self.selected = set([])
        self.close()


class CustomQCompleter(QCompleter):
    # adapted from http://stackoverflow.com/a/26440173
    def __init__(self, *args):#parent=None):
        super(CustomQCompleter, self).__init__(*args)
        self.local_completion_prefix = ""
        self.source_model = None
        self.filterProxyModel = QSortFilterProxyModel(self)
        self.usingOriginalModel = False

    def setModel(self, model):
        self.source_model = model
        self.filterProxyModel = QSortFilterProxyModel(self)
        self.filterProxyModel.setSourceModel(self.source_model)
        super(CustomQCompleter, self).setModel(self.filterProxyModel)
        self.usingOriginalModel = True

    def updateModel(self):
        if not self.usingOriginalModel:
            self.filterProxyModel.setSourceModel(self.source_model)

        pattern = QRegExp(self.local_completion_prefix,
                                Qt.CaseInsensitive,
                                QRegExp.FixedString)

        self.filterProxyModel.setFilterRegExp(pattern)

    def splitPath(self, path):
        self.local_completion_prefix = path
        self.updateModel()
        if self.filterProxyModel.rowCount() == 0:
            self.usingOriginalModel = False
            self.filterProxyModel.setSourceModel(QStringListModel([path]))
            return [path]

        return []

class AutoCompleteComboBox(QComboBox):
    # adapted from http://stackoverflow.com/a/26440173
    def __init__(self, labels, *args, **kwargs):
        super(AutoCompleteComboBox, self).__init__(*args, **kwargs)

        self.setEditable(True)
        self.setInsertPolicy(self.NoInsert)

        self.comp = CustomQCompleter(self)
        self.comp.setCompletionMode(QCompleter.PopupCompletion)
        self.setCompleter(self.comp)#
        self.setModel(labels)

    def setModel(self, strList):
        self.clear()
        self.insertItems(0, strList)
        self.comp.setModel(self.model())

    def focusInEvent(self, event):
        self.clearEditText()
        super(AutoCompleteComboBox, self).focusInEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == 16777220:
            # Enter (if event.key() == QtCore.Qt.Key_Enter) does not work
            # for some reason

            # make sure that the completer does not set the
            # currentText of the combobox to "" when pressing enter
            text = self.currentText()
            self.setCompleter(None)
            self.setEditText(text)
            self.setCompleter(self.comp)

        return super(AutoCompleteComboBox, self).keyPressEvent(event)

class AutoCompleteInputDialog(QDialog):

    def __init__(self, labels, *args, **kwargs):
        super(AutoCompleteInputDialog, self).__init__(*args, **kwargs)
        self.comboBox = AutoCompleteComboBox(parent=self, labels=labels)
        va = QVBoxLayout(self)
        va.addWidget(self.comboBox)
        box = QWidget(self)
        ha = QHBoxLayout(self)
        va.addWidget(box)
        box.setLayout(ha)
        self.OK = QPushButton("OK", self)
        self.OK.setDefault(True)
        # cancel = QPushButton("Cancel", self)
        ha.addWidget(self.OK)
        # ha.addWidget(cancel)

    def set_test_callback(self, callback):
        self.OK.clicked.connect(callback)
        # OK.clicked.connect(self.accept)
        # cancel.clicked.connect(self.reject)

class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
    def __init__(self, parent=None, stack=None):
        """
        Initialization of BrainLabelingGUI.
        """

        self.params_dir = '../params'

        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        # self.init_data(stack)
        self.stack = stack
        self.initialize_brain_labeling_gui()

        from collections import OrderedDict
        
        self.structure_names = {}
        with open('structure_names.txt', 'r') as f:
            for ln in f.readlines():
                abbr, fullname = ln.split('\t')
                self.structure_names[abbr] = fullname.strip()

        self.structure_names = OrderedDict(sorted(self.structure_names.items()))

    def init_data(self, section):

        self.section = section

        self.dm = DataManager(
            data_dir=os.environ['LOCAL_DATA_DIR'], 
                 repo_dir=os.environ['LOCAL_REPO_DIR'], 
                 result_dir=os.environ['LOCAL_RESULT_DIR'], 
                 labeling_dir=os.environ['LOCAL_LABELING_DIR'],
            stack=stack, section=section, segm_params_id='tSLIC200')

        print self.dm.slice_ind

        t = time.time()
        self.dm._load_image(versions=['rgb-jpg'])
        print 1, time.time() - t

        t = time.time()
        required_results = [
        # 'segmentationTransparent', 
        'segmentation',
        # 'segmentationWithText',
        # 'allSeedClusterScoreDedgeTuples',
        # 'proposals',
        # 'spCoveredByProposals',
        'edgeMidpoints',
        'edgeEndpoints',
        # 'spAreas',
        # 'spBbox',
        # 'spCentroids'
        ]

        # t = time.time()
        # self.dm.download_results(required_results)
        # print 2, time.time() - t

        t = time.time()
        self.dm.load_multiple_results(required_results, download_if_not_exist=True)
        print 3, time.time() - t

        self.segm_transparent = None
        self.under_img = None
        self.textonmap_vis = None
        self.dirmap_vis = None

        self.mode = Mode.REVIEW_PROPOSAL

        self.boundary_colors = [(0,1,1), (0,1,0), (1,0,0),(0,0,1)] # unknown, accepted, rejected

        self.accepted_proposals = defaultdict(dict)

        self.curr_proposal_pathPatch = None
        self.alternative_global_proposal_ind = 0
        self.alternative_local_proposal_ind = 0

        self.new_labelnames = []

    def paramSettings_clicked(self):
        pass

    def openMenu(self, canvas_pos):

        self.endDrawClosed_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
        # self.endDrawOpen_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
        # self.confirmTexture_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
        # self.confirmTextureWithContour_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
        # self.confirmDirectionality_Action.setVisible(self.mode == Mode.PLACING_VERTICES)
        # self.deletePolygon_Action.setVisible(self.selected_polygon is not None)
        self.deleteVertex_Action.setVisible(self.selected_circle is not None)
        self.addVertex_Action.setVisible(self.curr_proposal_pathPatch is not None and \
            self.curr_proposal_pathPatch in self.accepted_proposals and \
            self.accepted_proposals[self.curr_proposal_pathPatch]['type'] == ProposalType.FREEFORM)

        # self.newPolygon_Action.setVisible(self.curr_proposal_pathPatch is None and self.mode == Mode.REVIEW_PROPOSAL)
        self.newPolygon_Action.setVisible(self.mode == Mode.REVIEW_PROPOSAL)

        self.accProp_Action.setVisible(self.curr_proposal_pathPatch is not None and self.curr_proposal_pathPatch not in self.accepted_proposals)
        self.rejProp_Action.setVisible(self.curr_proposal_pathPatch is not None and self.curr_proposal_pathPatch in self.accepted_proposals)
        self.changeLabel_Action.setVisible(self.curr_proposal_pathPatch is not None and self.curr_proposal_pathPatch in self.accepted_proposals)

        action = self.menu.exec_(self.cursor().pos())

        if action == self.endDrawClosed_Action:

            polygon = Polygon(self.curr_polygon_vertices, closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=UNSELECTED_POLYGON_LINEWIDTH)
            self.axis.add_patch(polygon)
            polygon.set_picker(True)

            self.curr_proposal_type = ProposalType.FREEFORM

            self.curr_proposal_pathPatch = polygon

            self.accepted_proposals[self.curr_proposal_pathPatch] = {'type': self.curr_proposal_type,
                                                                    'subtype': PolygonType.CLOSED,
                                                                    'vertices': self.curr_polygon_vertices,
                                                                    'vertexPatches': self.curr_polygon_vertex_circles
                                                                    }

            self.mode = Mode.REVIEW_PROPOSAL

            self.curr_polygon_vertices = []
            self.curr_polygon_vertex_circles = []

            self.accProp_callback()
            
        # elif action == self.endDrawOpen_Action:

        #   self.add_polygon(self.curr_polygon_vertices, PolygonType.OPEN)
        #   # self.statusBar().showMessage('Done drawing edge segment using label %d (%s)' % (self.curr_label,
        #   #                                           self.free_proposal_labels[self.curr_label]))

        #   self.mode = Mode.REVIEW_PROPOSAL

        #   self.curr_proposal_type = ProposalType.FREEFORM
        #   self.curr_proposal_id = len(self.freeform_proposal_labels)
        #   self.curr_proposal_pathPatch = self.freeform_polygons[self.curr_proposal_id]

        #   self.freeform_proposal_labels.append('')

        #   self.accProp_callback()

        # elif action == self.confirmTexture_Action:
        #   self.add_polygon(self.curr_polygon_vertices, PolygonType.TEXTURE)
        #   # self.statusBar().showMessage('Done drawing textured regions using label %d (%s)' % (self.curr_label,
        #   #                                           self.free_proposal_labels[self.curr_label]))
        #   self.mode = Mode.REVIEW_PROPOSAL

        #   self.curr_proposal_type = ProposalType.FREEFORM
        #   self.curr_proposal_id = len(self.freeform_proposal_labels)
        #   self.curr_proposal_pathPatch = self.freeform_polygons[self.curr_proposal_id]

        #   self.freeform_proposal_labels.append('')

        #   self.accProp_callback()

        # elif action == self.confirmTextureWithContour_Action:
        #   self.add_polygon(self.curr_polygon_vertices, PolygonType.TEXTURE_WITH_CONTOUR)
        #   # self.statusBar().showMessage('Done drawing textured regions using label %d (%s)' % (self.curr_label,
        #   #                                           self.free_proposal_labels[self.curr_label]))
        #   self.mode = Mode.REVIEW_PROPOSAL

        #   self.curr_proposal_type = ProposalType.FREEFORM
        #   self.curr_proposal_id = len(self.freeform_proposal_labels)
        #   self.curr_proposal_pathPatch = self.freeform_polygons[self.curr_proposal_id]

        #   self.freeform_proposal_labels.append('')

        #   self.accProp_callback()

        # elif action == self.confirmDirectionality_Action:
        #   self.add_polygon(self.curr_polygon_vertices, PolygonType.DIRECTION)
        #   # self.statusBar().showMessage('Done drawing striated regions using label %d (%s)' % (self.curr_label,
        #   #                                           self.free_proposal_labels[self.curr_label]))
        #   self.mode = Mode.REVIEW_GLOBAL_PROPOSAL
        #   self.curr_freeform_polygon_id = len(self.freeform_proposal_labels)
        #   self.freeform_proposal_labels.append('')
        #   self.accProp_callback()
    
        # elif action == self.deletePolygon_Action:
        #   self.remove_polygon()

        elif action == self.deleteVertex_Action:
            self.remove_selected_vertex()

        elif action == self.addVertex_Action:
            self.add_vertex_to_existing_polygon(canvas_pos)

        # elif action == self.crossReference_Action:
        #   self.parent().refresh_data()
        #   self.parent().comboBoxBrowseMode.setCurrentIndex(self.curr_label + 1)
        #   self.parent().set_labelnameFilter(self.curr_label)
        #   self.parent().switch_to_labeling()

        elif action == self.accProp_Action:
            self.accProp_callback()

        elif action == self.rejProp_Action:
            self.rejProp_callback()

        elif action == self.newPolygon_Action:
            
            self.cancel_current_selection()

            self.statusBar().showMessage('Left click to place vertices')
            self.mode = Mode.PLACING_VERTICES

        elif action == self.changeLabel_Action:
            self.open_label_selection_dialog()

        else:
            # raise 'do not know how to deal with action %s' % action
            pass


    def reload_brain_labeling_gui(self):

        # self.click_on_object = False
        self.seg_loaded = False
        self.superpixels_on = False
        self.labels_on = True
        self.object_picked = False
        self.shuffle_global_proposals = True # instead of local proposals

        self.base_scale = 1.2       # multiplication factor for zoom using scroll wheel

        self.pressed = False           # related to pan (press and drag) vs. select (click)

        self.selected_circle = None
        self.selected_polygon = None

        self.curr_polygon_vertices = []
        self.curr_polygon_vertex_circles = []

        # self.polygon_types = []

        self.setWindowTitle('BrainLabelingGUI, stack %s'%self.stack + ', section %d' %self.section)

        # self.spinBox_section.setValue(self.section)

        # self.fig.clear()
        # self.fig.set_facecolor('white')

        if hasattr(self, 'axis'):
            for p in self.axis.patches:
                p.remove()
        else:
            self.axis = self.fig.add_subplot(111)
            self.axis.axis('off')

        t = time.time()
        self.orig_image_handle = self.axis.imshow(self.dm.image_rgb_jpg, cmap=plt.cm.Greys_r,aspect='equal')
        print 4, time.time() - t
        
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        self.newxmin, self.newxmax = self.axis.get_xlim()
        self.newymin, self.newymax = self.axis.get_ylim()

        self.canvas.draw()
        self.show()

    def initialize_brain_labeling_gui(self):

        self.menu = QMenu()
        self.endDrawClosed_Action = self.menu.addAction("Confirm closed contour")

        # self.endDrawOpen_Action = self.menu.addAction("Confirm open boundary")
        # self.confirmTexture_Action = self.menu.addAction("Confirm textured region without contour")
        # self.confirmTextureWithContour_Action = self.menu.addAction("Confirm textured region with contour")
        # self.confirmDirectionality_Action = self.menu.addAction("Confirm striated region")

        # self.deletePolygon_Action = self.menu.addAction("Delete polygon")
        self.deleteVertex_Action = self.menu.addAction("Delete vertex")
        self.addVertex_Action = self.menu.addAction("Add vertex")

        self.newPolygon_Action = self.menu.addAction("New polygon")

        # self.crossReference_Action = self.menu.addAction("Cross reference")

        self.accProp_Action = self.menu.addAction("Accept")
        self.rejProp_Action = self.menu.addAction("Reject")

        self.changeLabel_Action = self.menu.addAction('Change label')

        # A set of high-contrast colors proposed by Green-Armytage
        self.colors = np.loadtxt('100colors.txt', skiprows=1)
        self.label_cmap = ListedColormap(self.colors, name='label_cmap')

        self.setupUi(self)

        self.fig = self.canvaswidget.fig
        self.canvas = self.canvaswidget.canvas

        self.canvas.setFocusPolicy( Qt.ClickFocus )
        self.canvas.setFocus()

        self.canvas.mpl_connect('scroll_event', self.on_zoom)
        self.bpe_id = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.bre_id = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.canvas.mpl_connect('key_press_event', self.on_zoom)

        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.button_autoDetect.clicked.connect(self.autoDetect_callback)
        self.button_updateDB.clicked.connect(self.updateDB_callback)
        self.button_loadLabeling.clicked.connect(self.load_callback)
        self.button_saveLabeling.clicked.connect(self.save_callback)
        self.button_quit.clicked.connect(self.close)
        
        self.spinBox_section.setRange(0, 200)
        self.spinBox_section.valueChanged.connect(self.section_changed)

        self.display_buttons = [self.img_radioButton, self.textonmap_radioButton, self.dirmap_radioButton]
        self.img_radioButton.setChecked(True)

        for b in self.display_buttons:
            b.toggled.connect(self.display_option_changed)

        self.radioButton_globalProposal.toggled.connect(self.mode_changed)
        self.radioButton_localProposal.toggled.connect(self.mode_changed)

        self.buttonSpOnOff.clicked.connect(self.display_option_changed)
        self.button_labelsOnOff.clicked.connect(self.toggle_labels)


        # self.thumbnail_list = QListWidget(parent=self)
        # self.thumbnail_list.setIconSize(QSize(200,200))
        self.thumbnail_list.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))
        self.thumbnail_list.addItem(QListWidgetItem(QIcon("/home/yuncong/CSHL_data_processed/MD593_thumbnail_warped/MD593_0130_thumbnail_warped.tif"), '130'))


    def toggle_labels(self):

        self.labels_on = not self.labels_on

        if not self.labels_on:

            for patch, props in self.accepted_proposals.iteritems():
                props['labelTextArtist'].remove()

            self.button_labelsOnOff.setText('Turns Labels ON')

        else:
            for patch, props in self.accepted_proposals.iteritems():
                self.axis.add_artist(props['labelTextArtist'])

            self.button_labelsOnOff.setText('Turns Labels OFF')

        self.canvas.draw()

    def updateDB_callback(self):
        cmd = 'rsync -az --include="*/" %(local_labeling_dir)s/%(stack)s yuncong@gcn-20-33.sdsc.edu:%(gordon_labeling_dir)s' % {'gordon_labeling_dir':os.environ['GORDON_LABELING_DIR'],
                                                                            'local_labeling_dir':os.environ['LOCAL_LABELING_DIR'],
                                                                            'stack': self.stack
                                                                            }
        os.system(cmd)

        cmd = 'rsync -az %(local_labeling_dir)s/labelnames.txt yuncong@gcn-20-33.sdsc.edu:%(gordon_labeling_dir)s' % {'gordon_labeling_dir':os.environ['GORDON_LABELING_DIR'],
                                                                    'local_labeling_dir':os.environ['LOCAL_LABELING_DIR'],
                                                                    }
        os.system(cmd)
        self.statusBar().showMessage('labelings synced')

        # payload = {'section': self.dm.slice_ind}
        # r = requests.get('http://gcn-20-32.sdsc.edu:5000/update_db', params=payload)
        r = requests.get('http://gcn-20-32.sdsc.edu:5000/update_db')
        res = r.json()
        if res['result'] == 0:
            self.statusBar().showMessage('Landmark database updated')


    def detect_landmark(self, labels):

        payload = {'labels': labels, 'section': self.dm.slice_ind}
        r = requests.get('http://gcn-20-32.sdsc.edu:5000/top_down_detect', params=payload)
        print r.url
        return r.json()

    def autoDetect_callback(self):
        self.labelsToDetect = ListSelection(self.dm.labelnames, parent=self)
        self.labelsToDetect.exec_()

        if len(self.labelsToDetect.selected) > 0:
        
            returned_alg_proposal_dict = self.detect_landmark(list(self.labelsToDetect.selected)) # list of tuples (sps, dedges, sig)

            for label, (sps, dedges, sig) in returned_alg_proposal_dict.iteritems():
                pp = self.pathPatch_from_dedges(dedges, color=self.boundary_colors[1])
                pp.set_picker(True)
                self.accepted_proposals[pp] = {'sps': sps, 'dedges': dedges, 'sig': sig, 'type':ProposalType.ALGORITHM,
                                            'label': label}

                self.axis.add_patch(pp)
        
        self.canvas.draw()


    def on_pick(self, event):

        if self.mode == Mode.PLACING_VERTICES:
            return

        self.object_picked = True

        patch_vertexInd_tuple = [(patch, props['vertexPatches'].index(event.artist)) for patch, props in self.accepted_proposals.iteritems() 
                    if 'vertexPatches' in props and event.artist in props['vertexPatches']]
        
        if len(patch_vertexInd_tuple) == 1:

            self.cancel_current_selection()

            print 'clicked on a vertex circle'
            self.curr_proposal_pathPatch = patch_vertexInd_tuple[0][0]
            self.selected_vertex_index = patch_vertexInd_tuple[0][1]

            self.selected_circle = event.artist
            self.selected_circle.set_radius(SELECTED_CIRCLE_SIZE)
            
            self.selected_polygon = self.curr_proposal_pathPatch

            self.curr_proposal_pathPatch.set_linewidth(SELECTED_POLYGON_LINEWIDTH)

            self.statusBar().showMessage('picked %s proposal (%s, %s), vertex %d' % (self.accepted_proposals[self.curr_proposal_pathPatch]['type'].value,
                                                                     self.accepted_proposals[self.curr_proposal_pathPatch]['label'],
                                                                     self.structure_names[self.accepted_proposals[self.curr_proposal_pathPatch]['label']],
                                                                     self.selected_vertex_index))

            self.curr_proposal_type = ProposalType.FREEFORM

        elif len(patch_vertexInd_tuple) == 0: # ignore if circle is picked
            print 'clicked on a polygon'

            if not (self.selected_circle is not None and \
                    self.accepted_proposals[self.curr_proposal_pathPatch]['type'] == ProposalType.FREEFORM):
                self.cancel_current_selection()
            else:
                return

            if event.artist in self.accepted_proposals:

                self.curr_proposal_pathPatch = event.artist
                self.curr_proposal_pathPatch.set_linewidth(SELECTED_POLYGON_LINEWIDTH)

                if self.accepted_proposals[self.curr_proposal_pathPatch]['type'] == ProposalType.FREEFORM:
                    self.selected_polygon = self.curr_proposal_pathPatch

                    self.selected_polygon_xy_before_drag = self.selected_polygon.get_xy()
                    self.selected_polygon_circle_centers_before_drag = [circ.center 
                                        for circ in self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches']]

                self.statusBar().showMessage('picked %s proposal (%s, %s)' % (self.accepted_proposals[self.curr_proposal_pathPatch]['type'].value,
                                                                         self.accepted_proposals[self.curr_proposal_pathPatch]['label'],
                                                                         self.structure_names[self.accepted_proposals[self.curr_proposal_pathPatch]['label']]))

                self.curr_proposal_type = self.accepted_proposals[self.curr_proposal_pathPatch]['type']

        else:
            raise 'unknown situation'

        self.canvas.draw()

    def pathPatch_from_dedges(self, dedges, color):

        vertices = []
        for de_ind, de in enumerate(dedges):
            midpt = self.dm.edge_midpoints[frozenset(de)]
            endpts = self.dm.edge_endpoints[frozenset(de)]
            endpts_next_dedge = self.dm.edge_endpoints[frozenset(dedges[(de_ind+1)%len(dedges)])]

            dij = cdist([endpts[0], endpts[-1]], [endpts_next_dedge[0], endpts_next_dedge[-1]])
            i,j = np.unravel_index(np.argmin(dij), (2,2))
            if i == 0:
                vertices += [endpts[-1], midpt, endpts[0]]
            else:
                vertices += [endpts[0], midpt, endpts[-1]]

        path_patch = PathPatch(Path(vertices=vertices, closed=True), color=color, fill=False, linewidth=UNSELECTED_POLYGON_LINEWIDTH)

        return path_patch

    def load_local_proposals(self):

        sys.stderr.write('loading local proposals ...\n')
        self.statusBar().showMessage('loading local proposals ...')
        
        cluster_tuples = self.dm.load_pipeline_result('allSeedClusterScoreDedgeTuples')
        self.local_proposal_tuples = [(cl, ed, sig) for seed, cl, sig, ed in cluster_tuples]
        self.local_proposal_clusters = [m[0] for m in self.local_proposal_tuples]
        self.local_proposal_dedges = [m[1] for m in self.local_proposal_tuples]
        self.local_proposal_sigs = [m[2] for m in self.local_proposal_tuples]

        self.n_local_proposals = len(self.local_proposal_tuples)
        
        if not hasattr(self, 'local_proposal_pathPatches'):
            self.local_proposal_pathPatches = [None] * self.n_local_proposals

        self.local_proposal_indices_from_sp = defaultdict(list)
        for i, (seed, _, _, _) in enumerate(cluster_tuples):
            self.local_proposal_indices_from_sp[seed].append(i)
        self.local_proposal_indices_from_sp.default_factory = None

        sys.stderr.write('%d local proposals loaded.\n' % self.n_local_proposals)
        self.statusBar().showMessage('Local proposals loaded.')

        self.local_proposal_labels = [None] * self.n_local_proposals


    def load_global_proposals(self):
        
        self.global_proposal_tuples =  self.dm.load_pipeline_result('proposals')
        self.global_proposal_clusters = [m[0] for m in self.global_proposal_tuples]
        self.global_proposal_dedges = [m[1] for m in self.global_proposal_tuples]
        self.global_proposal_sigs = [m[2] for m in self.global_proposal_tuples]

        self.n_global_proposals = len(self.global_proposal_tuples)

        if not hasattr(self, 'global_proposal_pathPatches'):
            self.global_proposal_pathPatches = [None] * self.n_global_proposals

        self.statusBar().showMessage('%d global proposals loaded' % self.n_global_proposals)

        self.sp_covered_by_proposals = self.dm.load_pipeline_result('spCoveredByProposals')
        self.sp_covered_by_proposals = dict([(s, list(props)) for s, props in self.sp_covered_by_proposals.iteritems()])

        self.global_proposal_labels = [None] * self.n_global_proposals

    def load_callback(self):

        fname = str(QFileDialog.getOpenFileName(self, 'Open file', self.dm.labelings_dir))
        stack, sec, username, timestamp, suffix = os.path.basename(fname[:-4]).split('_')

        # if suffix == 'consolidated':

        self.accepted_proposals = {}

        _, _, _, accepted_proposal_props = self.dm.load_proposal_review_result(username, timestamp, suffix)

        for props in accepted_proposal_props:
            if props['type'] == ProposalType.GLOBAL or props['type'] == ProposalType.LOCAL or props['type'] == ProposalType.ALGORITHM:
                patch = self.pathPatch_from_dedges(props['dedges'], color=self.boundary_colors[1])
            elif props['type'] == ProposalType.FREEFORM:
                patch = Polygon(props['vertices'], closed=True, fill=False, edgecolor=self.boundary_colors[1], linewidth=UNSELECTED_POLYGON_LINEWIDTH)
                props['vertexPatches'] = []
                for x,y in props['vertices']:
                    vertex_circle = plt.Circle((x, y), radius=UNSELECTED_CIRCLE_SIZE, color=self.boundary_colors[1], alpha=.8)
                    vertex_circle.set_picker(CIRCLE_PICK_THRESH)
                    props['vertexPatches'].append(vertex_circle)
                    self.axis.add_patch(vertex_circle)

            self.axis.add_patch(patch)
            patch.set_picker(True)

            self.accepted_proposals[patch] = props

        self.canvas.draw()


    def open_label_selection_dialog(self):

        self.label_selection_dialog = AutoCompleteInputDialog(parent=self, labels=[abbr + ' (' + fullname + ')' for abbr, fullname in self.structure_names.iteritems()])
        # self.label_selection_dialog = QInputDialog(self)
        self.label_selection_dialog.setWindowTitle('Select landmark label')

        # self.label_selection_dialog.setComboBoxItems(['New label'] + sorted([abbr + ' (' + fullname + ')' for abbr, fullname in self.structure_names.iteritems()] + self.new_labelnames))

        if 'label' in self.accepted_proposals[self.curr_proposal_pathPatch]:
            self.label_selection_dialog.comboBox.setEditText(self.accepted_proposals[self.curr_proposal_pathPatch]['label'])
        else:
            self.accepted_proposals[self.curr_proposal_pathPatch]['label'] = ''

        self.label_selection_dialog.set_test_callback(self.label_dialog_text_changed)

        # self.label_selection_dialog.accepted.connect(self.label_dialog_text_changed)
        # self.label_selection_dialog.textValueSelected.connect(self.label_dialog_text_changed)

        self.label_selection_dialog.exec_()

    # def set_selected_proposal_label(self, label):
    #   self.accepted_proposals[self.curr_proposal_pathPatch]['label'] = label

    def label_dialog_text_changed(self):

        text = str(self.label_selection_dialog.comboBox.currentText())

        import re
        m = re.match('^(.+?)\s*\((.+)\)$', text)

        if m is None:
            QMessageBox.warning(self, 'oops', 'structure name must be of the form "abbreviation (full description)"')
            return

        else:
            if text not in [abbr + ' (' + fullname + ')' for abbr, fullname in self.structure_names.iteritems()]:  # new label
                abbr, fullname = m.groups()
                if abbr in self.structure_names:
                    QMessageBox.warning(self, 'oops', 'structure with abbreviation %s already exists' % abbr)
                    return
                else:
                    self.structure_names[abbr] = fullname
        
        self.accepted_proposals[self.curr_proposal_pathPatch]['label'] = abbr

        if 'labelTextArtist' in self.accepted_proposals[self.curr_proposal_pathPatch] and self.accepted_proposals[self.curr_proposal_pathPatch]['labelTextArtist'] is not None:
            self.accepted_proposals[self.curr_proposal_pathPatch]['labelTextArtist'].set_text(abbr)
        else:
            centroid = self.curr_proposal_pathPatch.get_xy().mean(axis=0)
            text_artist = Text(centroid[0], centroid[1], abbr, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
            self.accepted_proposals[self.curr_proposal_pathPatch]['labelTextArtist'] = text_artist
            self.axis.add_artist(text_artist)

        self.label_selection_dialog.accept()


    def accProp_callback(self):

        if self.curr_proposal_type == ProposalType.GLOBAL:
            self.accepted_proposals[self.curr_proposal_pathPatch] = {'sps': self.global_proposal_clusters[self.curr_proposal_id],
                                                                    'dedges': self.global_proposal_dedges[self.curr_proposal_id],
                                                                    'sig': self.global_proposal_sigs[self.curr_proposal_id],
                                                                    'type': self.curr_proposal_type,
                                                                    'id': self.curr_proposal_id}

        elif self.curr_proposal_type == ProposalType.LOCAL:
            self.accepted_proposals[self.curr_proposal_pathPatch] = {'sps': self.local_proposal_clusters[self.curr_proposal_id],
                                                                    'dedges': self.local_proposal_dedges[self.curr_proposal_id],
                                                                    'sig': self.local_proposal_sigs[self.curr_proposal_id],
                                                                    'type': self.curr_proposal_type,
                                                                    'id': self.curr_proposal_id}

        self.curr_proposal_pathPatch.set_color(self.boundary_colors[1])
        self.curr_proposal_pathPatch.set_picker(True)
            
        self.canvas.draw()

        self.open_label_selection_dialog()

    def rejProp_callback(self):

        if self.curr_proposal_type == ProposalType.FREEFORM:
            self.selected_polygon = None
            for circ in self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches']:
                circ.remove()

        self.accepted_proposals.pop(self.curr_proposal_pathPatch)

        self.curr_proposal_pathPatch.remove()
        self.curr_proposal_pathPatch.set_color(self.boundary_colors[0])
        self.curr_proposal_pathPatch.set_picker(None)

        self.canvas.draw()
    
    def show_global_proposal_covering_sp(self, sp_ind):

        if sp_ind not in self.sp_covered_by_proposals or sp_ind == -1:
            self.statusBar().showMessage('No proposal covers superpixel %d' % sp_ind)
            return 

        self.cancel_current_selection()

        self.curr_proposal_type = ProposalType.GLOBAL
        self.alternative_global_proposal_ind = (self.alternative_global_proposal_ind + 1) % len(self.sp_covered_by_proposals[sp_ind])
        self.curr_proposal_id = self.sp_covered_by_proposals[sp_ind][self.alternative_global_proposal_ind]

        if self.global_proposal_pathPatches[self.curr_proposal_id] is None:
            self.global_proposal_pathPatches[self.curr_proposal_id] = self.pathPatch_from_dedges(self.global_proposal_dedges[self.curr_proposal_id], 
                                        color=self.boundary_colors[0])
        
        self.curr_proposal_pathPatch = self.global_proposal_pathPatches[self.curr_proposal_id]

        if self.curr_proposal_pathPatch not in self.axis.patches:
            self.axis.add_patch(self.curr_proposal_pathPatch)

        self.curr_proposal_pathPatch.set_picker(None)

        if self.curr_proposal_pathPatch in self.accepted_proposals:
            self.curr_proposal_pathPatch.set_linewidth(UNSELECTED_POLYGON_LINEWIDTH)
            label =  self.accepted_proposals[self.curr_proposal_pathPatch]['label']
        else:
            label = ''          

        self.statusBar().showMessage('global proposal (%s) covering seed %d, score %.4f' % (label, sp_ind, self.global_proposal_sigs[self.curr_proposal_id]))
        self.canvas.draw()

    def show_local_proposal_from_sp(self, sp_ind):

        self.cancel_current_selection()

        self.curr_proposal_type = ProposalType.LOCAL

        if sp_ind == -1:
            return

        self.alternative_local_proposal_ind = (self.alternative_local_proposal_ind + 1) % len(self.local_proposal_indices_from_sp[sp_ind])
        self.curr_proposal_id = self.local_proposal_indices_from_sp[sp_ind][self.alternative_local_proposal_ind]

        cl, dedges, sig = self.local_proposal_tuples[self.curr_proposal_id]

        if self.local_proposal_pathPatches[self.curr_proposal_id] is None:  
            self.local_proposal_pathPatches[self.curr_proposal_id] = self.pathPatch_from_dedges(dedges, 
                                                                color=self.boundary_colors[0])

        self.curr_proposal_pathPatch = self.local_proposal_pathPatches[self.curr_proposal_id]

        if self.curr_proposal_pathPatch not in self.axis.patches:
            self.axis.add_patch(self.curr_proposal_pathPatch)

        self.curr_proposal_pathPatch.set_picker(None)

        if  self.curr_proposal_pathPatch in self.accepted_proposals:
            self.curr_proposal_pathPatch.set_linewidth(UNSELECTED_POLYGON_LINEWIDTH)
            label = self.accepted_proposals[self.curr_proposal_pathPatch]['label']
        else:
            label = ''

        self.statusBar().showMessage('local proposal (%s) from seed %d, score %.4f' % (label, sp_ind, sig))
        self.canvas.draw()


    def section_changed(self, val):

        self.spinBox_section.findChild(QLineEdit).deselect()
        
        if hasattr(self, 'global_proposal_tuples'):
            del self.global_proposal_tuples
        if hasattr(self, 'global_proposal_pathPatches'):
            del self.global_proposal_pathPatches
        if hasattr(self, 'local_proposal_tuples'):
            del self.local_proposal_tuples
        if hasattr(self, 'local_proposal_pathPatches'):
            del self.local_proposal_pathPatches

        self.init_data(section=val)
        self.reload_brain_labeling_gui()

        self.mode_changed()


    def save_callback(self):

        timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
        username, okay = QInputDialog.getText(self, "Username", "Please enter your username:", QLineEdit.Normal, 'anon')
        if not okay: return

        self.username = str(username)

        accepted_proposal_props = []
        for patch, props in self.accepted_proposals.iteritems():
            accepted_proposal_props.append(dict([(k,v) for k, v in props.iteritems() if k != 'vertexPatches']))

        self.dm.save_proposal_review_result(accepted_proposal_props, self.username, timestamp, suffix='consolidated')

        self.dm.add_labels(self.new_labelnames)

        self.statusBar().showMessage('Labelings saved to %s' % (self.username+'_'+timestamp))

        cur_xlim = self.axis.get_xlim()
        cur_ylim = self.axis.get_ylim()

        self.axis.set_xlim([0, self.dm.image_width])
        self.axis.set_ylim([self.dm.image_height, 0])

        self.fig.savefig('/tmp/preview.jpg', bbox_inches='tight')

        self.axis.set_xlim(cur_xlim)
        self.axis.set_ylim(cur_ylim)

        self.canvas.draw()

    def labelbutton_callback(self):
        pass

    ############################################
    # matplotlib canvas CALLBACKs
    ############################################

    def on_zoom(self, event):

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

        if isinstance(event, MouseEvent):

            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1/self.base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = self.base_scale

        elif isinstance(event, KeyEvent):
            if event.key == '=':
                scale_factor = 1/self.base_scale
            elif event.key == '-':
                scale_factor = self.base_scale              
        
        self.newxmin = xdata - left*scale_factor
        self.newxmax = xdata + right*scale_factor
        self.newymin = ydata - up*scale_factor
        self.newymax = ydata + down*scale_factor

        self.axis.set_xlim([self.newxmin, self.newxmax])
        self.axis.set_ylim([self.newymin, self.newymax])

        self.canvas.draw() # force re-draw

    def on_press(self, event):
        self.press_x = event.xdata
        self.press_y = event.ydata

        self.pressed = True
        self.press_time = time.time()


    def on_motion(self, event):

        if self.mode == Mode.PLACING_VERTICES:
            return
        
        if hasattr(self, 'selected_circle') and self.selected_circle is not None and self.pressed and self.object_picked: # drag vertex

            print 'dragging vertex'

            self.selected_circle.center = event.xdata, event.ydata

            xys = self.selected_polygon.get_xy()
            xys[self.selected_vertex_index] = self.selected_circle.center

            if self.selected_polygon.get_closed():
                self.selected_polygon.set_xy(xys[:-1])
            else:
                self.selected_polygon.set_xy(xys)
            
            self.canvas.draw()

        elif hasattr(self, 'selected_polygon') and self.selected_polygon is not None and self.pressed and self.object_picked and bool(self.selected_polygon.contains(event)[0]):
            # drag polygon

            print 'dragging polygon'

            offset_x = event.xdata - self.press_x
            offset_y = event.ydata - self.press_y

            for c, center0 in zip(self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches'], 
                                    self.selected_polygon_circle_centers_before_drag):
                c.center = (center0[0] + offset_x, center0[1] + offset_y)

            xys = self.selected_polygon_xy_before_drag + (offset_x, offset_y)
            
            if self.selected_polygon.get_closed():
                self.selected_polygon.set_xy(xys[:-1])
            else:
                self.selected_polygon.set_xy(xys)
            
            self.canvas.draw()


        elif hasattr(self, 'pressed') and self.pressed and time.time() - self.press_time > .5:

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


    def on_release(self, event):
        """
        The release-button callback is responsible for picking a color or changing a color.
        """

        self.pressed = False
        self.release_x = event.xdata
        self.release_y = event.ydata
        self.release_time = time.time()

        print self.mode, 

        # Fixed panning issues by using the time difference between the press and release event
        # Long times refer to a press and hold
        # if (self.release_time - self.press_time) < .21 and self.release_x > 0 and self.release_y > 0 and\
        #                   self.release_x - self.press_x < 100 and self.release_y - self.press_y < 100:
        if self.release_x - self.press_x < 100 and self.release_y - self.press_y < 100:
            # fast click

            if event.button == 1: # left click
                            
                if self.mode == Mode.PLACING_VERTICES:
                    self.place_vertex(event.xdata, event.ydata)

                else:
                    if not self.object_picked or self.selected_circle is not None: 
                    # clear selection if not clicking on an object, or was clicking on a vertex circle
                        self.cancel_current_selection()

                    if self.superpixels_on:
                        self.handle_sp_press(event.xdata, event.ydata)

            elif event.button == 3: # right click
                canvas_pos = (event.xdata, event.ydata)
                self.openMenu(canvas_pos)

        else:
            if not self.object_picked or self.selected_circle is not None: 
                # clear selection if not clicking on an object, or was clicking on a vertex circle
                self.cancel_current_selection()


        # else:
        #   self.cancel_current_selection()         

        print self.mode

        self.canvas.draw() # force re-draw

        self.object_picked = False

    # def remove_polygon(self):
    #   # self.selected_polygon.remove()

    #   self.curr_proposal_pathPatch.remove()
    #   self.selected_polygon = None

    #   for circ in self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches']:
    #       circ.remove()

    #   self.accepted_proposals.pop(self.curr_proposal_pathPatch)

    def add_vertex_to_existing_polygon(self, pos):
        from scipy.spatial.distance import cdist

        xys = self.selected_polygon.get_xy()
        xys = xys[:-1] if self.selected_polygon.get_closed() else xys
        dists = np.squeeze(cdist([pos], xys))
        two_neighbor_inds = np.argsort(dists)[:2]

        print two_neighbor_inds

        if min(two_neighbor_inds) == 0 and max(two_neighbor_inds) != 1: # two neighbors are the first point and the last point
            new_vertex_ind = max(two_neighbor_inds) + 1
        else:
            new_vertex_ind = max(two_neighbor_inds)
        xys = np.insert(xys, new_vertex_ind, pos, axis=0)
        self.selected_polygon.set_xy(xys)

        vertex_circle = plt.Circle(pos, radius=UNSELECTED_CIRCLE_SIZE, color=self.boundary_colors[1], alpha=.8)
        self.axis.add_patch(vertex_circle)

        self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches'].insert(new_vertex_ind, vertex_circle)
        self.accepted_proposals[self.curr_proposal_pathPatch]['vertices'] = xys

        # self.all_polygons_vertex_circles[self.curr_freeform_polygon_id].insert(new_vertex_ind, vertex_circle)

        vertex_circle.set_picker(CIRCLE_PICK_THRESH)

        self.canvas.draw()


    def remove_selected_vertex(self):
        self.selected_circle.remove()
        self.accepted_proposals[self.curr_proposal_pathPatch]['vertexPatches'].remove(self.selected_circle)
        self.selected_circle = None

        p = self.curr_proposal_pathPatch

        xys = p.get_xy()
        xys = np.vstack([xys[:self.selected_vertex_index], xys[self.selected_vertex_index+1:]])

        vertices = xys[:-1] if p.get_closed() else xys

        self.curr_proposal_pathPatch.set_xy(vertices)

        self.accepted_proposals[self.curr_proposal_pathPatch]['vertices'] = vertices

        self.canvas.draw()


    def place_vertex(self, x,y):
        self.curr_polygon_vertices.append([x, y])

        # curr_vertex_circle = plt.Circle((x, y), radius=10, color=self.colors[self.curr_label + 1], alpha=.8)
        curr_vertex_circle = plt.Circle((x, y), radius=UNSELECTED_CIRCLE_SIZE, color=self.boundary_colors[1], alpha=.8)
        self.axis.add_patch(curr_vertex_circle)
        self.curr_polygon_vertex_circles.append(curr_vertex_circle)

        curr_vertex_circle.set_picker(CIRCLE_PICK_THRESH)



    ############################################
    # other functions
    ############################################

    def pick_color(self, selected_label):
        pass

    def handle_sp_press(self, x, y):
        print 'clicked'
        self.clicked_sp = self.dm.segmentation[int(y), int(x)]
        sys.stderr.write('clicked sp %d\n'%self.clicked_sp)

        self.cancel_current_selection()

        # self.cancel_curr_global()
        # self.cancel_curr_local()

        if self.mode == Mode.REVIEW_PROPOSAL:
            if self.shuffle_global_proposals:
                self.show_global_proposal_covering_sp(self.clicked_sp)
            else:
                self.show_local_proposal_from_sp(self.clicked_sp)
            # self.show_region(self.clicked_sp)



    def load_segmentation(self):
        sys.stderr.write('loading segmentation...\n')
        self.statusBar().showMessage('loading segmentation...')
        # self.dm.load_multiple_results(results=[
        #   # 'segmentation', 
        #   'edgeEndpoints', 'edgeMidpoints'])
        # self.segmentation = self.dm.load_pipeline_result('segmentation')
        # self.n_superpixels = self.dm.segmentation.max() + 1
        self.seg_loaded = True
        sys.stderr.write('segmentation loaded.\n')

        sys.stderr.write('loading sp props...\n')
        self.statusBar().showMessage('loading sp properties..')
        # self.sp_centroids = self.dm.load_pipeline_result('spCentroids')
        # self.sp_bboxes = self.dm.load_pipeline_result('spBbox')
        sys.stderr.write('sp properties loaded.\n')

        self.statusBar().showMessage('')

        # self.sp_rectlist = [None for _ in range(self.dm.n_superpixels)]


    def turn_superpixels_off(self):
        self.statusBar().showMessage('Supepixels OFF')

        self.buttonSpOnOff.setText('Turn Superpixels ON')

        self.segm_handle.remove()
        self.superpixels_on = False
        
        # self.axis.imshow(self.masked_img, cmap=plt.cm.Greys_r,aspect='equal')
        # self.orig_image_handle = self.axis.imshow(self.masked_img, aspect='equal')

    def turn_superpixels_on(self):
        self.statusBar().showMessage('Supepixels ON')

        self.buttonSpOnOff.setText('Turn Superpixels OFF')

        if self.segm_transparent is None:
            self.segm_transparent = self.dm.load_pipeline_result('segmentationTransparent')
            self.my_cmap = plt.cm.Reds
            self.my_cmap.set_under(color="white", alpha="0")

        if not self.seg_loaded:
            self.load_segmentation()

        self.superpixels_on = True
        
        # self.orig_image_handle.remove()

        self.segm_handle = self.axis.imshow(self.segm_transparent, aspect='equal', 
                                cmap=self.my_cmap, alpha=1.)

        # self.axis.clear()
        
        # self.seg_vis = self.dm.load_pipeline_result('segmentationWithText')
        # # self.seg_vis[~self.dm.mask] = 0
        # self.seg_viz_handle = self.axis.imshow(self.seg_vis, aspect='equal')

        # self.canvas.draw()

    def cancel_current_selection(self):
        if self.curr_proposal_pathPatch is not None:

            # restore line width from 5 to 3
            if self.curr_proposal_pathPatch.get_linewidth() != UNSELECTED_POLYGON_LINEWIDTH:
                self.curr_proposal_pathPatch.set_linewidth(UNSELECTED_POLYGON_LINEWIDTH)

            if self.curr_proposal_pathPatch in self.axis.patches:
                if self.curr_proposal_pathPatch not in self.accepted_proposals:
                    self.curr_proposal_pathPatch.remove()

        self.curr_proposal_pathPatch = None

        if self.selected_circle is not None:
            self.selected_circle.set_radius(UNSELECTED_CIRCLE_SIZE)
            self.selected_circle = None


    def mode_changed(self):

        self.cancel_current_selection()

        if self.radioButton_globalProposal.isChecked():

            self.shuffle_global_proposals = True

            if not self.superpixels_on:
                self.turn_superpixels_on()

            if not hasattr(self, 'global_proposal_tuples'):
                self.load_global_proposals()

        elif self.radioButton_localProposal.isChecked():

            if not self.superpixels_on:
                self.turn_superpixels_on()

            self.shuffle_global_proposals = False

            if not hasattr(self, 'local_proposal_tuples'):
                self.load_local_proposals()

        self.canvas.draw()

    def display_option_changed(self):
        if self.sender() == self.buttonSpOnOff:

            if not self.superpixels_on:
                self.turn_superpixels_on()
            else:
                self.turn_superpixels_off()
        else:
            # if self.under_img is not None:
            #   self.under_img.remove()

            self.axis.clear()

            if self.sender() == self.img_radioButton:

                # self.axis.clear()
                # self.axis.axis('off')

                # self.under_img = self.axis.imshow(self.masked_img, aspect='equal', cmap=plt.cm.Greys_r)
                self.axis.imshow(self.dm.image_rgb_jpg, aspect='equal', cmap=plt.cm.Greys_r)
                # self.superpixels_on = False

            elif self.sender() == self.textonmap_radioButton:

                # self.axis.clear()
                # self.axis.axis('off')

                if self.textonmap_vis is None:
                    self.textonmap_vis = self.dm.load_pipeline_result('texMapViz')

                # if self.under_img is not None:
                #   self.under_img.remove()

                # self.under_img = self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
                self.axis.imshow(self.textonmap_vis, cmap=plt.cm.Greys_r, aspect='equal')
                # self.superpixels_on = False

            elif self.sender() == self.dirmap_radioButton:

                # self.axis.clear()
                # self.axis.axis('off')

                if self.dirmap_vis is None:
                    self.dirmap_vis = self.dm.load_pipeline_result('dirMap', 'jpg')
                    self.dirmap_vis[~self.dm.mask] = 0


                # self.under_img = self.axis.imshow(self.dirmap_vis, aspect='equal')
                self.axis.imshow(self.dirmap_vis, aspect='equal')

                # if not self.seg_loaded:
                #   self.load_segmentation()

                # self.superpixels_on = False

            # elif self.sender() == self.labeling_radioButton:
            #   pass

        self.axis.axis('off')

        self.axis.set_xlim([self.newxmin, self.newxmax])
        self.axis.set_ylim([self.newymin, self.newymax])

        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.canvas.draw()

               
if __name__ == "__main__":
    from sys import argv, exit
    appl = QApplication(argv)

    # if len(sys.argv) == 2:
    stack = sys.argv[1]
    # section = int(sys.argv[2])
    m = BrainLabelingGUI(stack=stack)
    # m = BrainLabelingGUI(stack=stack, section=section)
    # elif len(sys.argv) == 3:
    #   section = int(sys.argv[1])
    #   labeling_name = sys.argv[2]
    #   m = BrainLabelingGUI(stack='MD593', section=section, parent_labeling_name='_'.join(labeling_name.split('_')[2:]))

    # m = BrainLabelingGUI(stack='RS141', section=1)
    # m.setWindowTitle("Brain Labeling")
    m.showMaximized()
    m.raise_()
    exit(appl.exec_())
