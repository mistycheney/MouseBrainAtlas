from PyQt4 import QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import QSize, QDir
from PyQt4.QtGui import QTableWidget, QHeaderView, QTableWidgetItem, QPixmap, \
    QIcon, QMainWindow, QWidget, QHBoxLayout, QApplication
# from Tkdnd import Icon
import os
import cPickle as pickle
# from visualization_utilities import *

from brain_labelling_gui_v9 import BrainLabelingGUI
from ui_param_settings_v2 import Ui_ParameterSettingsWindow
from operator import itemgetter

import sys
sys.path.append(os.path.realpath('../notebooks'))
from utilities import *

SPACING = 7
FIXED_WIDTH = 1600
FIXED_HEIGHT = 970
THUMB_WIDTH = 300

data_dir = os.environ['LOCAL_DATA_DIR']

import sip

class ParamSettingsForm(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_ParameterSettingsWindow()
        self.ui.setupUi(self)


class PreviewerWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.scroll = QtGui.QScrollArea(self)
        self.scroll.setFixedWidth(FIXED_WIDTH)
        self.scroll.setFixedHeight(FIXED_HEIGHT)

        self.setFixedWidth(FIXED_WIDTH)

        self.client = None

    def set_images(self, imgs=[], callback=None):
        """
        callback takes an integer index as argument
        """

        self.callback = callback

        if self.client is not None:
            assert len(self.actions) > 0
            assert len(self.thumbnail_buttons) > 0

            for a in self.actions:
                sip.delete(a)

            for w in self.thumbnail_buttons:
                self.layout.removeWidget(w)
                sip.delete(w)

        self.client = QWidget()

        self.layout = QtGui.QGridLayout(self.client)
        self.layout.setHorizontalSpacing(SPACING)
        self.layout.setVerticalSpacing(SPACING)
        self.layout.setAlignment(Qt.AlignTop)

        img_per_row = (FIXED_WIDTH + SPACING) / THUMB_WIDTH
 
        self.actions = []
        self.thumbnail_buttons = []

        for count, (img_filename, img_text) in enumerate(imgs):
                                
            actionLoad = QAction(QIcon(img_filename), img_text, self)
            # actionLoad.setToolTip(toolTip)
            
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

            # Action = image + text
            button.addAction(actionLoad)
            button.setDefaultAction(actionLoad);
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setIconSize(QSize(THUMB_WIDTH, 180))
            
            button.clicked.connect(self.image_clicked)

            self.thumbnail_buttons.append(button)
            self.actions.append(actionLoad)

            i = count / img_per_row
            j = count % img_per_row

            self.layout.addWidget(button, i, j)
    
        self.client.setLayout(self.layout)
        self.scroll.setWidget(self.client)

        self.selected_image = None

    # def set_toplevel(self, g):
    #     self.top = g

    # Folder chosen event
    def image_clicked(self):
        thumbnail_clicked = self.sender()
        thumbnail_clicked.setDown(True)

        index_clicked = self.thumbnail_buttons.index(thumbnail_clicked)
        
        self.callback(index_clicked)


class MainWindow(QMainWindow):
    
    def __init__(self, parent=None, **kwargs):
        QMainWindow.__init__(self, parent, **kwargs)
        
        self.dm = DataManager(data_dir=os.environ['LOCAL_DATA_DIR'], 
            repo_dir=os.environ['LOCAL_REPO_DIR'],
            result_dir=os.environ['LOCAL_RESULT_DIR'], 
            labeling_dir=os.environ['LOCAL_LABELING_DIR'])

        # Create set of widgets in the central widget window
        self.cWidget = QWidget()
        self.vLayout = QVBoxLayout(self.cWidget)
        self.leftListLayout = QVBoxLayout()
        self.topLayout = QHBoxLayout()
        self.bottomLayout = QHBoxLayout()
        
        # remote_ds = pickle.load(open('remote_directory_structure.pkl', 'r'))

        self.stack_model = QStandardItemModel()
        
        self.labeling_model = QStandardItemModel()

        for stack_info in self.dm.local_ds['stacks']:
            item = QStandardItem(stack_info['name'] + ' (%d sections)' % stack_info['section_num'])
            self.stack_model.appendRow(item)

        self.stack_list = QtGui.QListView()
        self.stack_list.setModel(self.stack_model)
        self.stack_list.clicked.connect(self.on_stacklist_clicked)
                
        # self.section_list = QtGui.QListView()

        # self.section_model = QStandardItemModel(self.section_list)

        # self.section_list.setModel(self.section_model)
        # self.section_list.clicked.connect(self.on_sectionlist_clicked)

        # self.labeling_list = QtGui.QListView()
        # self.labeling_list.setModel(self.labeling_model)
        # self.labeling_list.clicked.connect(self.on_labelinglist_clicked)

        self.previewer = PreviewerWidget()
        # self.previewer.set_toplevel(self)

        self.leftListLayout.addWidget(self.stack_list)
        # self.leftListLayout.addWidget(self.section_list)
        # self.leftListLayout.addWidget(self.labeling_list)

        # Add both widgets to gadget
        self.topLayout.addLayout(self.leftListLayout)
        self.topLayout.addWidget(self.previewer)

        # Bottom buttons
        self.bottomLayout.addStretch();
        self.buttonS = QPushButton("Parameter Settings", self)
        # self.buttonR = QPushButton("Refresh", self)
        self.buttonQ = QPushButton("Quit", self)
        
        # Bind buttons presses
        # self.buttonS.clicked.connect(self.pref_clicked)
        # self.buttonR.clicked.connect(self.refresh_clicked)
        self.buttonQ.clicked.connect(self.exit_clicked)

        # Add buttons to widget
        self.bottomLayout.addWidget(self.buttonS);
        # self.bottomLayout.addWidget(self.buttonR);
        self.bottomLayout.addWidget(self.buttonQ);

        # Set topLayout of widget as horizontal
        #cWidget.setLayout(self.topLayout)
        self.vLayout.addLayout(self.topLayout)
        self.vLayout.addLayout(self.bottomLayout)

        self.setCentralWidget(self.cWidget)


    def on_stacklist_clicked(self, list_index):
        # selected_stack_index = list_index.row()
        self.stack_name = str(list_index.data().toString()).split()[0]
        # self.stack_name = self.dm.local_ds['available_stack_names'][selected_stack_index]
        self.dm.set_stack(self.stack_name, 'x1.25')

        # self.section_model.clear()

        imgs_path_caption = []
        # self.section_names = []

        for section_info in self.dm.sections_info:

            self.dm.set_slice(section_info['index'])

            # self.section_names.append(self.dm.slice_str)

            if 'labelings' in section_info:
                caption = self.dm.slice_str + ' (%d labelings)' % len(section_info['labelings'])
            else:
                caption = self.dm.slice_str + ' (0 labelings)'
            
            # sectionList_item = QStandardItem(caption)
            # self.section_model.appendRow(sectionList_item)

            imgs_path_caption.append((self.dm.image_path, caption))

        self.previewer.set_images(imgs=imgs_path_caption, callback=self.process_section_selected)


    def process_section_selected(self, item_index):
        self.dm.set_slice(item_index)

        self.labeling_model.clear()

        # list of (file path, caption) tuples
        previews_path_caption = []

        # add default "new labeling"
        newLabeling_name = 'new labeling'
        newLabeling_item = QStandardItem(newLabeling_name)
        self.labeling_model.appendRow(newLabeling_item)
        previews_path_caption.append((self.dm.image_path, newLabeling_name))
        self.labeling_names = [newLabeling_name]

        # add human labelings if there is any
        if 'labelings' in self.dm.section_info:
            self.labeling_names += self.dm.section_info['labelings']
            for labeling_name in self.labeling_names:
                item = QStandardItem(labeling_name)
                self.labeling_model.appendRow(item)

                preview_path = self.load_labeling_preview(labeling_name[:-4])
                previews_path_caption.append((preview_path, labeling_name))

        self.previewer.set_images(imgs=previews_path_caption, callback=self.process_labeling_selected)


    # def on_sectionlist_clicked(self, list_index):
    #     item_index = list_index.row()
    #     # self.section_name = str(index.data().toString())
    #     self.process_section_selected(item_index)


    def process_labeling_selected(self, labeling_index):

        self.labeling_name = self.labeling_names[labeling_index]

        print labeling_index, self.labeling_name

        self.dm.set_gabor_params(gabor_params_id='blueNisslWide')
        self.dm.set_segmentation_params(segm_params_id='blueNisslRegular')
        self.dm.set_vq_params(vq_params_id='blueNissl')
        self.dm.set_resol('x5')

        self.dm._load_image()


        self.labeling_gui = BrainLabelingGUI(dm=self.dm)
            
    def exit_clicked(self): 
        exit()

               
if __name__ == "__main__":
    from sys import argv, exit

    a = QApplication(argv)
    m = MainWindow()
    m.setWindowTitle("Data Manager")
    m.showMaximized()
    m.raise_()
    exit(a.exec_())
