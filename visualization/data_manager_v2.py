from PyQt4 import QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import QSize, QDir
from PyQt4.QtGui import QTableWidget, QHeaderView, QTableWidgetItem, QPixmap, \
    QIcon, QMainWindow, QWidget, QHBoxLayout, QApplication
# from Tkdnd import Icon
import os
import cPickle as pickle
from visualization_utilities import *

SPACING = 7
FIXED_WIDTH = 1600
FIXED_HEIGHT = 970
THUMB_WIDTH = 300

data_dir = os.environ['LOCAL_DATA_DIR']

import sip

class PreviewerWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.scroll = QtGui.QScrollArea(self)
        self.scroll.setFixedWidth(FIXED_WIDTH)
        self.scroll.setFixedHeight(FIXED_HEIGHT)

        self.setFixedWidth(FIXED_WIDTH)

        self.client = None

    def set_imgs(self, imgs=[]):

        if self.client is not None:
            for a in self.actions:
                sip.delete(a)

            for w in self.buttons:
                self.layout.removeWidget(w)
                sip.delete(w)

        self.client = QWidget()

        self.layout = QtGui.QGridLayout(self.client)
        self.layout.setHorizontalSpacing(SPACING)
        self.layout.setVerticalSpacing(SPACING)
        self.layout.setAlignment(Qt.AlignTop)

        img_per_row = (FIXED_WIDTH + SPACING) / THUMB_WIDTH
 
        self.actions = []
        self.buttons = []

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

            self.buttons.append(button)
            self.actions.append(actionLoad)

            i = count / img_per_row
            j = count % img_per_row

            self.layout.addWidget(button, i, j)
    
        self.client.setLayout(self.layout)
        self.scroll.setWidget(self.client)

        self.selected_image = None
        
    # Folder chosen event
    def image_clicked(self):
        image = self.sender()
        
        if self.selected_image is not None:
            self.selected_image.setDown(False)
        
        image.setDown(True)
        self.selected_image = image
                

class MainWindow(QMainWindow):
    
    def __init__(self, parent=None, **kwargs):
        QMainWindow.__init__(self, parent, **kwargs)
        
        # Create set of widgets in the central widget window
        self.cWidget = QWidget()
        self.vLayout = QVBoxLayout(self.cWidget)
        self.leftListLayout = QVBoxLayout()
        self.topLayout = QHBoxLayout()
        self.bottomLayout = QHBoxLayout()
        
        # remote_ds = pickle.load(open('remote_directory_structure.pkl', 'r'))

        self.local_ds = generate_json(data_dir)

        self.stack_model = QStandardItemModel()
        
        self.labeling_model = QStandardItemModel()

        self.stack_names = []
        for stack_info in self.local_ds:
            item = QStandardItem(stack_info['name'] + ' (%d sections)' % len(stack_info['sections']))
            self.stack_model.appendRow(item)
            self.stack_names.append(stack_info['name'])

        self.stack_list = QtGui.QListView()
        self.stack_list.setModel(self.stack_model)
        self.stack_list.clicked.connect(self.on_stacklist_clicked)
                
        self.section_list = QtGui.QListView()

        self.section_model = QStandardItemModel(self.section_list)

        self.section_list.setModel(self.section_model)
        self.section_list.clicked.connect(self.on_sectionlist_clicked)

        # self.labeling_list = QtGui.QListView()
        # self.labeling_list.setModel(self.labeling_model)
        # self.labeling_list.clicked.connect(self.on_labelinglist_clicked)

        self.previewer = PreviewerWidget()

        self.leftListLayout.addWidget(self.stack_list)
        self.leftListLayout.addWidget(self.section_list)
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
        self.buttonS.clicked.connect(self.pref_clicked)
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


    def on_stacklist_clicked(self, index):
        row = index.row()
        # self.stack_name = str(index.data().toString())
        self.stack_name = self.stack_names[row]
        stack_path = os.path.join(data_dir, self.stack_name)

        # self.stack_info = [s for s in self.local_ds if s['name'] == self.stack_name][0]
        self.stack_info = self.local_ds[row]

        self.section_model.clear()

        imgs_path_text = []
        self.section_names = []
        for section_info in self.stack_info['sections']:
            section_str = '%04d'%section_info['index']
            self.section_names.append(section_str)
            if 'labelings' in section_info:
                item = QStandardItem(section_str + ' (%d labelings)' % len(section_info['labelings']))
            else:
                item = QStandardItem(section_str + ' (0 labelings)')
            self.section_model.appendRow(item)

            img_filename = os.path.join(stack_path, 'x5', section_str, '_'.join([self.stack_name, 'x5', section_str]) + '.tif')
            imgs_path_text.append((img_filename, section_str))

        self.previewer.set_imgs(imgs=imgs_path_text)

    def on_sectionlist_clicked(self, index):
        row = index.row()
        # self.section_name = str(index.data().toString())
        self.section_name = self.section_names[row]
        section_path = os.path.join(data_dir, self.stack_name, 'x5', self.section_name)
        
        self.section_info = self.stack_info['sections'][row]

        self.labeling_model.clear()

        previews_path_text = []
        if 'labelings' in self.section_info:
            for labeling_name in self.section_info['labelings']:
                item = QStandardItem(labeling_name)
                self.labeling_model.appendRow(item)

                preview_filename = os.path.join(section_path, 'labelings', labeling_name[:-4] + '.tif')
                previews_path_text.append((preview_filename, labeling_name))

        self.previewer.set_imgs(imgs=previews_path_text)

    # def on_labelinglist_clicked(self, index):
    #     self.labeling_name = str(index.data().toString())
    #     labeling_path = os.path.join(data_dir, self.stack_name, self.section_name, 'labelings', self.labeling_name)
    #     labeling_preview_path = os.path.join(data_dir, self.stack_name, self.section_name, 'labelings', self.labeling_name)






    # WIP 
    def pref_clicked(self):
        self.error = QErrorMessage()
        self.error.setWindowTitle("Attention")
        if (self.previewer.selectedFolder == {}):
            self.error.showMessage("Please check a data folder first.")
        else:
            self.error.showMessage("WIP") 
        self.error.show()
        
    
    # # Refresh button kills-creates new preview
    # def refresh_clicked(self):
    #     self.refresh_preview(self.fview.model().rootPath())
        
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
