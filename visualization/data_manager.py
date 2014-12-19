from PyQt4 import QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import QSize, QDir
from PyQt4.QtGui import QTableWidget, QHeaderView, QTableWidgetItem, QPixmap, \
    QIcon, QMainWindow, QWidget, QHBoxLayout, QApplication
# from Tkdnd import Icon
import os

SPACING = 7
FIXED_WIDTH = 1600
FIXED_HEIGHT = 970
THUMB_WIDTH = 300
# INITIAL_FOLDER = "/home/yuncong/BrainLocal/DavidData_v4"

class PreviewerWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.scroll = QtGui.QScrollArea(self)
        self.scroll.setFixedWidth(FIXED_WIDTH)
        self.scroll.setFixedHeight(FIXED_HEIGHT)

        self.setFixedWidth(FIXED_WIDTH)

    def set_imgpath(self, imgpath=None):

        if imgpath is None:
            return

        self.client = QWidget()

        self.layout = QtGui.QGridLayout(self.client)
        self.layout.setHorizontalSpacing(SPACING)
        self.layout.setVerticalSpacing(SPACING)
        self.layout.setAlignment(Qt.AlignTop)

        # Current folder of images
        pictureDirs = QDir(imgpath)
        pictureDirs = pictureDirs.entryList(filters=QDir.Dirs | QDir.NoDotAndDotDot) 
 
        img_per_row = (FIXED_WIDTH + SPACING) / THUMB_WIDTH
 
        count = 0
        for picDir in pictureDirs:

            picDir = QDir(imgpath + "/" + picDir)
            pictures = picDir.entryList(['*.jpg', '*.png', '*.gif', '*.tif'], sort=QDir.Size)
            
            if (len(pictures) == 0):
                continue
            
            picture = pictures[0]
            
            i = count / img_per_row
            j = count % img_per_row
            count = count + 1

            actionLoad = QAction(QIcon(picDir.path() + "/" + picture), 
                                 picDir.dirName(), self);
            # actionLoad.setToolTip(toolTip)
            
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon);

            # Action = image + text
            button.addAction(actionLoad)
            button.setDefaultAction(actionLoad);
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setIconSize(QSize(THUMB_WIDTH, 180))
            
            button.clicked.connect(self.folder_clicked)

            # add to the layout
            self.layout.addWidget(button, i, j)
    
        self.client.setLayout(self.layout)
        self.scroll.setWidget(self.client) 

        # Folder for current selected directory
        self.selectedFolder = {}
        
    
    # Folder chosen event
    def folder_clicked(self):
        button = self.sender()
        if (self.selectedFolder != {}):
            self.selectedFolder.setDown(False)
        
        button.setDown(True)
        self.selectedFolder = button
                

class MainWindow(QMainWindow):
    
    def __init__(self, parent=None, **kwargs):
        QMainWindow.__init__(self, parent, **kwargs)
        
        # Create set of widgets in the central widget window
        self.cWidget = QWidget()
        self.vLayout = QVBoxLayout(self.cWidget)
        self.topLayout = QHBoxLayout()
        self.bottomLayout = QHBoxLayout()
        
        with open('data_list.txt') as f:
            contents = f.readlines()
        data_dir = contents[0].strip()
        data_list = [row.strip().split() for row in contents[1:]]

        print data_dir

        root_dir = QDir(data_dir)

        self.stack_model = QtGui.QFileSystemModel()
        i = self.stack_model.setRootPath(root_dir.path())
        self.stack_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        self.stack_list = QtGui.QListView()
        self.stack_list.setModel(self.stack_model)
        self.stack_list.setRootIndex(i)
        self.stack_list.clicked.connect(self.on_stacklist_clicked)
        
        # self.resol_model = QtGui.QFileSystemModel()
        # i = self.resol_model.setRootPath(root_dir.path())
        # self.resol_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        # self.resol_list = QtGui.QListView()
        # self.resol_list.setModel(self.resol_model)
        # self.resol_list.setRootIndex(i)
        # self.resol_list.clicked.connect(self.on_resollist_clicked)
        
        self.previewer = PreviewerWidget()

        # Add both widgets to gadget
        self.topLayout.addWidget(self.stack_list)
        # self.topLayout.addWidget(self.resol_list)
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
        self.stack_name = self.stack_model.fileName(index)
        stack_path = self.stack_model.filePath(index)

        preview_path = os.path.join(str(stack_path), 'x5')

        # i = self.resol_model.setRootPath(stack_path)
        # self.resol_list.setRootIndex(i)

        self.previewer.set_imgpath(imgpath=preview_path)

        
    # def on_resollist_clicked(self, index):
    #     self.resol_name = self.resol_model.fileName(index)
    #     resol_path = self.resol_model.filePath(index)


        # self.topLayout.removeWidget(self.previewer)

        # self.previewer = PreviewerWidget(imgpath=resol_path)

        # self.topLayout.addWidget(self.previewer)

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
