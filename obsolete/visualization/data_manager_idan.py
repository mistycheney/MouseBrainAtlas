from PyQt4 import QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import QSize, QDir
from PyQt4.QtGui import QTableWidget, QHeaderView, QTableWidgetItem, QPixmap, \
    QIcon, QMainWindow, QWidget, QHBoxLayout, QApplication
from Tkdnd import Icon


SPACING = 7
FIXED_WIDTH = 1600
FIXED_HEIGHT = 970
THUMB_WIDTH = 300
INITIAL_FOLDER = "/home/yuncong/BrainLocal/DavidData_v4"
 
class PreviewerWidget(QWidget):
    def __init__(self, parent, picturesPath):

        print '1', picturesPath

        QWidget.__init__(self, parent)
        
        # Create client of photos previewer
        client = QWidget()
        
        # Fixed size, spacing, width
        self.setFixedWidth(FIXED_WIDTH)
        self.layout = QtGui.QGridLayout(client)
        self.layout.setHorizontalSpacing(SPACING)
        self.layout.setVerticalSpacing(SPACING)
        
        # Current folder of images
        pictureDirs = QDir(picturesPath)
        pictureDirs = pictureDirs.entryList(filters=QDir.Dirs | QDir.NoDotAndDotDot) 
 
        img_per_row = (FIXED_WIDTH + SPACING) / THUMB_WIDTH
 
        count = 0
        for picDir in pictureDirs:

            print picturesPath

            picDir = QDir(picturesPath + "/" + picDir)
            pictures = picDir.entryList(['*.jpg', '*.png', '*.gif', '*.tif'], sort=QDir.Size)
            if (len(pictures) == 0):
                continue
            
            picture = pictures[0]
            
            i = count / img_per_row
            j = count % img_per_row
            count = count + 1
             
            textPart = (picDir.dirName() + "/" + picture)
            toolTip = picturesPath + "/" + textPart
            if (len(textPart) > 50):
                textPart = '<-- ' + textPart[-50::]
                print textPart
            actionLoad = QAction(QIcon(picDir.path() + "/" + picture), 
                                 textPart, self);
            actionLoad.setToolTip(toolTip)
                                 
            
            # Create a tool button. Load button and recent files will be added as a drop down menu
            button = QToolButton(self)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon);

            # Action = image + text
            button.addAction(actionLoad)
            button.setDefaultAction(actionLoad);
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setIconSize(QSize(THUMB_WIDTH, 180))
            
            button.clicked.connect(self.folder_clicked)

            # add to the layout
            self.layout.addWidget(button, i, j)
    
        self.layout.setAlignment(Qt.AlignTop)
        client.setLayout(self.layout)
    
        # Enable scrolling    
        scroll = QtGui.QScrollArea(self)
        scroll.setFixedWidth(FIXED_WIDTH)
        scroll.setFixedHeight(FIXED_HEIGHT)
        scroll.setWidget(client) 
        
        # Folder for current selected directory
        self.selectedFolder = {}
        
    
    # Folder chosen event
    def folder_clicked(self):
        button = self.sender()
        if (self.selectedFolder != {}):
            self.selectedFolder.setDown(False)
        
        button.setDown(True)
        self.selectedFolder = button
                

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////#

class MainWindow(QMainWindow):
    
    def __init__(self, parent=None, **kwargs):
        QMainWindow.__init__(self, parent, **kwargs)

        path = INITIAL_FOLDER # QDir.currentPath - or some other initial path
        
        # Create set of widgets in the central widget window
        cWidget = QWidget()
        self.vLayout = QVBoxLayout(cWidget)
        self.topLayout = QHBoxLayout(self)
        self.bottomLayout = QHBoxLayout(self)
        
        # File System widget
        model = QtGui.QFileSystemModel()
        sub = QDir(path)
        # sub.cdUp()          # Get parent of selected folder
        i = model.setRootPath(sub.path())
        model.setFilter(QDir.AllDirs)
        self.view = QtGui.QListView()
        self.view.setModel(model)
        self.view.setRootIndex(i)
        self.view.clicked.connect(self.on_treeView_clicked)
        
        # Sub-folders widget
        folders = QtGui.QFileSystemModel()
        i = folders.setRootPath(path)
        folders.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        self.fview = QtGui.QListView()
        self.fview.setModel(folders)
        self.fview.setRootIndex(i)
        self.fview.clicked.connect(self.on_ftreeView_clicked)
       
        # Folder preview widget
        self.previewer = PreviewerWidget(self, INITIAL_FOLDER)
        
        # Add both widgets to gadget
        self.topLayout.addWidget(self.view)
        self.topLayout.addWidget(self.fview)
        self.topLayout.addWidget(self.previewer)

        # Bottom buttons
        self.bottomLayout.addStretch();
        self.buttonS = QPushButton("Parameter Settings", self)
        self.buttonR = QPushButton("Refresh", self)
        self.buttonQ = QPushButton("Quit", self)
        
        # Bind buttons presses
        self.buttonS.clicked.connect(self.pref_clicked)
        self.buttonR.clicked.connect(self.refresh_clicked)
        self.buttonQ.clicked.connect(self.exit_clicked)

        # Add buttons to widget
        self.bottomLayout.addWidget(self.buttonS);
        self.bottomLayout.addWidget(self.buttonR);
        self.bottomLayout.addWidget(self.buttonQ);

        # Set topLayout of widget as horizontal
        #cWidget.setLayout(self.topLayout)
        self.vLayout.addLayout(self.topLayout)
        self.vLayout.addLayout(self.bottomLayout)
        self.setCentralWidget(cWidget)
        
        
    # Helper function to refresh file browsers
    def set_right_indices(self, indexL, indexR):
        modelL = self.view.model()
        modelR = self.fview.model()
        
        if (indexL == 0):
            pathL = modelL.rootPath()
        else:
            pathL = modelL.filePath(indexL)
            
        pathR = modelR.filePath(indexR)
        
        l = modelL.setRootPath(pathL)
        r = modelR.setRootPath(pathR)
        
        self.view.setRootIndex(l)
        self.fview.setRootIndex(r)
        
        self.refresh_preview(pathR)
        
    
    # Folders preview kill-create refresh event
    def refresh_preview(self, path):
        # Folder preview widget
        self.topLayout.removeWidget(self.previewer)
        self.previewer.destroy()
        self.previewer = PreviewerWidget(self, path)
        self.topLayout.addWidget(self.previewer)
        

    def on_treeView_clicked(self, index):
        if (index.row() == 1):  # ".." clicked
            indexL = self.view.rootIndex()
            self.set_right_indices(index, indexL)
        else:
            dPath = self.view.model().filePath(index)
            i = self.fview.model().setRootPath(dPath)
            self.set_right_indices(0, i)
        
        
    def on_ftreeView_clicked(self, index):
        dPath = self.fview.model().rootPath()
        i = self.view.model().setRootPath(dPath)
        self.set_right_indices(i, index)
    
    
    # WIP 
    def pref_clicked(self):
        self.error = QErrorMessage()
        self.error.setWindowTitle("Attention")
        if (self.previewer.selectedFolder == {}):
            self.error.showMessage("Please check a data folder first.")
        else:
            self.error.showMessage("WIP") 
        self.error.show()
        
    
    # Refresh button kills-creates new preview
    def refresh_clicked(self):
        self.refresh_preview(self.fview.model().rootPath())
        
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
