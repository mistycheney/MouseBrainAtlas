from PyQt4 import QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import QSize, QDir
from PyQt4.QtGui import QTableWidget, QHeaderView, QTableWidgetItem, QPixmap, \
    QIcon, QMainWindow, QWidget, QHBoxLayout, QApplication

import sip

# from ui_CrossReferenceGallery import Ui_CrossReferenceGallery

# from operator import itemgetter
# from collections import defaultdict
# import itertools

class PreviewerWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)


        self.SPACING = 7
        self.FIXED_WIDTH = 1600
        self.FIXED_HEIGHT = 1000
        self.THUMB_WIDTH = 300

        self.scroll = QtGui.QScrollArea(self)
        self.scroll.setFixedWidth(self.FIXED_WIDTH)
        self.scroll.setFixedHeight(self.FIXED_HEIGHT)

        self.setFixedWidth(self.FIXED_WIDTH)

        self.client = None

    def set_images(self, path_caption_tuples=[], callback=None):
        """
        callback takes an integer index as argument
        """

        self.callback = callback

        self.imagepath_caption_tuples = path_caption_tuples
        
        if len(path_caption_tuples) > 0:
            self.imagepaths, self.captions = zip(*path_caption_tuples)
        else:
            self.imagepaths = []
            self.captions = []

        if self.client is not None:
            # assert len(self.actions) > 0
            # assert len(self.thumbnail_buttons) > 0

            for a in self.actions:
                sip.delete(a)

            for w in self.thumbnail_buttons:
                self.layout.removeWidget(w)
                sip.delete(w)

        self.client = QWidget()

        self.layout = QtGui.QGridLayout(self.client)
        self.layout.setHorizontalSpacing(self.SPACING)
        self.layout.setVerticalSpacing(self.SPACING)
        self.layout.setAlignment(Qt.AlignTop)

        img_per_row = (self.FIXED_WIDTH + self.SPACING) / self.THUMB_WIDTH
 
        self.actions = []
        self.thumbnail_buttons = []

        for count, (img_filename, img_text) in enumerate(path_caption_tuples):
                                
            actionLoad = QAction(QIcon(img_filename), img_text, self)
            # actionLoad.setToolTip(toolTip)
            
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

            # Action = image + text
            button.addAction(actionLoad)
            button.setDefaultAction(actionLoad);
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setIconSize(QSize(self.THUMB_WIDTH, 180))
            
            button.clicked.connect(self.image_clicked)

            self.thumbnail_buttons.append(button)
            self.actions.append(actionLoad)

            i = count / img_per_row
            j = count % img_per_row

            self.layout.addWidget(button, i, j)
    
        self.client.setLayout(self.layout)
        self.scroll.setWidget(self.client)

        self.selected_image = None

    # Folder chosen event
    def image_clicked(self):
        thumbnail_clicked = self.sender()
        thumbnail_clicked.setDown(True)

        index_clicked = self.thumbnail_buttons.index(thumbnail_clicked)
        
        self.callback(index_clicked)


# class CrossReferenceGui(QMainWindow, Ui_CrossReferenceGallery):
    
#     def __init__(self, parent=None, **kwargs):
#         QMainWindow.__init__(self, parent, **kwargs)

#         self.setupUi(self)

#         self.stack_model = QStandardItemModel()
#         self.stack_list.setModel(self.stack_model)
#         self.stack_list.clicked.connect(self.on_stacklist_clicked)

#         self.section_model = QStandardItemModel()
#         self.section_list.setModel(self.section_model)
#         self.section_list.clicked.connect(self.on_sectionlist_clicked)

#         self.previewer = PreviewerWidget()

#         self.topLayout.addWidget(self.previewer)
#         self.buttonQuit.clicked.connect(self.exit_clicked)
#         self.setCentralWidget(self.cWidget)

#         self.show()
    
#     def set_images(self, imgs=[], callback=None):
#         self.callback = callback

#         self.previewer.set_images(imgs, callback)

#         self.preview_caption_tuples = imgs
#         self.labeling_tuples = [labeling_filename[:-4].split('_') for labeling_path, labeling_filename in imgs]
#         self.d = defaultdict(lambda: defaultdict(list))
#         for i, (stack, section_str, user, timestamp) in enumerate(self.labeling_tuples):
#             self.d[stack][int(section_str)].append(i)

#         for stack_name in self.d:
#             item = QStandardItem(stack_name)
#             self.stack_model.appendRow(item)

#     def on_stacklist_clicked(self, list_index):
#         self.selected_stack_name = str(list_index.data().toString()).split()[0]
#         self.sections = self.d[self.selected_stack_name].keys()

#         for section_ind in self.sections:
#             item = QStandardItem('%04d'%section_ind)
#             self.section_model.appendRow(item)

#         indices_toshow = list(itertools.chain.from_iterable(self.d[self.selected_stack_name].values()))
#         tuples_toshow = [self.preview_caption_tuples[i] for i in indices_toshow]
#         self.previewer.set_images(imgs=tuples_toshow, callback=self.callback)

#     def on_sectionlist_clicked(self, list_index):
#         self.selected_section = int(str(list_index.data().toString()))
#         indices_toshow = self.d[self.selected_stack_name][self.selected_section]
#         tuples_toshow = [self.preview_caption_tuples[i] for i in indices_toshow]
#         self.previewer.set_images(imgs=tuples_toshow, callback=self.callback)

#     def exit_clicked(self): 
#         exit()