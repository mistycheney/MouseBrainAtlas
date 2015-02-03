from PyQt4 import QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import QSize, QDir
from PyQt4.QtGui import QTableWidget, QHeaderView, QTableWidgetItem, QPixmap, \
    QIcon, QMainWindow, QWidget, QHBoxLayout, QApplication

from brain_labelling_gui_v9 import BrainLabelingGUI
from ui_param_settings_v2 import Ui_ParameterSettingsWindow
from ui_DataManager_v3 import Ui_DataManager
from preview_widget import PreviewerWidget

import os
import sys
import cPickle as pickle
from operator import itemgetter

sys.path.append(os.path.realpath('../notebooks'))
from utilities import DataManager

class ParamSettingsForm(QtGui.QWidget, Ui_ParameterSettingsWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)

class DataManagerGui(QMainWindow, Ui_DataManager):
    
    def __init__(self, parent=None, **kwargs):
        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent, **kwargs)

        self.setupUi(self)
        
        self.dm = DataManager(data_dir=os.environ['LOCAL_DATA_DIR'], 
            repo_dir=os.environ['LOCAL_REPO_DIR'],
            result_dir=os.environ['LOCAL_RESULT_DIR'], 
            labeling_dir=os.environ['LOCAL_LABELING_DIR'])

        self.gabor_params_id='blueNisslWide'
        self.segm_params_id='blueNisslRegular'
        self.vq_params_id='blueNissl'
        
        self.stack_model = QStandardItemModel()
        self.labeling_model = QStandardItemModel()

        for stack_info in self.dm.local_ds['stacks']:
            item = QStandardItem(stack_info['name'] + ' (%d sections)' % stack_info['section_num'])
            self.stack_model.appendRow(item)

        self.stack_list.setModel(self.stack_model)
        self.stack_list.clicked.connect(self.on_stacklist_clicked)
        self.previewer = PreviewerWidget()

        self.topLayout.addWidget(self.previewer)
        
        self.buttonParams.clicked.connect(self.paramSettings_clicked)
        # self.buttonR.clicked.connect(self.refresh_clicked)
        self.buttonQuit.clicked.connect(self.exit_clicked)

        self.setCentralWidget(self.cWidget)

        self.labeling_guis = []

    def on_stacklist_clicked(self, list_index):
        # selected_stack_index = list_index.row()
        self.stack_name = str(list_index.data().toString()).split()[0]
        # self.stack_name = self.dm.local_ds['available_stack_names'][selected_stack_index]
        self.dm.set_stack(self.stack_name)
        
        try:
            self.dm.set_resol('x1.25')
        except Exception as e:
            print e
            self.previewer.set_images(imgs=[], callback=self.process_section_selected)
            return

        imgs_path_caption = []
        
        for section in self.dm.sections_info:

            self.dm.set_slice(section['index'])

            # self.section_names.append(self.dm.slice_str)

            if 'labelings' in section:
                caption = self.dm.slice_str + ' (%d labelings)' % len(section['labelings'])
            else:
                caption = self.dm.slice_str + ' (0 labelings)'
            
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
        
            for labeling in self.dm.section_info['labelings']:
                labeling_name = labeling['filename']
                self.labeling_names.append(labeling_name)
                item = QStandardItem(labeling_name)
                self.labeling_model.appendRow(item)

                stack, section, user, timestamp = labeling_name[:-4].split('_')

                preview_path = self.dm._load_labeling_preview_path(labeling_name='_'.join([user, timestamp]))
                previews_path_caption.append((preview_path, labeling_name))

        print previews_path_caption
        self.previewer.set_images(imgs=previews_path_caption, callback=self.process_labeling_selected)


    def process_labeling_selected(self, labeling_index):

        self.labeling_name = self.labeling_names[labeling_index]

        self.dm.set_resol('x5')
        self.dm._load_image()

        self.dm.set_gabor_params(gabor_params_id=self.gabor_params_id)
        self.dm.set_segmentation_params(segm_params_id=self.segm_params_id)
        self.dm.set_vq_params(vq_params_id=self.vq_params_id)


        if self.labeling_name != 'new labeling':
            stack, section, user, timestamp = self.labeling_name[:-4].split('_')
            labeling_gui = BrainLabelingGUI(dm=self.dm, parent_labeling_name='_'.join([user, timestamp]))
        else:
            labeling_gui = BrainLabelingGUI(dm=self.dm)

        self.labeling_guis.append(labeling_gui)

        # labeling_gui.exec_()

    def paramSettings_clicked(self):
        self.paramsForm = ParamSettingsForm()
        self.paramsForm.show()

        self.gabor_params_id='blueNisslWide'
        self.segm_params_id='blueNisslRegular'
        self.vq_params_id='blueNissl'


    def exit_clicked(self): 
        exit()

               
if __name__ == "__main__":
    from sys import argv, exit

    a = QApplication(argv)
    m = DataManagerGui()
    m.setWindowTitle("Data Manager")
    m.showMaximized()
    m.raise_()
    exit(a.exec_())
