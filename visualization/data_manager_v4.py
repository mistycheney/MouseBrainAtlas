from PyQt4 import QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import QSize, QDir
from PyQt4.QtGui import QTableWidget, QHeaderView, QTableWidgetItem, QPixmap, \
    QIcon, QMainWindow, QWidget, QHBoxLayout, QApplication

from brain_labelling_gui_v9 import BrainLabelingGUI
from ui_param_settings_v2 import Ui_ParameterSettingsWindow
from ui_DataManager_v4 import Ui_DataManager
from preview_widget import PreviewerWidget

import os
import sys
import cPickle as pickle
from operator import itemgetter

sys.path.append(os.path.realpath('../notebooks'))
from utilities import DataManager

from operator import itemgetter
from collections import defaultdict
import itertools

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
        
        self.stack_model = QStandardItemModel()

        for stack_info in self.dm.local_ds['stacks']:
            item = QStandardItem(stack_info['name'] + ' (%d sections)' % stack_info['section_num'])
            self.stack_model.appendRow(item)

        self.stack_list.setModel(self.stack_model)
        self.stack_list.clicked.connect(self.on_stacklist_clicked_images)
        self.previewer = PreviewerWidget()

        self.topLayout.addWidget(self.previewer)
        
        self.buttonParams.clicked.connect(self.paramSettings_clicked)
        # self.buttonR.clicked.connect(self.refresh_clicked)
        self.buttonQuit.clicked.connect(self.exit_clicked)

        self.setCentralWidget(self.cWidget)

        self.labeling_guis = []

        self.actionLabeling.triggered.connect(self.switch_to_labeling)

    def switch_to_labeling(self, event):
        self.stack_list.clicked.connect(self.on_stacklist_clicked_labelings)

        self.section_model = QStandardItemModel()
        self.section_list.setModel(self.section_model)
        self.section_list.clicked.connect(self.on_sectionlist_clicked_labelings)

        self.label_toshow = 3

        self.statusBar().showMessage('Only showing labelings with label %d (%s)' % (self.label_toshow,
                                                    self.dm.labelnames[self.label_toshow]))

        self.preview_caption_tuples = [(labeling['previewpath'], labeling['filename']) 
                                for labeling in self.dm.inv_labeing_index[self.label_toshow]]

        labeling_tuples = [labeling_filename[:-4].split('_') 
                            for labeling_path, labeling_filename in self.preview_caption_tuples]

        self.d = defaultdict(lambda: defaultdict(list))
        for i, (stack, section_str, user, timestamp) in enumerate(labeling_tuples):
            self.d[stack][int(section_str)].append(i)

        self.stack_model.clear()
        for stack_name in self.d:
            item = QStandardItem(stack_name)
            self.stack_model.appendRow(item)

        self.previewer.set_images(imgs=self.preview_caption_tuples, callback=self.process_labeling_selected)

    def on_stacklist_clicked_labelings(self, list_index):
        self.selected_stack_name = str(list_index.data().toString()).split()[0]
        self.sections = self.d[self.selected_stack_name].keys()

        self.dm.set_stack(self.selected_stack_name)

        self.section_model.clear()
        for section_ind in self.sections:
            item = QStandardItem('%04d'%section_ind)
            self.section_model.appendRow(item)

        indices_toshow = list(itertools.chain.from_iterable(self.d[self.selected_stack_name].values()))
        tuples_toshow = [self.preview_caption_tuples[i] for i in indices_toshow]
        self.previewer.set_images(imgs=tuples_toshow, callback=self.process_labeling_selected)

    def on_sectionlist_clicked_labelings(self, list_index):
        self.selected_section = int(str(list_index.data().toString()))
        indices_toshow = self.d[self.selected_stack_name][self.selected_section]
        tuples_toshow = [self.preview_caption_tuples[i] for i in indices_toshow]
        self.previewer.set_images(imgs=tuples_toshow, callback=self.process_labeling_selected)


    def on_stacklist_clicked_images(self, list_index):
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

        # list of (file path, caption) tuples
        previews_path_caption = []

        # add default "new labeling"
        newLabeling_name = 'new labeling'

        previews_path_caption.append((self.dm.image_path, newLabeling_name))
        # add human labelings if there is any
        if 'labelings' in self.dm.section_info:
        
            for labeling in self.dm.section_info['labelings']:
                labeling_name = labeling['filename']

                stack, section, user, timestamp = labeling_name[:-4].split('_')

                preview_path = self.dm._load_labeling_preview_path(labeling_name='_'.join([user, timestamp]))
                previews_path_caption.append((preview_path, labeling_name))

        print previews_path_caption
        self.previewer.set_images(imgs=previews_path_caption, callback=self.process_labeling_selected)

    def process_labeling_selected(self, labeling_index):

        self.labeling_name = self.previewer.captions[labeling_index]

        if self.labeling_name != 'new labeling':
            labeling_gui = BrainLabelingGUI(parent_labeling_name=self.labeling_name)
        else:
            labeling_gui = BrainLabelingGUI(stack=self.dm.stack, section=self.dm.slice_ind)

        self.labeling_guis.append(labeling_gui)

        # labeling_gui.exec_()

    def paramSettings_clicked(self):
        self.paramsForm = ParamSettingsForm()
        self.paramsForm.show()

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
