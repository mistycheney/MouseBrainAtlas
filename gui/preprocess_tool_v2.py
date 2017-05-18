#! /usr/bin/env python

import cPickle as pickle
import sys, os
from subprocess import check_output

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import pandas

from ui.ui_PreprocessGui import Ui_PreprocessGui
from ui.ui_AlignmentGui import Ui_AlignmentGui

from widgets.ZoomableBrowsableGraphicsScene import ZoomableBrowsableGraphicsScene, SimpleGraphicsScene2, SimpleGraphicsScene3, SimpleGraphicsScene4
from widgets.ZoomableBrowsableGraphicsSceneWithReadonlyPolygon import ZoomableBrowsableGraphicsSceneWithReadonlyPolygon
from widgets.MultiplePixmapsGraphicsScene import MultiplePixmapsGraphicsScene
from widgets.DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene
from widgets.SignalEmittingItems import *

from gui_utilities import *
from qt_utilities import *
from DataFeeder import ImageDataFeeder

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *
from preprocess_utilities import *

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'web_services'))
from web_service import WebService

def identify_shape(img_fp):
    return map(int, check_output("identify -format %%Wx%%H %s" % img_fp, shell=True).split('x'))

# Use the third method in http://pyqt.sourceforge.net/Docs/PyQt4/designer.html
class PreprocessGUI(QMainWindow, Ui_PreprocessGui):
    def __init__(self, parent=None, stack=None, tb_fmt='png'):
        """
        Initialization of preprocessing tool.
        """
        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.setupUi(self)

        self.stack = stack
        self.currently_showing = 'original'

        # for fluorescent stack, tif is 16 bit (not visible in GUI), png is 8 bit
        # if self.stack in all_ntb_stacks or self.stack in all_alt_nissl_ntb_stacks:
        self.tb_fmt = tb_fmt
            # self.pad_bg_color = 'black'
        # else:
            # self.tb_fmt = 'tif'
            # self.pad_bg_color = 'white'

        self.stack_data_dir = os.path.join(thumbnail_data_dir, stack)
        # self.stack_data_dir_gordon = os.path.join(gordon_thumbnail_data_dir, stack)

        self.web_service = WebService(server_ip='ec2-52-53-122-62.us-west-1.compute.amazonaws.com')

        ###############################################

        self.slide_gscene = ZoomableBrowsableGraphicsScene(id='section', gview=self.slide_gview)
        self.slide_gview.setScene(self.slide_gscene)

        # slide_indices = ['N11, N12, IHC28']
        macros_dir = os.path.join(RAW_DATA_DIR, 'macros/%(stack)s/' % {'stack': self.stack})

        slide_filenames = {}
        import re
        for fn in os.listdir(macros_dir):
            res = re.findall('^(.*?)\s?-\s?(F|N|IHC)\s*([0-9]+)\s?-\s?(.*?) (.*?)_macro.jpg$', fn)
            if len(res) > 0:
                _, prefix, slide_num, date, hour = res[0]
            else:
                res = re.findall('^macro_(.*?)-(F|N|IHC)([0-9]+)-(.*?)-(.*?).jpg$', fn)
                if len(res) > 0:
                    _, prefix, slide_num, date, hour = res[0]
                else:
                    continue
            slide_filenames[prefix + '_%d' % int(slide_num)] = fn

        create_if_not_exists(self.stack_data_dir)
        with open(self.stack_data_dir + '/' + self.stack + '_slide_list.txt', 'w') as f:
            for slide_name, fn in sorted(slide_filenames.items(), key=lambda x: int(x[0].split('_')[1])):
                f.write(slide_name + ' ' + fn + '\n')

        with open(self.stack_data_dir + '/' + self.stack + '_missing_slide_list.txt', 'w') as f:
            all_IHC_slide_indices = [int(k.split('_')[1]) for k in slide_filenames.keys() if k.startswith('IHC')]
            if len(all_IHC_slide_indices) > 0:
                max_index = np.max(all_IHC_slide_indices)
                min_index = np.min(all_IHC_slide_indices)
                missing_IHC_indices = set(range(min_index, max_index+1)) - set(all_IHC_slide_indices)
            else:
                missing_IHC_indices = []

            all_N_slide_indices = [int(k.split('_')[1]) for k in slide_filenames.keys() if k.startswith('N')]
            if len(all_N_slide_indices) > 0:
                max_index = np.max(all_N_slide_indices)
                min_index = np.min(all_N_slide_indices)
                missing_N_indices = set(range(min_index, max_index+1)) - set(all_N_slide_indices)
            else:
                missing_N_indices = []

            f.write(' '.join(sorted(['IHC_%d' % i for i in missing_IHC_indices], key=lambda x: int(x.split('_')[1])) + \
                            sorted(['N_%d' % i for i in missing_N_indices], key=lambda x: int(x.split('_')[1]))))

        # print sorted(slide_filenames.keys())

        slide_image_feeder = ImageDataFeeder('image feeder', stack=self.stack, sections=sorted(sorted(slide_filenames.keys(), reverse=True), key=lambda slide_name: int(slide_name.split('_')[1])),
                                            use_data_manager=False)
        # slide_image_feeder.set_orientation(stack_orientation[self.stack])
        slide_image_feeder.set_images(labels=slide_filenames.keys(),
                                    filenames=[os.path.join(macros_dir, filename) for filename in slide_filenames.itervalues()],
                                    downsample=32)
        slide_image_feeder.set_downsample_factor(32)

        self.slide_gscene.set_data_feeder(slide_image_feeder)
        self.slide_gscene.set_active_section(sorted(slide_filenames.keys())[0])
        self.slide_gscene.active_image_updated.connect(self.slide_image_updated)

        ################################################

        # self.slide_position_to_fn = defaultdict(lambda: defaultdict(lambda: 'Unknown'))
        self.slide_position_to_fn = {slide_index: {p: 'Unknown' for p in [1,2,3]}
                                        for slide_index in slide_filenames.iterkeys()}

        ###############

        self.thumbnails_dir = os.path.join(RAW_DATA_DIR, '%(stack)s/' % {'stack': self.stack})
        from glob import glob

        self.thumbnail_filenames = defaultdict(lambda: defaultdict(lambda: dict()))
        self.filename_to_slide = {}

        for fp in glob(self.thumbnails_dir+'/*.%s' % self.tb_fmt):
            fn = os.path.splitext(os.path.basename(fp))[0]
            _, prefix, slide_num, date, hour, _, position, index = re.findall('^(.*?)-([A-Z]+)([0-9]+)-(.*?)-(.*?)_(.*?)_([0-9])_([0-9]{4})$', fn)[0]
            # print prefix, slide_num, position, index
            slide_name = prefix + '_%d' % int(slide_num)
            self.filename_to_slide[fn] = slide_name
            self.thumbnail_filenames[slide_name][int(position)][date+'_'+index] = fn

        ################

        self.section1_gscene = SimpleGraphicsScene2(id=1, gview=self.section1_gview)
        self.section1_gview.setScene(self.section1_gscene)
        self.section1_gscene.active_image_updated.connect(self.slide_position_image_updated)
        self.section2_gscene = SimpleGraphicsScene2(id=2, gview=self.section2_gview)
        self.section2_gview.setScene(self.section2_gscene)
        self.section2_gscene.active_image_updated.connect(self.slide_position_image_updated)
        self.section3_gscene = SimpleGraphicsScene2(id=3, gview=self.section3_gview)
        self.section3_gview.setScene(self.section3_gscene)
        self.section3_gscene.active_image_updated.connect(self.slide_position_image_updated)

        self.section1_gscene.status_updated.connect(self.slide_position_status_updated)
        self.section2_gscene.status_updated.connect(self.slide_position_status_updated)
        self.section3_gscene.status_updated.connect(self.slide_position_status_updated)
        self.section1_gscene.send_to_sorted_requested.connect(self.send_to_sorted)
        self.section2_gscene.send_to_sorted_requested.connect(self.send_to_sorted)
        self.section3_gscene.send_to_sorted_requested.connect(self.send_to_sorted)

        self.slide_position_gscenes = {1: self.section1_gscene, 2: self.section2_gscene, 3: self.section3_gscene}

        self.section_image_feeders = {}

        # blank_image_data = np.zeros((800, 500), np.uint8) * 255
        # self.blank_qimage = QImage(blank_image_data.flatten(), 500, 800, 500, QImage.Format_Indexed8)
        # self.blank_qimage.setColorTable(gray_color_table)

        self.placeholder_qimage = QImage(800, 500, QImage.Format_RGB888)
        painter = QPainter(self.placeholder_qimage)
        painter.fillRect(self.placeholder_qimage.rect(), Qt.yellow);
        painter.drawText(self.placeholder_qimage.rect(), Qt.AlignCenter | Qt.AlignVCenter, "Placeholder");

        self.rescan_qimage = QImage(800, 500, QImage.Format_RGB888)
        painter = QPainter(self.rescan_qimage)
        painter.fillRect(self.rescan_qimage.rect(), Qt.green);
        painter.drawText(self.rescan_qimage.rect(), Qt.AlignCenter | Qt.AlignVCenter, "Rescan");

        self.nonexisting_qimage = QImage(800, 500, QImage.Format_RGB888)
        painter = QPainter(self.nonexisting_qimage)
        painter.fillRect(self.nonexisting_qimage.rect(), Qt.white);
        painter.drawText(self.nonexisting_qimage.rect(), Qt.AlignCenter | Qt.AlignVCenter, "Nonexisting");

        for slide_index in slide_filenames.iterkeys():

            filenames = [fn for x in self.thumbnail_filenames[slide_index].values() for fn in x.values()]

            section_image_feeder = ImageDataFeeder('image feeder', stack=self.stack,
                                                    sections=filenames + ['Nonexisting', 'Rescan', 'Placeholder'],
                                                    use_data_manager=False)
            section_image_feeder.set_images(labels=filenames,
                                            filenames=[os.path.join(self.thumbnails_dir, fn + '.' + self.tb_fmt) for fn in filenames],
                                            downsample=32)

            section_image_feeder.set_downsample_factor(32)
            # section_image_feeder.set_orientation(stack_orientation[self.stack])

            # for fn in filenames:
            #     qimage = QImage(os.path.join(self.thumbnails_dir, fn + '.tif'))
            #     section_image_feeder.set_image(qimage, fn)

            section_image_feeder.set_image(qimage=self.nonexisting_qimage, sec='Nonexisting')
            section_image_feeder.set_image(qimage=self.rescan_qimage, sec='Rescan')
            section_image_feeder.set_image(qimage=self.placeholder_qimage, sec='Placeholder')

            self.section_image_feeders[slide_index] = section_image_feeder


        #######################

        self.first_section = 1
        self.last_section = 2

        # self.sorted_filenames = []
        #
        # filename_map_fp = '/home/yuncong/CSHL_data_processed/%(stack)s_filename_map.txt' % {'stack': stack}
        # if os.path.exists(filename_map_fp):
        #     with open(filename_map_fp, 'r') as f:
        #         for line in f.readlines():
        #             fn, sequence_index = line.split()
        #             self.sorted_filenames.append((int(sequence_index), fn))
        #     self.sorted_filenames = [fn for si, fn in sorted(self.sorted_filenames)]

        self.sorted_sections_gscene = SimpleGraphicsScene3(id='sorted', gview=self.sorted_sections_gview)
        self.sorted_sections_gview.setScene(self.sorted_sections_gscene)

        # ordered_image_feeder = ImageDataFeeder('image feeder', stack=self.stack, sections=range(1, len(self.sorted_filenames)+1), use_data_manager=False)
        # ordered_image_feeder.set_downsample_factor(32)
        # ordered_image_feeder.set_orientation(stack_orientation[self.stack])
        #
        # for i, filename in enumerate(self.sorted_filenames):
        #     qimage = QImage(os.path.join(self.thumbnails_dir, filename))
        #     ordered_image_feeder.set_image(qimage, i+1)
        #
        # self.sorted_sections_gscene.set_data_feeder(ordered_image_feeder)
        #
        # self.sorted_sections_gscene.set_active_section(2)
        self.sorted_sections_gscene.active_image_updated.connect(self.sorted_sections_image_updated)
        # self.sorted_sections_gscene.bad_status_changed.connect(self.bad_status_changed)
        self.sorted_sections_gscene.first_section_set.connect(self.set_first_section)
        self.sorted_sections_gscene.last_section_set.connect(self.set_last_section)
        self.sorted_sections_gscene.anchor_set.connect(self.set_anchor)

        # self.sorted_sections_gscene.move_forward_requested.connect(self.move_forward)
        # self.sorted_sections_gscene.move_backward_requested.connect(self.move_backward)
        # self.sorted_sections_gscene.edit_transform_requested.connect(self.edit_transform)

        #######################

        self.installEventFilter(self)

        self.comboBox_show.activated.connect(self.show_option_changed)

        self.comboBox_slide_position_adjustment.activated.connect(self.slide_position_adjustment_changed)

        self.button_save_slide_position_map.clicked.connect(self.save)
        self.button_load_slide_position_map.clicked.connect(self.load_slide_position_map)
        self.button_sort.clicked.connect(self.sort)
        # self.button_confirm_order.clicked.connect(self.confirm_order)
        self.button_load_sorted_filenames.clicked.connect(self.load_sorted_filenames)
        self.button_save_sorted_filenames.clicked.connect(self.save_sorted_filenames)
        # self.button_align.clicked.connect(self.align)
        # self.button_align.setEnabled(False)
        self.button_confirm_alignment.clicked.connect(self.compose)
        # self.button_download.clicked.connect(self.download)
        self.button_edit_transform.clicked.connect(self.edit_transform)
        self.button_crop.clicked.connect(self.crop)
        self.button_load_crop.clicked.connect(self.load_crop)
        self.button_save_crop.clicked.connect(self.save_crop)
        # self.button_gen_mask.clicked.connect(self.generate_masks)
        # self.button_warp_crop_mask.clicked.connect(self.warp_crop_masks)
        # self.button_syncWorkstation.clicked.connect(self.send_to_workstation)
        # self.button_edit_masks.clicked.connect(self.edit_masks)

        # self.button_send_info_gordon.clicked.connect(self.send_info_gordon)
        # self.button_send_info_workstation.clicked.connect(self.send_info_workstation)

        ################################

        self.placeholders = set([])
        self.rescans = set([])

        # self.status_comboBoxes = {1: self.comboBox_status1, 2: self.comboBox_status2, 3: self.comboBox_status3}
        self.labels_slide_position_filename = {1: self.label_section1_filename, 2: self.label_section2_filename, 3: self.label_section3_filename}
        self.labels_slide_position_index = {1: self.label_section1_index, 2: self.label_section2_index, 3: self.label_section3_index}


    # def bad_status_changed(self, is_bad):
    #     """
    #     This function is called when a section is marked bad in the Sorted Images panel.
    #     """
    #
    #     print is_bad
    #
    #     fn = self.sorted_filenames[self.sorted_sections_gscene.active_section-1]
    #
    #     slide_name = self.filename_to_slide[fn]
    #     position = self.slide_position_to_fn[slide_name].keys()[self.slide_position_to_fn[slide_name].values().index(fn)]
    #
    #     if is_bad > 0:
    #         self.sorted_filenames[self.sorted_filenames.index(fn)] = 'Placeholder'
    #         self.set_status(slide_name, position, 'Placeholder')
    #     else:
    #         raise Exception('Cannot undo set Bad..')
    #         self.set_status(slide_name, position, 'Normal')
    #
    #     self.sorted_sections_gscene.set_bad_sections(self.get_bad_sections())


    def send_to_sorted(self, position):
        slide_name = self.slide_gscene.active_section
        fn = self.slide_position_to_fn[slide_name][position]
        index = self.sorted_filenames.index(fn) + 1
        self.sorted_sections_gscene.set_active_section(index)

    def slide_position_adjustment_changed(self, index):
        adjustment_str = str(self.sender().currentText())
        slide_name = self.slide_gscene.active_section

        def swap_normal_positions(x):
            if x[1] == 'Nonexisting':
                x[2], x[3] = x[3], x[2]
            elif x[3] == 'Nonexisting':
                x[1], x[2] = x[2], x[1]
            else:
                x[3], x[1] = x[1], x[3]

        def adjust(x, status, pos):
            abnormal_statuses = ['Nonexisting', 'Rescan', 'Placeholder']
            if pos == 'right':
                if x[1] in abnormal_statuses:
                    x[1], x[2], x[3] = status, x[2], x[3]
                elif x[2] in abnormal_statuses:
                    x[1], x[2], x[3] = status, x[1], x[3]
                elif x[3] in abnormal_statuses:
                    x[1], x[2], x[3] = status, x[1], x[2]
            elif pos == 'left':
                if x[1] in abnormal_statuses:
                    x[1], x[2], x[3] = x[2], x[3], status
                elif x[2] in abnormal_statuses:
                    x[1], x[2], x[3] = x[1], x[3], status
                elif x[3] in abnormal_statuses:
                    x[1], x[2], x[3] = x[1], x[2], status

        if adjustment_str == 'Reverse positions':

            x = self.slide_position_to_fn[slide_name]

            # if x.values().count('Nonexisting') == 0:

            # if x[1] == 'Nonexisting':
            #     x[2], x[3] = x[3], x[2]
            # elif x[3] == 'Nonexisting':
            #     x[1], x[2] = x[2], x[1]
            # else:
            #     x[3], x[1] = x[1], x[3]
            swap_normal_positions(x)

            for position in [1,2,3]:
                self.set_status(slide_name, position, self.slide_position_to_fn[slide_name][position])

        elif adjustment_str == 'Reverse positions on all slides':
            for sn, x in self.slide_position_to_fn.iteritems():
                # if x.values().count('Nonexisting') == 0:
                # if x[1] == 'Nonexisting':
                #     x[2], x[3] = x[3], x[2]
                # elif x[3] == 'Nonexisting':
                #     x[1], x[2] = x[2], x[1]
                # else:
                #     x[3], x[1] = x[1], x[3]
                swap_normal_positions(x)

                for pos in [1,2,3]:
                    self.set_status(sn, pos, self.slide_position_to_fn[sn][pos])

        elif adjustment_str == 'Reverse positions on all IHC slides':
            for sn, x in self.slide_position_to_fn.iteritems():
                if sn.startswith('IHC'):
                    # if x[1] == 'Nonexisting':
                    #     x[2], x[3] = x[3], x[2]
                    # elif x[3] == 'Nonexisting':
                    #     x[1], x[2] = x[2], x[1]
                    # else:
                    #     x[3], x[1] = x[1], x[3]
                    swap_normal_positions(x)

                    for pos in [1,2,3]:
                        self.set_status(sn, pos, self.slide_position_to_fn[sn][pos])

        elif adjustment_str == 'Reverse positions on all N slides':
            for sn, x in self.slide_position_to_fn.iteritems():
                # if sn.startswith('N') and x.values().count('Nonexisting') == 0:
                if sn.startswith('N'):
                    # print sn, x
                    swap_normal_positions(x)

                    # if x[1] == 'Nonexisting':
                    #     x[2], x[3] = x[3], x[2]
                    # elif x[3] == 'Nonexisting':
                    #     x[1], x[2] = x[2], x[1]
                    # else:
                    #     x[3], x[1] = x[1], x[3]

                    for pos in [1,2,3]:
                        self.set_status(sn, pos, self.slide_position_to_fn[sn][pos])

        elif adjustment_str == 'Move all nonexisting to left':

            for sn, x in self.slide_position_to_fn.iteritems():

                if x.values().count('Nonexisting') == 1:

                    adjust(x, 'Nonexisting', 'left')

                    # if x[1] in ['Nonexisting', 'Rescan', 'Placeholder']:
                    #     x[1], x[2], x[3] = x[2], x[3], 'Nonexisting'
                    # elif x[2] in ['Nonexisting', 'Rescan', 'Placeholder']:
                    #     x[1], x[2], x[3] = x[1], x[3], 'Nonexisting'
                    # elif x[3] in ['Nonexisting', 'Rescan', 'Placeholder']:
                    #     x[1], x[2], x[3] = x[1], x[2], 'Nonexisting'

                    for pos in [1,2,3]:
                        self.set_status(sn, pos, self.slide_position_to_fn[sn][pos])

        elif adjustment_str == 'Move all nonexisting to right':

            for sn, x in self.slide_position_to_fn.iteritems():

                if x.values().count('Nonexisting') == 1:
                    adjust(x, 'Nonexisting', 'right')

                    # if x[1] in ['Nonexisting', 'Rescan', 'Placeholder']:
                    #     x[1], x[2], x[3] = 'Nonexisting', x[2], x[3]
                    # elif x[2] in ['Nonexisting', 'Rescan', 'Placeholder']:
                    #     x[1], x[2], x[3] = 'Nonexisting', x[1], x[3]
                    # elif x[3] in ['Nonexisting', 'Rescan', 'Placeholder']:
                    #     x[1], x[2], x[3] = 'Nonexisting', x[1], x[2]

                    for pos in [1,2,3]:
                        self.set_status(sn, pos, self.slide_position_to_fn[sn][pos])
        else:
            x = self.slide_position_to_fn[slide_name]
            assert x.values().count('Nonexisting') == 1

            if adjustment_str == 'Move placeholder to right':
                adjust(x, 'Placeholder', 'right')
            elif adjustment_str == 'Move rescan to right':
                adjust(x, 'Rescan', 'right')
            elif adjustment_str == 'Move nonexisting to right':
                adjust(x, 'Nonexisting', 'right')
            elif adjustment_str == 'Move placeholder to left':
                adjust(x, 'Placeholder', 'left')
            elif adjustment_str == 'Move rescan to left':
                adjust(x, 'Rescan', 'left')
            elif adjustment_str == 'Move nonexisting to left':
                adjust(x, 'Nonexisting', 'left')

            for position in [1,2,3]:
                self.set_status(slide_name, position, self.slide_position_to_fn[slide_name][position])

    def slide_position_status_updated(self, position, status):
        print 'slide_position_status_updated:', position, status
        slide_name = self.slide_gscene.active_section
        # position = self.slide_position_gscenes[position].active_section
        if status == 'Normal':
            if position in self.thumbnail_filenames[slide_name]:
                newest_fn = sorted(self.thumbnail_filenames[slide_name][position].items())[-1][1]
                self.set_status(slide_name, position, newest_fn)
            else:
                arbitrary_image = self.thumbnail_filenames[slide_name].values()[0].values()[0]
                self.set_status(slide_name, position, arbitrary_image)
        else:
            self.set_status(slide_name, position, str(status))



    def load_crop(self):
        """
        Load crop box.
        """
        self.set_show_option('aligned')
        cropbox_fp = DataManager.get_cropbox_filename(stack=self.stack, anchor_fn=self.anchor_fn)
        with open(cropbox_fp, 'r') as f:
            ul_x, lr_x, ul_y, lr_y, self.first_section, self.last_section = map(int, f.readline().split())
            self.sorted_sections_gscene.set_box(ul_x, lr_x, ul_y, lr_y)
            print ul_x, lr_x, ul_y, lr_y, self.first_section, self.last_section

    def save_crop(self):
        ul_pos = self.sorted_sections_gscene.corners['ul'].scenePos()
        lr_pos = self.sorted_sections_gscene.corners['lr'].scenePos()
        ul_x = int(ul_pos.x())
        ul_y = int(ul_pos.y())
        lr_x = int(lr_pos.x())
        lr_y = int(lr_pos.y())

        # If not set yet.
        if ul_x == 100 and ul_y == 100 and lr_x == 200 and lr_y == 200:
            return

        cropbox_fp = DataManager.get_cropbox_filename(stack=self.stack, anchor_fn=self.anchor_fn)
        with open(cropbox_fp, 'w') as f:
            f.write('%d %d %d %d %d %d' % (ul_x, lr_x, ul_y, lr_y, self.first_section, self.last_section))

    def crop(self):
        ## Note that in cropbox, xmax, ymax are not included, so w = xmax-xmin, instead of xmax-xmin+1

        self.save_crop()

        ul_pos = self.sorted_sections_gscene.corners['ul'].scenePos()
        lr_pos = self.sorted_sections_gscene.corners['lr'].scenePos()
        ul_x = int(ul_pos.x())
        ul_y = int(ul_pos.y())
        lr_x = int(lr_pos.x())
        lr_y = int(lr_pos.y())

        if self.stack in all_nissl_stacks:
            pad_bg_color = 'white'
        elif self.stack in all_ntb_stacks:
            pad_bg_color = 'black'
        elif self.stack in all_alt_nissl_ntb_stacks or self.stack in all_alt_nissl_tracing_stacks:
            pad_bg_color = 'auto'

        self.web_service.convert_to_request('crop', stack=self.stack, x=ul_x, y=ul_y, w=lr_x+1-ul_x, h=lr_y+1-ul_y,
                                            f=self.first_section, l=self.last_section, anchor_fn=self.anchor_fn,
                                            filenames=self.get_valid_sorted_filenames(),
                                            first_fn=self.sorted_filenames[self.first_section-1],
                                            last_fn=self.sorted_filenames[self.last_section-1],
                                            pad_bg_color=pad_bg_color)

        # # Download unsorted thumbnail cropped images
        # self.statusBar().showMessage('Downloading aligned cropped thumbnail images ...')
        #
        # execute_command(('ssh gcn-20-34.sdsc.edu \"cd %(gordon_data_dir)s && tar -I pigz -cf %(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped.tar.gz %(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped/*.tif\" && '
        #                 'scp oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped.tar.gz %(local_data_dir)s/ &&'
        #                 'ssh gcn-20-34.sdsc.edu rm %(gordon_data_dir)s/%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped.tar.gz &&'
        #                 'cd %(local_data_dir)s && rm -rf %(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped && tar -xf %(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped.tar.gz && rm %(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped.tar.gz' ) % \
        #                 dict(gordon_data_dir=self.stack_data_dir_gordon,
        #                     local_data_dir=self.stack_data_dir,
        #                     stack=self.stack,
        #                     anchor_fn=self.anchor_fn))


    ##################
    # Edit Alignment #
    ##################

    def edit_transform(self):

        sys.stderr.write('Loading Edit Transform GUI...\n')
        self.statusBar().showMessage('Loading Edit Transform GUI...')

        self.alignment_ui = Ui_AlignmentGui()
        self.alignment_gui = QDialog(self)
        self.alignment_ui.setupUi(self.alignment_gui)

        self.alignment_ui.button_anchor.clicked.connect(self.add_anchor_pair_clicked)
        # self.alignment_ui.button_upload_transform.clicked.connect(self.upload_custom_transform)
        self.alignment_ui.button_align.clicked.connect(self.align_using_elastix)
        self.alignment_ui.button_compute.clicked.connect(self.compute_custom_transform)

        param_fns = os.listdir(os.path.join(REPO_DIR, 'preprocess', 'parameters'))
        all_parameter_names = ['_'.join(pf[:-4].split('_')[1:]) for pf in param_fns]
        self.alignment_ui.comboBox_parameters.addItems(all_parameter_names)

        self.valid_section_filenames = [fn for fn in self.sorted_filenames if fn != 'Placeholder' and fn != 'Rescan']
        self.valid_section_indices = [self.sorted_filenames.index(fn)+1 for fn in self.valid_section_filenames]

        valid_sections_feeder = ImageDataFeeder('valid image feeder', stack=self.stack, sections=self.valid_section_indices, use_data_manager=False)
        valid_sections_feeder.set_images(self.valid_section_indices, [os.path.join(self.thumbnails_dir, fn + '.' + self.tb_fmt) for fn in self.valid_section_filenames], downsample=32)
        valid_sections_feeder.set_downsample_factor(32)

        self.curr_gscene = SimpleGraphicsScene4(id='current', gview=self.alignment_ui.curr_gview)
        self.alignment_ui.curr_gview.setScene(self.curr_gscene)
        self.curr_gscene.set_data_feeder(valid_sections_feeder)
        self.curr_gscene.set_active_i(1)
        self.curr_gscene.active_image_updated.connect(self.current_section_image_changed)
        self.curr_gscene.anchor_point_added.connect(self.anchor_point_added)

        self.prev_gscene = SimpleGraphicsScene4(id='previous', gview=self.alignment_ui.prev_gview)
        self.alignment_ui.prev_gview.setScene(self.prev_gscene)
        self.prev_gscene.set_data_feeder(valid_sections_feeder)
        self.prev_gscene.set_active_i(0)
        self.prev_gscene.active_image_updated.connect(self.previous_section_image_changed)
        self.prev_gscene.anchor_point_added.connect(self.anchor_point_added)

        self.aligned_gscene = MultiplePixmapsGraphicsScene(id='aligned', pixmap_labels=['moving', 'fixed'], gview=self.alignment_ui.aligned_gview)
        self.alignment_ui.aligned_gview.setScene(self.aligned_gscene)
        self.transformed_images_feeder = ImageDataFeeder('aligned image feeder', stack=self.stack,
                                                sections=self.valid_section_indices, use_data_manager=False)
        self.transformed_images_feeder.set_downsample_factor(32)
        self.update_aligned_images_feeder()
        self.aligned_gscene.set_data_feeder(self.transformed_images_feeder, 'moving')
        self.aligned_gscene.set_data_feeder(valid_sections_feeder, 'fixed')
        self.aligned_gscene.set_active_indices({'moving': 1, 'fixed': 0})
        self.aligned_gscene.set_opacity('moving', .8)
        self.aligned_gscene.set_opacity('fixed', .8)
        # self.aligned_gscene.active_image_updated.connect(self.aligned_image_changed)

        self.alignment_gui.show()

    def update_aligned_images_feeder(self):

        transformed_image_filenames = []
        for i in range(len(self.valid_section_indices)):

            custom_aligned_image_fn = self.stack_data_dir + '/%(stack)s_custom_transforms/%(curr_fn)s_to_%(prev_fn)s/%(curr_fn)s_alignedTo_%(prev_fn)s.tif' % \
            {'stack': self.stack, 'curr_fn': self.valid_section_filenames[i], 'prev_fn': self.valid_section_filenames[i-1]}

            custom_aligned_image_fn2 = self.stack_data_dir + '/%(stack)s_custom_transforms/%(curr_fn)s_to_%(prev_fn)s/result.0.tif' % \
            {'stack': self.stack, 'curr_fn': self.valid_section_filenames[i], 'prev_fn': self.valid_section_filenames[i-1]}

            if os.path.exists(custom_aligned_image_fn):
                sys.stderr.write('Load custom transform image. %s\n' % custom_aligned_image_fn)
                transformed_image_filenames.append(custom_aligned_image_fn)
            elif os.path.exists(custom_aligned_image_fn2):
                sys.stderr.write('Load custom transform image. %s\n' % custom_aligned_image_fn2)
                transformed_image_filenames.append(custom_aligned_image_fn2)
            else:
                fn = self.stack_data_dir + '/%(stack)s_elastix_output/%(curr_fn)s_to_%(prev_fn)s/result.0.tif' % \
                {'stack': self.stack, 'curr_fn': self.valid_section_filenames[i], 'prev_fn': self.valid_section_filenames[i-1]}
                transformed_image_filenames.append(fn)

        self.transformed_images_feeder.set_images(self.valid_section_indices, transformed_image_filenames, downsample=32, load_with_cv2=True)

    def align_using_elastix(self):
        selected_elastix_parameter_name = str(self.alignment_ui.comboBox_parameters.currentText())
        param_fn = os.path.join(REPO_DIR, 'preprocess', 'parameters', 'Parameters_' + selected_elastix_parameter_name + '.txt')

        curr_fn = self.valid_section_filenames[self.curr_gscene.active_i]
        prev_fn = self.valid_section_filenames[self.prev_gscene.active_i]
        out_dir = os.path.join(self.stack_data_dir, self.stack + '_custom_transforms', curr_fn + '_to_' + prev_fn)

        curr_fp = os.path.join(RAW_DATA_DIR, self.stack, curr_fn + '.' + self.tb_fmt)
        prev_fp = os.path.join(RAW_DATA_DIR, self.stack, prev_fn + '.' + self.tb_fmt)

        execute_command('rm -rf %(out_dir)s; mkdir -p %(out_dir)s; elastix -f %(fixed_fn)s -m %(moving_fn)s -out %(out_dir)s -p %(param_fn)s' % \
        dict(param_fn=param_fn, out_dir=out_dir, fixed_fn=prev_fp, moving_fn=curr_fp))

        self.update_aligned_images_feeder()

    def anchor_point_added(self, index):
        gscene_id = self.sender().id
        if gscene_id == 'current':
            self.current_section_anchor_received = True
            # self.curr_gscene.anchor_points.index
        elif gscene_id == 'previous':
            self.previous_section_anchor_received = True

        if self.current_section_anchor_received and self.previous_section_anchor_received:
            self.curr_gscene.set_mode('idle')
            self.prev_gscene.set_mode('idle')
            self.current_section_anchor_received = False
            self.previous_section_anchor_received = False
            self.alignment_ui.button_anchor.setEnabled(True)

    def compute_custom_transform(self):
        """
        Compute transform based on added control points.
        """

        curr_points = []
        for i, c in enumerate(self.curr_gscene.anchor_circle_items):
            pos = c.scenePos()
            curr_points.append((pos.x(), pos.y()))

        prev_points = []
        for i, c in enumerate(self.prev_gscene.anchor_circle_items):
            pos = c.scenePos()
            prev_points.append((pos.x(), pos.y()))

        print self.curr_gscene.active_section, np.array(curr_points)
        print self.prev_gscene.active_section, np.array(prev_points)

        curr_points = np.array(curr_points)
        prev_points = np.array(prev_points)
        curr_centroid = curr_points.mean(axis=0)
        prev_centroid = prev_points.mean(axis=0)
        curr_points0 = curr_points - curr_centroid
        prev_points0 = prev_points - prev_centroid

        H = np.dot(curr_points0.T, prev_points0)
        U, S, VT = np.linalg.svd(H)
        R = np.dot(VT.T, U.T)

        t = -np.dot(R, curr_centroid) + prev_centroid

        print R, t

        # Write to custom transform file
        curr_section_fn = self.sorted_filenames[self.valid_section_indices[self.curr_gscene.active_i]-1]
        prev_section_fn = self.sorted_filenames[self.valid_section_indices[self.prev_gscene.active_i]-1]

        custom_tf_dir = os.path.join(self.stack_data_dir, stack + '_custom_transforms', curr_section_fn + '_to_' + prev_section_fn)

        execute_command("rm -rf %(out_dir)s; mkdir -p %(out_dir)s" % dict(out_dir=custom_tf_dir))
        custom_tf_fp = os.path.join(custom_tf_dir, '%(curr_fn)s_to_%(prev_fn)s_customTransform.txt' % \
                    dict(curr_fn=curr_section_fn, prev_fn=prev_section_fn))

        with open(custom_tf_fp, 'w') as f:
            f.write('%f %f %f %f %f %f\n' % (R[0,0], R[0,1], t[0], R[1,0], R[1,1], t[1]))

        self.apply_custom_transform()
        self.update_aligned_images_feeder()

    def apply_custom_transform(self):

        curr_section_fn = self.sorted_filenames[self.valid_section_indices[self.curr_gscene.active_i]-1]
        prev_section_fn = self.sorted_filenames[self.valid_section_indices[self.prev_gscene.active_i]-1]

        custom_tf_fn = os.path.join(self.stack_data_dir, self.stack+'_custom_transforms', curr_section_fn + '_to_' + prev_section_fn, curr_section_fn + '_to_' + prev_section_fn + '_customTransform.txt')
        with open(custom_tf_fn, 'r') as f:
            t11, t12, t13, t21, t22, t23 = map(float, f.readline().split())

        prev_img_w, prev_img_h = map(int, check_output("identify -format %%Wx%%H %(raw_data_dir)s/%(stack)s/%(prev_fn)s.%(tb_fmt)s" %dict(stack=self.stack, prev_fn=prev_section_fn, tb_fmt=self.tb_fmt, raw_data_dir=RAW_DATA_DIR),
                                            shell=True).split('x'))

        output_image_fn = os.path.join(self.stack_data_dir, '%(stack)s_custom_transforms/%(curr_fn)s_to_%(prev_fn)s/%(curr_fn)s_alignedTo_%(prev_fn)s.tif' % \
                        dict(stack=self.stack,
                        curr_fn=curr_section_fn,
                        prev_fn=prev_section_fn) )

        execute_command("convert %(raw_data_dir)s/%(stack)s/%(curr_fn)s.%(tb_fmt)s -virtual-pixel background +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw %(output_fn)s" %\
        dict(stack=self.stack,
            curr_fn=curr_section_fn,
            output_fn=output_image_fn,
            tb_fmt=self.tb_fmt,
            sx=t11,
            sy=t22,
            rx=t21,
            ry=t12,
            tx=t13,
            ty=t23,
            w=prev_img_w,
            h=prev_img_h,
            x='+0',
            y='+0',
            raw_data_dir=RAW_DATA_DIR))

    def add_anchor_pair_clicked(self):
        self.curr_gscene.set_mode('add point')
        self.prev_gscene.set_mode('add point')
        self.current_section_anchor_received = False
        self.previous_section_anchor_received = False
        self.alignment_ui.button_anchor.setEnabled(False)

    def current_section_image_changed(self):
        curr_section_i = self.curr_gscene.active_i
        curr_section_label = self.valid_section_indices[curr_section_i]
        curr_section_fn = self.sorted_filenames[curr_section_label-1]
        self.alignment_ui.label_current_filename.setText(curr_section_fn)
        self.alignment_ui.label_current_index.setText(str(curr_section_label))
        prev_section_i = curr_section_i - 1
        self.prev_gscene.set_active_i(prev_section_i)
        self.aligned_gscene.set_active_indices({'moving': curr_section_i, 'fixed': prev_section_i})

    def previous_section_image_changed(self):
        prev_section_i = self.prev_gscene.active_i
        prev_section_label = self.valid_section_indices[prev_section_i]
        prev_section_fn = self.sorted_filenames[prev_section_label-1]
        self.alignment_ui.label_previous_filename.setText(prev_section_fn)
        self.alignment_ui.label_previous_index.setText(str(prev_section_label))
        curr_section_i = prev_section_i + 1
        self.curr_gscene.set_active_i(curr_section_i)
        self.aligned_gscene.set_active_indices({'moving': curr_section_i, 'fixed': prev_section_i})

    ########################## END OF EDIT TRANSFORM ######################################3


    def sort(self):
        """
        Sort images.
        """

        # self.fn_to_slide_position = {fn: sp for sp, fn in self.slide_position_to_fn.iteritems()}

        # If sorting is already performed,
        # if hasattr(self, 'sorted_slide_positions'):
        #     sys.stderr.write('Sorted slide position available. Simply reload filenames.\n')
        #     self.sorted_filenames = [self.slide_position_to_fn[slide_pos] for slide_pos in self.sorted_slide_positions]
        # else:

        if self.stack in all_alt_nissl_ntb_stacks or self.stack in all_alt_nissl_tracing_stacks:

            F_series = {int(slide_name.split('_')[1]): x for slide_name, x in self.slide_position_to_fn.items() if slide_name.split('_')[0] == 'F'}
            N_series = {int(slide_name.split('_')[1]): x for slide_name, x in self.slide_position_to_fn.items() if slide_name.split('_')[0] == 'N'}

            sorted_fns = []
            for i in sorted(set(F_series.keys() + N_series.keys())):
                if i in N_series and i in F_series:
                    for pos in range(1, 4):
                        sorted_fns.append(N_series[i][pos])
                        sorted_fns.append(F_series[i][pos])
                elif i in N_series:
                    sorted_fns += [N_series[i][pos] for pos in range(1, 4)]
                elif i in F_series:
                    sorted_fns += [F_series[i][pos] for pos in range(1, 4)]

        elif self.stack in all_ntb_stacks:
            # fluro
            F_series = {int(slide_name.split('_')[1]): x for slide_name, x in self.slide_position_to_fn.items() if slide_name.split('_')[0] == 'F'}
            sorted_fns = []
            for i in sorted(set(F_series.keys())):
                sorted_fns += [F_series[i][pos] for pos in range(1, 4)]
        else:
            if self.stack == 'MD639':
                IHC_series = {int(np.ceil(int(slide_name.split('_')[1])/2.)): x for slide_name, x in self.slide_position_to_fn.items() if int(slide_name.split('_')[1]) % 2 == 0}
                N_series = {int(np.ceil(int(slide_name.split('_')[1])/2.)): x for slide_name, x in self.slide_position_to_fn.items() if int(slide_name.split('_')[1]) % 2 == 1}
            else:
                IHC_series = {int(slide_name.split('_')[1]): x for slide_name, x in self.slide_position_to_fn.items() if slide_name.split('_')[0] == 'IHC'}
                N_series = {int(slide_name.split('_')[1]): x for slide_name, x in self.slide_position_to_fn.items() if slide_name.split('_')[0] == 'N'}

            sorted_fns = []
            for i in sorted(set(IHC_series.keys() + N_series.keys())):
                if i in N_series and i in IHC_series:
                    for pos in range(1, 4):
                        sorted_fns.append(N_series[i][pos])
                        sorted_fns.append(IHC_series[i][pos])
                elif i in N_series:
                    sorted_fns += [N_series[i][pos] for pos in range(1, 4)]
                elif i in IHC_series:
                    sorted_fns += [IHC_series[i][pos] for pos in range(1, 4)]

        if len(sorted_fns) == 0:
            raise Exception('sorted_fns is empty.')

        self.sorted_filenames = [fn for fn in sorted_fns if fn != 'Nonexisting']
        # self.sorted_slide_positions = [self.fn_to_slide_position[fn] for fn in self.sorted_filenames]

        self.update_sorted_sections_gscene_from_sorted_filenames()

    def save_everything(self):

        # Dump preprocessing info
        placeholder_indices = [idx+1 for idx, fn in enumerate(self.sorted_filenames) if fn == 'Placeholder']
        placeholder_slide_positions = [(slide_name, pos) for slide_name, x in self.slide_position_to_fn.iteritems() for pos, fn in x.iteritems() if fn == 'Placeholder']
        rescan_indices = [idx+1 for idx, fn in enumerate(self.sorted_filenames) if fn == 'Rescan']
        rescan_slide_positions = [(slide_name, pos) for slide_name, x in self.slide_position_to_fn.iteritems() for pos, fn in x.iteritems() if fn == 'Rescan']

        ul_pos = self.sorted_sections_gscene.corners['ul'].scenePos()
        lr_pos = self.sorted_sections_gscene.corners['lr'].scenePos()
        ul_x = int(ul_pos.x())
        ul_y = int(ul_pos.y())
        lr_x = int(lr_pos.x())
        lr_y = int(lr_pos.y())

        info = {'placeholder_indices': placeholder_indices,
        'placeholder_slide_positions': placeholder_slide_positions,
        'rescan_indices': rescan_indices,
        'rescan_slide_positions': rescan_slide_positions,
        'sorted_filenames': self.sorted_filenames,
        'slide_position_to_fn': self.slide_position_to_fn,
        'first_section': self.first_section,
        'last_section': self.last_section,
        'anchor_fn': self.anchor_fn,
        # 'bbox': (ul_x, lr_x, ul_y, lr_y) #xmin,xmax,ymin,ymax
        'bbox': (ul_x, ul_y, lr_x+1-ul_x, lr_y+1-ul_y) #xmin,ymin,w,h
        }

        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
        pickle.dump(info, open(self.stack_data_dir + '/%(stack)s_preprocessInfo_%(timestamp)s.pkl' % {'stack': self.stack, 'timestamp':timestamp}, 'w'))

        execute_command('cd %(stack_data_dir)s && rm -f %(stack)s_preprocessInfo.pkl && ln -s %(stack)s_preprocessInfo_%(timestamp)s.pkl %(stack)s_preprocessInfo.pkl' % {'stack': self.stack, 'timestamp':timestamp, 'stack_data_dir':self.stack_data_dir})

        self.save_crop()
        self.save_sorted_filenames()
        self.save()


    # def send_info_gordon(self):
    #     # Upload cropbox file, sorted filenames file, anchor file
    #     execute_command(('scp %(stack_data_dir)s/%(stack)s_cropbox.txt oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/ &&'
    #                     'scp %(stack_data_dir)s/%(stack)s_anchor.txt oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/ &&'
    #                     'scp %(stack_data_dir)s/%(stack)s_sorted_filenames.txt oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/') %\
    #                     dict(stack=self.stack, stack_data_dir=self.stack_data_dir, stack_data_dir_gordon=self.stack_data_dir_gordon))


    # def send_info_workstation(self):
    #     pass

    # def send_to_workstation(self):
    #
    #     return
    #
    #     upload_to_remote_synced(stack=self.stack, fp_relative=self.stack + '_cropbox.txt')
    #
    #     remote_sorted_filenames_fp = os.path.join(self.stack_data_dir_gordon, self.stack + '_sorted_filenames.txt')
    #     upload_to_remote(fp_remote=remote_sorted_filenames_fp, fp_local=local_stack_data_processed_dir, remote_hostname='oasis-dm.sdsc.edu')
    #
    #     remote_anchor_fp = os.path.join(self.stack_data_dir_gordon, self.stack + '_anchor.txt')
    #     upload_to_remote(fp_remote=remote_anchor_fp, fp_local=local_stack_data_processed_dir, remote_hostname='oasis-dm.sdsc.edu')
    #
    #     remote_elastix_output_fp = os.path.join(self.stack_data_dir_gordon, self.stack + '_elastix_output')
    #     upload_to_remote(fp_remote=remote_elastix_output_fp, fp_local=local_stack_data_processed_dir, remote_hostname='oasis-dm.sdsc.edu')
    #
    #     commands_on_brainstem_download_metadata = \
    #     ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
    #     'scp dm:%(stack_data_dir_gordon)s/%(stack)s_cropbox.txt . &&'
    #     'scp dm:%(stack_data_dir_gordon)s/%(stack)s_sorted_filenames.txt . &&'
    #     'scp dm:%(stack_data_dir_gordon)s/%(stack)s_anchor.txt . &&'
    #     'mkdir %(stack)s_elastix_output; scp dm:%(stack_data_dir_gordon)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl %(stack)s_elastix_output/') \
    #     % dict(stack=self.stack, workstation_data_dir=WORKSTATION_ROOTDIR, stack_data_dir_gordon=self.stack_data_dir_gordon)

        # commands_gordon_tar_sorted_saturation = \
        # ('cd %(stack_data_dir_gordon)s &&'
        # 'tar -cf %(stack)s_lossless_sorted_aligned_cropped_saturation.tar %(stack)s_lossless_sorted_aligned_cropped_saturation') \
        # % dict(stack=self.stack,
        #         stack_data_dir_gordon=self.stack_data_dir_gordon)

        # commands_on_brainstem_download_sorted_saturation = \
        # ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        # 'rm -rf %(stack)s_lossless_sorted_aligned_cropped_saturation &&'
        # 'scp -r oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/%(stack)s_lossless_sorted_aligned_cropped_saturation.tar . &&'
        # 'tar -xf %(stack)s_lossless_sorted_aligned_cropped_saturation.tar') \
        # % dict(stack=self.stack,
        #         workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
        #         stack_data_dir_gordon=self.stack_data_dir_gordon)

        # commands_on_brainstem_download_unsorted_saturation = \
        # ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        # 'rm -rf %(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_saturation &&'
        # 'scp -r dm:%(stack_data_dir_gordon)s/%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_saturation .') \
        # % dict(stack=self.stack,
        #         workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
        #         stack_data_dir_gordon=self.stack_data_dir_gordon,
        #         anchor_fn=self.anchor_fn)

        commands_on_brainstem_download_unsorted_saturation = \
        ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        'rm -rf %(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_saturation &&'
        'scp -r dm:%(stack_data_dir_gordon)s/%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_saturation .') \
        % dict(stack=self.stack,
                workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
                stack_data_dir_gordon=self.stack_data_dir_gordon,
                anchor_fn=self.anchor_fn)

        # commands_gordon_tar_sorted_compressed = \
        # ('cd %(stack_data_dir_gordon)s &&'
        # 'tar -cf %(stack)s_lossless_sorted_aligned_cropped_compressed.tar %(stack)s_lossless_sorted_aligned_cropped_compressed') \
        # % dict(stack=self.stack,
        #         stack_data_dir_gordon=self.stack_data_dir_gordon)

        # commands_on_brainstem_download_sorted_compressed = \
        # ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        # 'rm -rf %(stack)s_lossless_sorted_aligned_cropped_compressed &&'
        # 'scp -r oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/%(stack)s_lossless_sorted_aligned_cropped_compressed.tar . &&'
        # 'tar -xf %(stack)s_lossless_sorted_aligned_cropped_compressed.tar') \
        # % dict(stack=self.stack,
        #         workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
        #         stack_data_dir_gordon=self.stack_data_dir_gordon)

        # commands_on_brainstem_download_unsorted_compressed = \
        # ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        # 'rm -rf %(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed &&'
        # 'scp -r dm:%(stack_data_dir_gordon)s/%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed .') \
        # % dict(stack=self.stack,
        #         workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
        #         stack_data_dir_gordon=self.stack_data_dir_gordon,
        #         anchor_fn=self.anchor_fn)

        commands_on_brainstem_download_unsorted_compressed = \
        ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        'rm -rf %(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_compressed &&'
        'scp -r dm:%(stack_data_dir_gordon)s/%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_compressed .') \
        % dict(stack=self.stack,
                workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
                stack_data_dir_gordon=self.stack_data_dir_gordon,
                anchor_fn=self.anchor_fn)

        # commands_gordon_tar_masks = \
        # ('cd %(stack_data_dir_gordon)s &&'
        # 'tar -cf %(stack)s_mask_sorted_aligned_cropped.tar %(stack)s_mask_sorted_aligned_cropped') \
        # % dict(stack=self.stack,
        #         stack_data_dir_gordon=self.stack_data_dir_gordon)

        # commands_on_brainstem_download_sorted_masks = \
        # ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        # 'rm -rf %(stack)s_mask_sorted_aligned_cropped &&'
        # 'scp -r oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/%(stack)s_mask_sorted_aligned_cropped.tar . &&'
        # 'tar -xf %(stack)s_mask_sorted_aligned_cropped.tar') \
        # % dict(stack=self.stack,
        #         workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
        #         stack_data_dir_gordon=self.stack_data_dir_gordon)

        # commands_on_brainstem_download_unsorted_masks = \
        # ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        # 'rm -rf %(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped &&'
        # 'scp -r dm:%(stack_data_dir_gordon)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped .') \
        # % dict(stack=self.stack,
        #         workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
        #         stack_data_dir_gordon=self.stack_data_dir_gordon,
        #         anchor_fn=self.anchor_fn)

        commands_on_brainstem_download_unsorted_masks = \
        ('cd %(workstation_data_dir)s && mkdir %(stack)s; cd %(stack)s &&'
        'rm -rf %(stack)s_masks_alignedTo_%(anchor_fn)s_cropped &&'
        'scp -r dm:%(stack_data_dir_gordon)s/%(stack)s_masks_alignedTo_%(anchor_fn)s_cropped .') \
        % dict(stack=self.stack,
                workstation_data_dir='/media/yuncong/BstemAtlasData/CSHL_data_processed/',
                stack_data_dir_gordon=self.stack_data_dir_gordon,
                anchor_fn=self.anchor_fn)

        # execute_command('ssh dm \"%(cmd)s\"' % dict(cmd=commands_gordon_tar_sorted_saturation))
        # execute_command('ssh brainstem \"%(cmd)s\"' % dict(cmd=commands_on_brainstem_download_sorted_saturation))
        execute_command('ssh brainstem \"%(cmd)s\"' % dict(cmd=commands_on_brainstem_download_unsorted_saturation))
        #
        # execute_command('ssh dm \"%(cmd)s\"' % dict(cmd=commands_gordon_tar_sorted_compressed))
        # execute_command('ssh brainstem \"%(cmd)s\"' % dict(cmd=commands_on_brainstem_download_sorted_compressed))
        # execute_command('ssh brainstem \"%(cmd)s\"' % dict(cmd=commands_on_brainstem_download_unsorted_compressed))

        execute_command('ssh brainstem \"%(cmd)s\"' % dict(cmd=commands_on_brainstem_download_metadata))

        # execute_command('ssh dm \"%(cmd)s\"' % dict(cmd=commands_gordon_tar_masks))
        # execute_command('ssh brainstem \"%(cmd)s\"' % dict(cmd=commands_on_brainstem_download_sorted_masks))
        execute_command('ssh brainstem \"%(cmd)s\"' % dict(cmd=commands_on_brainstem_download_unsorted_masks))


    # def confirm_order(self):
    #     sort_json = self.web_service.convert_to_request('confirm_order',
    #                     stack=self.stack, sorted_filenames=self.sorted_filenames, anchor_fn=self.anchor_fn)
    #
    #     # Download sorted data folder symbolic links
    #     download_sorted_thumbnails_symlinks_cmd = ('ssh oasis-dm.sdsc.edu \"cd %(stack_data_dir_gordon)s && tar -cf %(stack)s_thumbnail_sorted_aligned.tar %(stack)s_thumbnail_sorted_aligned\" && '
    #             'cd %(thumbnail_data_dir)s && mkdir %(stack)s ; cd %(stack)s &&'
    #             'scp -r oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/%(stack)s_thumbnail_sorted_aligned.tar . &&'
    #             'rm -rf %(stack)s_thumbnail_sorted_aligned && tar -xf %(stack)s_thumbnail_sorted_aligned.tar &&'
    #             'rm -r %(stack)s_thumbnail_sorted_aligned.tar') %\
    #             dict(stack=self.stack, stack_data_dir=self.stack_data_dir, stack_data_dir_gordon=self.stack_data_dir_gordon,
    #             thumbnail_data_dir=thumbnail_data_dir)
    #             # 'ssh oasis-dm.sdsc.edu rm %(stack_data_dir_gordon)s/%(stack)s_thumbnail_sorted_aligned.tar') % \
    #
    #     execute_command(download_sorted_thumbnails_symlinks_cmd)
    #
    #     self.statusBar().showMessage('Aligned cropped thumbnail images downloaded.')
    #
    #     # Download sorted thumbnail cropped data folder symbolic links
    #     download_sorted_thumbnails_symlinks_cmd = ('ssh oasis-dm.sdsc.edu \"cd %(stack_data_dir_gordon)s && tar -cf %(stack)s_thumbnail_sorted_aligned_cropped.tar %(stack)s_thumbnail_sorted_aligned_cropped\" && '
    #             'cd %(thumbnail_data_dir)s && mkdir %(stack)s ; cd %(stack)s &&'
    #             'scp -r oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/%(stack)s_thumbnail_sorted_aligned_cropped.tar . &&'
    #             'rm -rf %(stack)s_thumbnail_sorted_aligned_cropped && tar -xf %(stack)s_thumbnail_sorted_aligned_cropped.tar &&'
    #             'rm -r %(stack)s_thumbnail_sorted_aligned_cropped.tar') %\
    #             dict(stack=self.stack, stack_data_dir=self.stack_data_dir, stack_data_dir_gordon=self.stack_data_dir_gordon,
    #             thumbnail_data_dir=thumbnail_data_dir)
    #             # 'ssh oasis-dm.sdsc.edu rm %(stack_data_dir_gordon)s/%(stack)s_thumbnail_sorted_aligned_cropped.tar') % \
    #
    #     execute_command(download_sorted_thumbnails_symlinks_cmd)
    #
    #     # Download sorted lossless aligned cropped compressed data folder symbolic links
    #     execute_command(('ssh oasis-dm.sdsc.edu \"cd %(stack_data_dir_gordon)s && tar -cf %(stack)s_lossless_sorted_aligned_cropped_compressed.tar %(stack)s_lossless_sorted_aligned_cropped_compressed\" && '
    #                     'cd %(data_dir)s && mkdir %(stack)s ; cd %(stack)s &&'
    #                     'scp -r oasis-dm.sdsc.edu:%(stack_data_dir_gordon)s/%(stack)s_lossless_sorted_aligned_cropped_compressed.tar . &&'
    #                     'rm -rf %(stack)s_lossless_sorted_aligned_cropped_compressed && tar -xf %(stack)s_lossless_sorted_aligned_cropped_compressed.tar &&'
    #                     'rm -r %(stack)s_lossless_sorted_aligned_cropped_compressed.tar') %\
    #                     dict(stack=self.stack, data_dir=data_dir, stack_data_dir_gordon=self.stack_data_dir_gordon))
	# 			# 'ssh oasis-dm.sdsc.edu rm %(stack_data_dir_gordon)s/%(stack)s_lossless_sorted_aligned_cropped_compressed.tar') % \
    #
    #     # Download unsorted lossless aligned cropped data MANUALLY !!
    #
    #     # self.send_to_workstation()
    #
    #     self.save_everything()


    def update_sorted_sections_gscene_from_sorted_filenames(self):

        if not hasattr(self, 'currently_showing'):
            self.currently_showing = 'original'

        self.valid_section_filenames = self.get_valid_sorted_filenames()
        self.valid_section_indices = [self.sorted_filenames.index(fn) + 1 for fn in self.valid_section_filenames]

        if not hasattr(self, 'anchor_fn'):
            anchor_fp = DataManager.get_anchor_filename_filename(self.stack)
            # anchor_fp = os.path.join(self.stack_data_dir, '%(stack)s_anchor.txt' % dict(stack=self.stack))
            if os.path.exists(anchor_fp):
                with open(anchor_fp) as f:
                    self.set_anchor(f.readline().strip())
            else:
                from joblib import Parallel, delayed
                shapes = Parallel(n_jobs=16)(delayed(identify_shape)(os.path.join(RAW_DATA_DIR, self.stack, img_fn + '.' + self.tb_fmt)) for img_fn in self.valid_section_filenames)
                largest_idx = np.argmax([h*w for h, w in shapes])
                print 'largest section is ', self.valid_section_filenames[largest_idx]
                self.set_anchor(self.valid_section_filenames[largest_idx])
                print self.valid_section_filenames[largest_idx]

        if self.currently_showing == 'original':
            ordered_image_feeder_labels = range(1, len(self.sorted_filenames)+1)
            self.ordered_image_feeder = ImageDataFeeder('image feeder', stack=self.stack, sections=ordered_image_feeder_labels, use_data_manager=False)
            self.ordered_image_feeder.set_images(labels=ordered_image_feeder_labels,
                                                filenames=[os.path.join(self.thumbnails_dir, filename)
                                                        for filename in self.sorted_filenames],
                                                downsample=32)
            self.ordered_image_feeder.set_downsample_factor(32)

            if self.sorted_sections_gscene.active_i is not None:
                active_i = self.sorted_sections_gscene.active_i
            else:
                active_i = self.valid_section_indices[0]

            # self.sorted_sections_gscene.set_bad_sections(self.get_bad_sections())
            self.sorted_sections_gscene.set_data_feeder(self.ordered_image_feeder)
            self.sorted_sections_gscene.set_active_section(active_i)

        elif self.currently_showing == 'aligned':

            self.aligned_images_feeder = ImageDataFeeder('aligned image feeder', stack=self.stack,
                                                    sections=self.valid_section_indices, use_data_manager=False)

            aligned_image_filenames = [DataManager.get_image_filepath(stack=self.stack, fn=fn, anchor_fn=self.anchor_fn,
                                                                        resol='thumbnail', version='aligned_tif')
                                                                        for fn in self.valid_section_filenames]
            # print aligned_image_filenames
            self.aligned_images_feeder.set_images(self.valid_section_indices, aligned_image_filenames, downsample=32, load_with_cv2=False)
            self.aligned_images_feeder.set_downsample_factor(32)

            active_i = self.sorted_sections_gscene.active_i
            self.sorted_sections_gscene.set_data_feeder(self.aligned_images_feeder)
            self.sorted_sections_gscene.set_active_i(active_i)

        elif self.currently_showing == 'mask_contour':

            self.maskContourViz_images_feeder = ImageDataFeeder('mask contoured image feeder', stack=self.stack,
                                                sections=self.valid_section_indices, use_data_manager=False)
            self.maskContourViz_images_dir = self.stack_data_dir + '/%(stack)s_maskContourViz_unsorted' % {'stack': self.stack}
            # aligned_image_filenames = [os.path.join(self.aligned_images_dir, '%(stack)s_%(fn)s_aligned.tif' % \
            #                             {'stack':self.stack, 'fn': self.sorted_filenames[i]}) for i in self.valid_section_indices]

            maskContourViz_image_filenames = [os.path.join(self.maskContourViz_images_dir, '%(fn)s_mask_contour_viz.tif' % {'fn': fn})
                                        for fn in self.valid_section_filenames]

            self.maskContourViz_images_feeder.set_images(self.valid_section_indices, maskContourViz_image_filenames, downsample=32, load_with_cv2=False)
            self.maskContourViz_images_feeder.set_downsample_factor(32)

            active_i = self.sorted_sections_gscene.active_i
            self.sorted_sections_gscene.set_data_feeder(self.maskContourViz_images_feeder)
            self.sorted_sections_gscene.set_active_i(active_i)


    def save_sorted_filenames(self):

        # dump to disk
        with open(self.stack_data_dir + '/%(stack)s_sorted_filenames.txt' % {'stack': self.stack}, 'w') as f:
            for i, fn in enumerate(self.sorted_filenames):
                f.write(fn + ' ' + str(i+1) + '\n') # index starts from 1

        sys.stderr.write('Sorted filename list saved.\n')
        self.statusBar().showMessage('Sorted filename list saved.')


    def load_sorted_filenames(self):

        filename_to_section, section_to_filename = DataManager.load_sorted_filenames(self.stack)
        self.sorted_filenames = section_to_filename.values()
        # self.sorted_filenames = []
        # with open(self.stack_data_dir + '/%(stack)s_sorted_filenames.txt' % {'stack': self.stack}, 'r') as f:
        #     for line in f.readlines():
        #         fn, idx = line.split()
        #         self.sorted_filenames.append(fn)
        sys.stderr.write('Sorted filename list is loaded.\n')
        self.statusBar().showMessage('Sorted filename list is loaded.')

        # self.fn_to_slide_position = {fn: (slide, pos) for slide, pos_to_fn in self.slide_position_to_fn.iteritems() for pos, fn in pos_to_fn.iteritems()}
        # self.sorted_slide_positions = [self.fn_to_slide_position[fn] for fn in self.sorted_filenames]

        self.update_sorted_sections_gscene_from_sorted_filenames()

    def load_slide_position_map(self):
        fn = self.stack_data_dir + '/%(stack)s_slide_position_to_fn.pkl' % {'stack': self.stack}
        if os.path.exists(fn):
            self.slide_position_to_fn = pickle.load(open(fn, 'r'))
            sys.stderr.write('Slide position to image filename mapping is loaded.\n')
            self.statusBar().showMessage('Slide position to image filename mapping is loaded.')
        else:
            sys.stderr.write('Cannot load slide position to image filename mapping - File does not exists.\n')
            self.statusBar().showMessage('Cannot load slide position to image filename mapping - File does not exists.')

    def save(self):

        pickle.dump(self.slide_position_to_fn, open(self.stack_data_dir + '/%(stack)s_slide_position_to_fn.pkl' % {'stack': self.stack}, 'w') )



    # def get_bad_sections(self):
    #     # return bad section indices, in sorted list
    #     return [idx+1 for idx, fn in enumerate(self.sorted_filenames) if fn == 'Placeholder' or fn == 'Rescan']

    def download(self):

        # Download thumbnails from Gordon
        execute_command("""mkdir %(local_data_dir)s/%(stack)s; scp oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s/*.%(tb_fmt)s %(local_data_dir)s/%(stack)s/""" % \
                        {'gordon_data_dir': GORDON_RAW_DATA_DIR,
                        'local_data_dir': RAW_DATA_DIR,
                        'stack': self.stack,
                        'tb_fmt': self.tb_fmt})

        # Download macros (annotated with bounding boxes)
        execute_command("""scp -r oasis-dm.sdsc.edu:%(gordon_data_dir)s/macros_annotated/%(stack)s/ %(local_data_dir)s/macros_annotated/""" % \
                        {'gordon_data_dir': GORDON_RAW_DATA_DIR,
                        'local_data_dir': RAW_DATA_DIR,
                        'stack': self.stack})

        # Download macros (without bounding boxes).
        execute_command("""scp -r oasis-dm.sdsc.edu:%(gordon_data_dir)s/macros/%(stack)s/ %(local_data_dir)s/macros/""" % \
                        {'gordon_data_dir': GORDON_RAW_DATA_DIR,
                        'local_data_dir': RAW_DATA_DIR,
                        'stack': self.stack})

    # def generate_masks(self):
    #
    #     self.web_service.convert_to_request('generate_masks', stack=self.stack,
    #                                         filenames=self.get_valid_sorted_filenames(),
    #                                         tb_fmt=self.tb_fmt)
    #
    #     execute_command("rm -rf %(local_data_dir)s/%(stack)s/%(stack)s_submasks && scp -r oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s/%(stack)s_submasks %(local_data_dir)s/%(stack)s/" % \
    #                     {'gordon_data_dir': gordon_thumbnail_data_dir,
    #                     'local_data_dir': thumbnail_data_dir,
    #                     'stack': self.stack})
    #
    # def warp_crop_masks(self):
    #     ul_pos = self.sorted_sections_gscene.corners['ul'].scenePos()
    #     lr_pos = self.sorted_sections_gscene.corners['lr'].scenePos()
    #     ul_x = int(ul_pos.x())
    #     ul_y = int(ul_pos.y())
    #     lr_x = int(lr_pos.x())
    #     lr_y = int(lr_pos.y())
    #
    #     self.web_service.convert_to_request('warp_crop_masks',
    #                                         stack=self.stack, filenames=self.get_valid_sorted_filenames(),
    #                                         x=ul_x, y=ul_y, w=lr_x+1-ul_x, h=lr_y+1-ul_y, anchor_fn=self.anchor_fn)
    #
    #     execute_command("""rm -rf %(gordon_data_dir)s/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s && scp -r oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s %(local_data_dir)s/%(stack)s/""" % \
    #                     {'gordon_data_dir': gordon_thumbnail_data_dir,
    #                     'local_data_dir': thumbnail_data_dir,
    #                     'anchor_fn': self.anchor_fn,
    #                     'stack': self.stack})
    #
    #     execute_command("""rm -rf %(gordon_data_dir)s/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped && scp -r oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped %(local_data_dir)s/%(stack)s/""" % \
    #                     {'gordon_data_dir': gordon_thumbnail_data_dir,
    #                     'local_data_dir': thumbnail_data_dir,
    #                     'anchor_fn': self.anchor_fn,
    #                     'stack': self.stack})


    # def align(self):
    #     pass
        # self.web_service.convert_to_request('align', stack=self.stack, filenames=self.get_valid_sorted_filenames())

        ## SSH speed is not stable. Performance is alternating: one 5MB/s, the next 800k/s, the next 5MB/s again.
        # execute_command(('ssh gcn-20-34.sdsc.edu \"cd %(gordon_data_dir)s && tar -I pigz -cf %(stack)s_elastix_output.tar.gz %(stack)s_elastix_output/*/*.tif\" &&'
        #                 'scp oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s_elastix_output.tar.gz %(local_data_dir)s/ &&'
        #                 'cd %(local_data_dir)s && rm -rf %(stack)s_elastix_output && tar -xf %(stack)s_elastix_output.tar.gz && rm %(stack)s_elastix_output.tar.gz &&'
        #                 'ssh gcn-20-34.sdsc.edu rm %(gordon_data_dir)s/%(stack)s_elastix_output.tar.gz') % \
        #                 dict(gordon_data_dir=self.stack_data_dir_gordon,
        #                     local_data_dir=self.stack_data_dir,
        #                     stack=self.stack))

        # download()
        #
        # self.statusBar().showMessage('Consecutive sections alignment results downloaded.')

    def get_valid_sorted_filenames(self):
        return [fn for fn in self.sorted_filenames if fn != 'Rescan' and fn != 'Placeholder']

    def compose(self):

        with open(os.path.join(thumbnail_data_dir, self.stack, self.stack + '_anchor.txt'), 'w') as f:
            f.write(self.anchor_fn)

        # if self.stack in all_nissl_stacks:
        #     pad_bg_color = 'white'
        # elif self.stack in all_ntb_stacks:
        #     pad_bg_color = 'black'
        # elif self.stack in all_alt_nissl_ntb_stacks or self.stack in all_alt_nissl_tracing_stacks:
        #     pad_bg_color = 'auto'
        #
        # self.statusBar().showMessage('Conmpose consecutive alignments...')
        # try:
        #     self.web_service.convert_to_request(name='compose', stack=self.stack,
        #                                         filenames=self.get_valid_sorted_filenames(),
        #                                         anchor_fn=self.anchor_fn,
        #                                         tb_fmt=self.tb_fmt,
        #                                         pad_bg_color=pad_bg_color)
        # except Exception as e:
        #     sys.stderr.write('Server error: compose\n')
        #     return
        #
        # self.statusBar().showMessage('Images aligned.')
        # self.statusBar().showMessage('Downloading aligned images ...')
        #
        # execute_command(('ssh gcn-20-34.sdsc.edu \"cd %(gordon_data_dir)s && tar -I pigz -cf %(stack)s_thumbnails_alignedTo_%(anchor_fn)s.tar.gz %(stack)s_thumbnails_alignedTo_%(anchor_fn)s/*.tif\" && '
        #                 'scp oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s_thumbnails_alignedTo_%(anchor_fn)s.tar.gz %(local_data_dir)s/ &&'
        #                 'cd %(local_data_dir)s && rm -rf %(stack)s_thumbnails_alignedTo_%(anchor_fn)s && tar -xf %(stack)s_thumbnails_alignedTo_%(anchor_fn)s.tar.gz && rm %(stack)s_thumbnails_alignedTo_%(anchor_fn)s.tar.gz && '
        #                 'scp oasis-dm.sdsc.edu:%(gordon_data_dir)s/%(stack)s_elastix_output/%(stack)s_transformsTo_%(anchor_fn)s.pkl %(stack)s_elastix_output/ && '
        #                 'cd %(stack)s_elastix_output && rm -f %(stack)s_transformsTo_anchor.pkl && ln -s %(stack)s_transformsTo_%(anchor_fn)s.pkl %(stack)s_transformsTo_anchor.pkl ') % \
        #                 dict(gordon_data_dir=self.stack_data_dir_gordon,
        #                     local_data_dir=self.stack_data_dir,
        #                     stack=self.stack,
        #                     anchor_fn=self.anchor_fn))
        #
        # self.statusBar().showMessage('Aligned images downloaded.')


    def set_first_section(self, i):
        self.first_section = i
        if self.last_section <= self.first_section:
            self.last_section = self.first_section + 1
        self.update_sorted_sections_gscene_label()

    def set_last_section(self, i):
        self.last_section = i
        if self.last_section <= self.first_section:
            self.first_section = self.last_section - 1
        self.update_sorted_sections_gscene_label()

    def set_anchor(self, anchor):
        if isinstance(anchor, int):
            self.anchor_fn = self.sorted_filenames[anchor-1]
        elif isinstance(anchor, str):
            # assert isinstance(anchor, str)
            self.anchor_fn = anchor

        with open(os.path.join(thumbnail_data_dir, self.stack, self.stack + '_anchor.txt'), 'w') as f:
            f.write(self.anchor_fn)

        self.update_sorted_sections_gscene_label()

    def sorted_sections_image_updated(self):
        # print self.sorted_filenames[self.sorted_sections_gscene.active_section]
        filename = self.sorted_filenames[self.sorted_sections_gscene.active_section-1]
        self.label_sorted_sections_filename.setText(filename)
        self.label_sorted_sections_index.setText(str(self.sorted_sections_gscene.active_section))
        if filename == 'Placeholder' or filename == 'Rescan':
            return
        assert filename != 'Unknown' and filename != 'Nonexisting'

        # Update slide scene
        slide_name = self.filename_to_slide[filename]
        position = self.slide_position_to_fn[slide_name].keys()[self.slide_position_to_fn[slide_name].values().index(filename)]
        self.slide_gscene.set_active_section(slide_name)

        self.update_sorted_sections_gscene_label()


    def update_sorted_sections_gscene_label(self):

        # print self.sorted_sections_gscene.active_section, self.anchor_fn
        # if self.sorted_sections_gscene.active_section is not None:
        #     print self.sorted_filenames[self.sorted_sections_gscene.active_section-1]

        if self.sorted_sections_gscene.active_section == self.first_section:
            self.label_sorted_sections_status.setText('FIRST')
        elif self.sorted_sections_gscene.active_section == self.last_section:
            self.label_sorted_sections_status.setText('LAST')
        elif hasattr(self, 'anchor_fn') and self.sorted_sections_gscene.active_section is not None and \
            self.sorted_filenames[self.sorted_sections_gscene.active_section-1] == self.anchor_fn:
            self.label_sorted_sections_status.setText('ANCHOR')
        else:
            self.label_sorted_sections_status.setText('')

    def set_status(self, slide_name, position, fn):
        """
        Update slide_position_to_fn variables.
        If active, change content and captions of the specified slide position gscene.
        """

        # old_fn = self.slide_position_to_fn[slide_name][position]
        self.slide_position_to_fn[slide_name][position] = fn

        # # if slide_name == 'N_92':
        # print position
        # print self.slide_position_gscenes[position].data_feeder.all_sections
        # print self.section_image_feeders[slide_name].all_sections

        if slide_name == self.slide_gscene.active_section:
            self.slide_position_gscenes[position].set_active_section(fn)
            self.labels_slide_position_filename[position].setText(fn)

            if hasattr(self, 'sorted_filenames'):
                if fn == 'Placeholder' or fn == 'Rescan' or fn == 'Nonexisting':
                    self.labels_slide_position_index[position].setText('')
                else:
                    if fn in self.sorted_filenames:
                        self.labels_slide_position_index[position].setText(str(self.sorted_filenames.index(fn)+1))
                    else:
                        self.labels_slide_position_index[position].setText('Not in sorted list.')


    def slide_image_updated(self):
        self.setWindowTitle('Slide %(slide_index)s' % {'slide_index': self.slide_gscene.active_section})

        slide_name = self.slide_gscene.active_section
        feeder = self.section_image_feeders[slide_name]

        if slide_name not in self.slide_position_to_fn:
            self.slide_position_to_fn[slide_name] = {p: 'Unknown' for p in [1,2,3]}

        for position, gscene in self.slide_position_gscenes.iteritems():

            gscene.set_data_feeder(feeder)

            if self.slide_position_to_fn[slide_name][position] != 'Unknown':
                self.set_status(slide_name, position, self.slide_position_to_fn[slide_name][position])
            else:
                if position in self.thumbnail_filenames[slide_name]:
                    newest_fn = sorted(self.thumbnail_filenames[slide_name][position].items())[-1][1]
                    self.set_status(slide_name, position, newest_fn)
                else:
                    self.set_status(slide_name, position, 'Nonexisting')

    def show_option_changed(self, index):

        show_option_text = str(self.sender().currentText())
        if show_option_text == 'Original Aligned':
            self.set_show_option('aligned')
        elif show_option_text == 'Original':
            self.set_show_option('original')
        elif show_option_text == 'Mask Contoured':
            self.set_show_option('mask_contour')
        else:
            raise Exception('Not implemented.')

    def set_show_option(self, showing):

        if self.currently_showing == showing:
            return
        else:
            self.currently_showing = showing
            print self.currently_showing
            self.update_sorted_sections_gscene_from_sorted_filenames()

    def slide_position_image_updated(self):
        position = self.sender().id
        # if self.slide_position_gscenes[position].active_section is not None:
        #     self.slide_position_to_fn[self.slide_gscene.active_section][position] = self.slide_position_gscenes[position].active_section
        self.set_status(self.slide_gscene.active_section, position, self.slide_position_gscenes[position].active_section)


    def eventFilter(self, obj, event):

        if event.type() == QEvent.GraphicsSceneMousePress:
            pass

        elif event.type() == QEvent.KeyPress:
            key = event.key()
        return False


if __name__ == "__main__":

    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Data Preprocessing GUI.')

    parser.add_argument("stack_name", type=str, help="stack name")
    parser.add_argument("--tb_fmt", type=str, help="thumbnail format", default='png')
    args = parser.parse_args()

    from sys import argv, exit
    app = QApplication(argv)

    m = PreprocessGUI(stack=args.stack_name, tb_fmt=args.tb_fmt)

    # m.show()
    m.showMaximized()
    # m.raise_()
    exit(app.exec_())
