#! /usr/bin/env python

import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

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

from ui_PreprocessGui import Ui_PreprocessGui
from ui_GalleryDialog import Ui_gallery_dialog

import cPickle as pickle

import sys, os
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

from DataFeeder import ImageDataFeeder
from drawable_gscene import *
from zoomable_browsable_gscene import SimpleGraphicsScene

from gui_utilities import *

gray_color_table = [qRgb(i, i, i) for i in range(256)]

class SimpleGraphicsScene3(SimpleGraphicsScene):

    bad_status_changed = pyqtSignal(int)
    move_forward_requested = pyqtSignal()
    move_backward_requested = pyqtSignal()

    def __init__(self, id, gview=None, parent=None):
        super(SimpleGraphicsScene3, self).__init__(id=id, gview=gview, parent=parent)

        self.bad_cross = QGraphicsSimpleTextItem(QString('X'), scene=self)
        self.bad_cross.setPos(50,50)
        self.bad_cross.setScale(5)
        self.bad_cross.setVisible(False)
        self.bad_cross.setBrush(Qt.red)

    def set_bad_sections(self, secs):
        self.bad_sections = secs

    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)

        is_bad = self.active_section in self.bad_sections
        action_setBad = myMenu.addAction("Unmark as Bad" if is_bad else "Mark as Bad")
        action_moveForward = myMenu.addAction("Move forward")
        action_moveBackward = myMenu.addAction("Move backward")

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))

        if selected_action == action_setBad:
            if is_bad:
                self.bad_cross.setVisible(False)
                self.bad_status_changed.emit(0)
            else:
                self.bad_cross.setVisible(True)
                self.bad_status_changed.emit(1)
        elif selected_action == action_moveForward:
            self.move_forward_requested.emit()
        elif selected_action == action_moveBackward:
            self.move_backward_requested.emit()

    def set_active_i(self, i, emit_changed_signal=True):
        super(SimpleGraphicsScene3, self).set_active_i(i, emit_changed_signal=True)

        # Determine whether to show bad section indicator
        if self.active_section in self.bad_sections:
            self.bad_cross.setVisible(True)
        else:
            self.bad_cross.setVisible(False)
        print self.bad_sections

    def eventFilter(self, obj, event):
        # print obj.metaObject().className(), event.type()
        # http://doc.qt.io/qt-4.8/qevent.html#Type-enum

        if event.type() == QEvent.KeyPress:
            key = event.key()
            pass

        elif event.type() == QEvent.Wheel:
            out_factor = .9
            in_factor = 1. / out_factor
            if event.delta() < 0: # negative means towards user
                self.gview.scale(out_factor, out_factor)
            else:
                self.gview.scale(in_factor, in_factor)
            return True

        # if event.type() == QEvent.GraphicsSceneMousePress:

            # self.gui.active_gscene = self
            # self.gscene_clicked.emit(self, self.id)

            # pos = event.scenePos()
            # gscene_x = pos.x()
            # gscene_y = pos.y()

            # if event.button() == Qt.RightButton:
            # obj.mousePressEvent(event)

        return False

# class SimpleDrawableGraphicsScene2(DrawableGraphicsScene):
#
#     # bad_status_changed = pyqtSignal(int)
#
#     def __init__(self, id, gui=None, gview=None, parent=None):
#         super(SimpleDrawableGraphicsScene2, self).__init__(id=id, gui=gui, gview=gview, parent=parent)
#
#     def show_context_menu(self, pos):
#         myMenu = QMenu(self.gview)

        # is_bad = self.sorted_filenames[self.active_section-1] in self.gui.bad_sections
        # action_setBad = myMenu.addAction("Unmark as Bad" if is_bad else "Mark as Bad")
        # action_moveBefore = myMenu.addAction("Move before")
        # action_moveAfter = myMenu.addAction("Move after")

        # selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))
        #
        # if selected_action == action_setBad:
        #     self.bad_status_changed.emit(is_bad)
            # status = actions_setStatus[selected_action]
            # self.status_updated.emit(self.id, status)
        # elif selected_action == action_moveBefore:
        #     self.move_before()
        #
        # elif selected_action == action_moveAfter:
        #     self.move_after()


class SimpleDrawableGraphicsScene(DrawableGraphicsScene):

    status_updated = pyqtSignal(int, str)

    def __init__(self, id, gui=None, gview=None, parent=None):
        super(SimpleDrawableGraphicsScene, self).__init__(id=id, gui=gui, gview=gview, parent=parent)

    def show_context_menu(self, pos):
        myMenu = QMenu(self.gview)

        setStatus_menu = QMenu("Set to", myMenu)
        myMenu.addMenu(setStatus_menu)
        action_setNormal = setStatus_menu.addAction('Normal')
        action_setRescan = setStatus_menu.addAction('Rescan')
        action_setPlaceholder = setStatus_menu.addAction('Placeholder')
        action_setNonexisting = setStatus_menu.addAction('Nonexisting')
        actions_setStatus = {action_setNormal: 'Normal', action_setRescan: 'Rescan',
                            action_setPlaceholder: 'Placeholder', action_setNonexisting: 'Nonexisting'}

        selected_action = myMenu.exec_(self.gview.viewport().mapToGlobal(pos))

        if selected_action in actions_setStatus:
            status = actions_setStatus[selected_action]
            self.status_updated.emit(self.id, status)

# Use the third method in http://pyqt.sourceforge.net/Docs/PyQt4/designer.html
class PreprocessGUI(QMainWindow, Ui_PreprocessGui):
    def __init__(self, parent=None, stack=None):
        """
        Initialization of preprocessing tool.
        """
        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.setupUi(self)

        self.stack = stack

        if stack in bad_sections:
            self.bad_sections = set(bad_sections[stack])
        else:
            self.bad_sections = set([])

        ###############################################

        self.slide_gscene = DrawableGraphicsScene(id='section', gui=self, gview=self.slide_gview)
        self.slide_gview.setScene(self.slide_gscene)

        # slide_indices = ['N11, N12, IHC28']
        macros_dir = '/home/yuncong/CSHL_data/macros_annotated/%(stack)s/' % {'stack': self.stack}

        # slide_indices = defaultdict(list)
        slide_filenames = {}
        # macro_fns = []
        import re
        for fn in os.listdir(macros_dir):
            _, prefix, slide_num, date, hour = re.findall('^(.*?)\s?-\s?(F|N|IHC)\s?([0-9]+)\s?-\s?(.*?) (.*?)_macro_annotated.jpg$', fn)[0]
            # slide_indices[prefix].append(slide_num)
            slide_filenames[prefix + '_%02d' % int(slide_num)] = fn

        # print sorted(slide_filenames.keys())

        slide_image_feeder = ImageDataFeeder('image feeder', stack=self.stack, sections=sorted(slide_filenames.keys()), use_data_manager=False)
        slide_image_feeder.set_downsample_factor(32)
        slide_image_feeder.set_orientation(stack_orientation[self.stack])

        for slide_index, filename in slide_filenames.iteritems():
            qimage = QImage(os.path.join(macros_dir, filename))
            slide_image_feeder.set_image(qimage, slide_index)

        self.slide_gscene.set_data_feeder(slide_image_feeder)
        self.slide_gscene.set_active_section(sorted(slide_filenames.keys())[0])
        self.slide_gscene.active_image_updated.connect(self.slide_image_updated)

        ################################################

        # self.slide_position_to_fn = defaultdict(lambda: defaultdict(lambda: 'Unknown'))
        fn = '%(stack)s_slide_position_to_fn.pkl' % {'stack':stack}
        if os.path.exists(fn):
            self.slide_position_to_fn = pickle.load(open(fn, 'r'))
            self.statusBar().showMessage('Slide position to image filename mapping is loaded.')
        else:
            self.slide_position_to_fn = {slide_index: {p: 'Unknown' for p in [1,2,3]}
                                        for slide_index in slide_filenames.iterkeys()}

        ###############

        self.filename_to_slide = {}

        self.thumbnails_dir = '/home/yuncong/CSHL_data/%(stack)s/' % {'stack': self.stack}
        from glob import glob
        # self.thumbnail_filenames = defaultdict(list)
        self.thumbnail_filenames = defaultdict(lambda: defaultdict(list))
        for fp in glob(self.thumbnails_dir+'/*.tif'):
            fn = os.path.splitext(os.path.basename(fp))[0]
            _, prefix, slide_num, date, hour, _, position, index = re.findall('^(.*?)-([A-Z]+)([0-9]+)-(.*?)-(.*?)_(.*?)_([0-9])_([0-9]{4})$', fn)[0]
            # print prefix, slide_num, position, index
            slide_name = prefix + '_%02d' % int(slide_num)
            self.filename_to_slide[fn] = slide_name
            # print fn
            self.thumbnail_filenames[slide_name][int(position)].append(fn)
            # self.thumbnail_filenames[prefix + '_%02d' % int(slide_num)].append(fn)
            # if fn == 'MD589-N1-2015.07.30-16.19.59_MD589_2_0002':
            #     print 1

        ################

        self.section1_gscene = SimpleDrawableGraphicsScene(id=1, gui=self, gview=self.section1_gview)
        self.section1_gview.setScene(self.section1_gscene)
        self.section1_gscene.active_image_updated.connect(self.slide_position_image_updated)
        self.section2_gscene = SimpleDrawableGraphicsScene(id=2, gui=self, gview=self.section2_gview)
        self.section2_gview.setScene(self.section2_gscene)
        self.section2_gscene.active_image_updated.connect(self.slide_position_image_updated)
        self.section3_gscene = SimpleDrawableGraphicsScene(id=3, gui=self, gview=self.section3_gview)
        self.section3_gview.setScene(self.section3_gscene)
        self.section3_gscene.active_image_updated.connect(self.slide_position_image_updated)

        self.section1_gscene.status_updated.connect(self.slide_position_status_updated)
        self.section2_gscene.status_updated.connect(self.slide_position_status_updated)
        self.section3_gscene.status_updated.connect(self.slide_position_status_updated)

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

            filenames = list(chain(*self.thumbnail_filenames[slide_index].values()))

            section_image_feeder = ImageDataFeeder('image feeder', stack=self.stack,
                                                    sections=filenames + ['Nonexisting', 'Rescan', 'Placeholder'],
                                                    use_data_manager=False)
            section_image_feeder.set_downsample_factor(32)
            section_image_feeder.set_orientation(stack_orientation[self.stack])

            for fn in filenames:
                qimage = QImage(os.path.join(self.thumbnails_dir, fn + '.tif'))
                section_image_feeder.set_image(qimage, fn)
            section_image_feeder.set_image(self.nonexisting_qimage, 'Nonexisting')
            section_image_feeder.set_image(self.rescan_qimage, 'Rescan')
            section_image_feeder.set_image(self.placeholder_qimage, 'Placeholder')

            self.section_image_feeders[slide_index] = section_image_feeder


        #######################

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
        self.sorted_sections_gscene.bad_status_changed.connect(self.bad_status_changed)
        self.sorted_sections_gscene.move_forward_requested.connect(self.move_forward)
        self.sorted_sections_gscene.move_backward_requested.connect(self.move_backward)

        #######################

        # self.orderedSection1_gscene = DrawableGraphicsScene(id='previous', gui=self, gview=self.orderedSection1_gview)
        # self.orderedSection1_gview.setScene(self.orderedSection1_gscene)
        # self.orderedSection2_gscene = DrawableGraphicsScene(id='current', gui=self, gview=self.orderedSection2_gview)
        # self.orderedSection2_gview.setScene(self.orderedSection2_gscene)
        # self.orderedSection3_gscene = DrawableGraphicsScene(id='next', gui=self, gview=self.orderedSection3_gview)
        # self.orderedSection3_gview.setScene(self.orderedSection3_gscene)
        #
        # self.orderedSection1_gscene.active_image_updated.connect(self.gallery_image_updated)
        # self.orderedSection2_gscene.active_image_updated.connect(self.gallery_image_updated)
        # self.orderedSection3_gscene.active_image_updated.connect(self.gallery_image_updated)

        # self.orderedSection1_gscene.set_data_feeder(ordered_image_feeder)
        # self.orderedSection1_gscene.set_active_section(1)
        # self.orderedSection2_gscene.set_data_feeder(ordered_image_feeder)
        # self.orderedSection2_gscene.set_active_section(2)
        # self.orderedSection3_gscene.set_data_feeder(ordered_image_feeder)
        # self.orderedSection3_gscene.set_active_section(3)


        #######################

        self.installEventFilter(self)

        # self.button_save.clicked.connect(self.save)
        # self.button_load.clicked.connect(self.load)
        # self.button_bad.clicked.connect(self.set_bad)
        self.comboBox_show.currentIndexChanged.connect(self.show_option_changed)

        # self.comboBox_status1.currentIndexChanged.connect(self.status_changed)
        # self.comboBox_status2.currentIndexChanged.connect(self.status_changed)
        # self.comboBox_status3.currentIndexChanged.connect(self.status_changed)

        # self.button_moveBefore.clicked.connect(self.moveBefore_clicked)
        # self.button_moveAfter.clicked.connect(self.moveAfter_clicked)
        # self.button_previous.clicked.connect(self.previous_clicked)
        # self.button_next.clicked.connect(self.next_clicked)
        self.button_save.clicked.connect(self.save)
        self.button_sort.clicked.connect(self.sort)
        # self.button_gallery.clicked.connect(self.show_gallery)

        ################################

        self.placeholders = set([])
        self.rescans = set([])

        # self.status_comboBoxes = {1: self.comboBox_status1, 2: self.comboBox_status2, 3: self.comboBox_status3}
        self.filename_labels = {1: self.label_section1_filename, 2: self.label_section2_filename, 3: self.label_section3_filename}


        #######################################

        # self.bad_cross = QGraphicsSimpleTextItem(QString('X'), scene=self.sorted_sections_gscene)
        # # self.section_gscene.addItem()
        # self.bad_cross.setPos(50,50)
        # self.bad_cross.setScale(5)
        # self.bad_cross.setVisible(False)
        # self.bad_cross.setBrush(Qt.red)

    # def set_bad(self, is_bad):
    #     fn = self.sorted_filenames[self.sorted_sections_gscene.active_section-1]
    #     if is_bad:
    #         self.bad_sections.add(fn)
    #         self.bad_cross.setVisible(True)
    #     else:
    #         self.bad_sections.remove(fn)
    #         self.bad_cross.setVisible(False)

    # def show_gallery(self):
    #     self.gallery_dialog = QDialog()
    #     ui = Ui_gallery_dialog()
    #     ui.setupUi(self.gallery_dialog)
    #
    #     ui.button_moveAfter.clicked.connect(self.move_after)
    #     ui.button_moveBefore.clicked.connect(self.move_before)
    #
    #     layout = QGridLayout()
    #
    #     for i, sec in enumerate(range(20,40)):
    #         col = i % 5
    #         row = i / 5
    #         gview = QGraphicsView(self.gallery_dialog)
    #         gscene = SimpleDrawableGraphicsScene3(id=i, gui=self, gview=gview, parent=gview)
    #         gscene.gscene_clicked.connect(self.gallery_gscene_clicked)
    #         gscene.move_forward_requested.connect(self.move_forward)
    #         gscene.move_backward_requested.connect(self.move_backward)
    #         gview.setScene(gscene)
    #         gscene.set_data_feeder(self.ordered_image_feeder)
    #         gscene.set_active_section(sec)
    #         layout.addWidget(gview, row, col)
    #
    #     ui.gallery_widget.setLayout(layout)
    #
    #     self.gallery_dialog.show()


    def move_backward(self):

        # gallery_gscene.id


        curr_index = self.sorted_sections_gscene.active_section
        print 'old index filename', self.sorted_filenames[curr_index-1]
        if curr_index == 1:
            return

        new_index = curr_index - 1

        self.sorted_filenames[new_index-1], self.sorted_filenames[curr_index-1] = self.sorted_filenames[curr_index-1], self.sorted_filenames[new_index-1]

        data_feeder = self.sorted_sections_gscene.data_feeder

        image1 = data_feeder.retrive_i(sec=curr_index)
        image2 = data_feeder.retrive_i(sec=new_index)
        data_feeder.set_image(image1, new_index)
        data_feeder.set_image(image2, curr_index)

        self.sorted_sections_gscene.set_active_section(new_index)
        print 'new index filename', self.sorted_filenames[new_index-1]

        # feeder = self.orderedSection2_gscene.data_feeder
        # tmp = feeder.image_cache[feeder.downsample][curr_index]
        # feeder.image_cache[feeder.downsample][curr_index] = feeder.image_cache[feeder.downsample][new_index]
        # feeder.image_cache[feeder.downsample][new_index] = tmp

        # self.orderedSection2_gscene.set_active_section(new_index)

    def move_forward(self):


        curr_index = self.sorted_sections_gscene.active_section
        print 'old index filename', self.sorted_filenames[curr_index-1]
        if curr_index == len(self.sorted_filenames)-1:
            return

        new_index = curr_index + 1

        # swap two filenames
        self.sorted_filenames[new_index-1], self.sorted_filenames[curr_index-1] = self.sorted_filenames[curr_index-1], self.sorted_filenames[new_index-1]

        data_feeder = self.sorted_sections_gscene.data_feeder

        # swap two images in data_feeder queue
        image1 = data_feeder.retrive_i(sec=curr_index)
        image2 = data_feeder.retrive_i(sec=new_index)
        data_feeder.set_image(image1, new_index)
        data_feeder.set_image(image2, curr_index)

        self.sorted_sections_gscene.set_active_section(new_index)
        print 'new index filename', self.sorted_filenames[new_index-1]

    # def gallery_gscene_clicked(self, gscene, id):

    # def gallery_gscene_move_forward_requested(self, gscene, forward_or_backward):
    #     if forward_or_backward == 0:
    #         # forward
    #         self.move_after()
    #     else:
    #         # backward
    #         self.move_before()

    def bad_status_changed(self, is_bad):
        fn = self.sorted_filenames[self.sorted_sections_gscene.active_section-1]
        if is_bad > 0:
            self.bad_sections.add(fn)
        else:
            self.bad_sections.remove(fn)
        self.sorted_sections_gscene.set_bad_sections([self.sorted_filenames.index(fn)+1 for fn in self.bad_sections])

    def slide_position_status_updated(self, position, status):
        slide_name = self.slide_gscene.active_section
        # position = self.slide_position_gscenes[position].active_section
        if status == 'Normal':
            if position in self.thumbnail_filenames[slide_name]:
                self.set_status(slide_name, position, self.thumbnail_filenames[slide_name][position][0])
            else:
                arbitrary_image = list(chain(*self.thumbnail_filenames[slide_name].values()))[0]
                self.set_status(slide_name, position, arbitrary_image)
        else:
            self.set_status(slide_name, position, status)


    def sort(self):

        prefixes = set([slide_name.split('_')[0] for slide_name in self.slide_position_to_fn.iterkeys()])
        if len(prefixes) == 2: # IHC and N
            IHC_series = {int(slide_name.split('_')[1]): x for slide_name, x in self.slide_position_to_fn.items() if slide_name.split('_')[0] == 'IHC'}
            N_series = {int(slide_name.split('_')[1]): x for slide_name, x in self.slide_position_to_fn.items() if slide_name.split('_')[0] == 'N'}
        elif len(prefixes) == 1:
            raise Exception('Not implemented.')

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

        self.sorted_filenames = [fn for fn in sorted_fns if fn != 'Nonexisting']
        with open('%(stack)s_sorted_filenames.txt' % {'stack': self.stack}, 'w') as f:
            for i, fn in enumerate(self.sorted_filenames):
                f.write(fn + ' ' + str(i+1) + '\n') # index starts from 1

        self.ordered_image_feeder = ImageDataFeeder('image feeder', stack=self.stack, sections=range(1, len(self.sorted_filenames)+1), use_data_manager=False)
        self.ordered_image_feeder.set_downsample_factor(32)
        self.ordered_image_feeder.set_orientation(stack_orientation[self.stack])

        for i, filename in enumerate(self.sorted_filenames):
            qimage = QImage(os.path.join(self.thumbnails_dir, filename))
            self.ordered_image_feeder.set_image(qimage, i+1)

        self.sorted_sections_gscene.set_bad_sections([self.sorted_filenames.index(fn)+1 for fn in self.bad_sections])
        self.sorted_sections_gscene.set_data_feeder(self.ordered_image_feeder)
        # self.orderedSection1_gscene.set_data_feeder(self.ordered_image_feeder)
        # self.orderedSection2_gscene.set_data_feeder(self.ordered_image_feeder)
        # self.orderedSection3_gscene.set_data_feeder(self.ordered_image_feeder)
        self.sorted_sections_gscene.set_active_section(2)
        # self.orderedSection1_gscene.set_active_section(1)
        # self.orderedSection2_gscene.set_active_section(2)
        # self.orderedSection1_gscene.set_active_section(3)

    def save(self):
        for a, b in sorted(self.slide_position_to_fn.items()):
            for c, d in sorted(b.items()):
                print a, c, d

        pickle.dump(self.slide_position_to_fn, open('%(stack)s_slide_position_to_fn.pkl' % {'stack': self.stack}, 'w') )

    # def previous_clicked(self):
    #     self.sorted_sections_gscene.show_previous()
    #     self.orderedSection2_gscene.show_previous()
        # self.sorted_sections_gscene.set_active_section(self.sorted_sections_gscene.active_section-1)
        # self.orderedSection1_gscene.set_active_section(self.orderedSection1_gscene.active_section-1)
        # self.orderedSection2_gscene.set_active_section(self.orderedSection2_gscene.active_section-1)
        # self.orderedSection3_gscene.set_active_section(self.orderedSection3_gscene.active_section-1)

    # def next_clicked(self):
    #     self.sorted_sections_gscene.show_next()
    #     self.orderedSection2_gscene.show_next()
        # self.sorted_sections_gscene.set_active_section(self.sorted_sections_gscene.active_section+1)
        # self.orderedSection1_gscene.set_active_section(self.orderedSection1_gscene.active_section+1)
        # self.orderedSection2_gscene.set_active_section(self.orderedSection2_gscene.active_section+1)
        # self.orderedSection3_gscene.set_active_section(self.orderedSection3_gscene.active_section+1)



    # def status_changed(self):
    #     sender = self.sender()
    #     slide_name = self.slide_gscene.active_section
    #
    #     position = self.status_comboBoxes.keys()[self.status_comboBoxes.values().index(sender)]
    #
    #     slide_position_tuple = (slide_name, position)
    #
    #     if slide_position_tuple in self.placeholders:
    #         self.placeholders.remove(slide_position_tuple)
    #     elif slide_position_tuple in self.rescans:
    #         self.rescans.remove(slide_position_tuple)
    #
    #     selected_status = sender.currentText()
    #     print selected_status
    #
    #     if selected_status == 'Placeholder':
    #         self.placeholders.add(slide_position_tuple)
    #         self.set_status(slide_name, position, 'Placeholder')
    #     elif selected_status == 'Rescan':
    #         self.rescans.add(slide_position_tuple)
    #         self.set_status(slide_name, position, 'Rescan')
    #     elif selected_status == 'Nonexisting':
    #         # self.slide_position_gscenes[position].pixmapItem.setVisible(False)
    #         self.set_status(slide_name, position, 'Nonexisting')
    #     elif selected_status == 'Normal':
    #         filenames = list(chain(*self.thumbnail_filenames[slide_name].values()))
    #         arbitrary_filename = os.path.splitext(filenames[0])[0]
    #         self.slide_position_to_fn[slide_name][position] = arbitrary_filename
    #         self.set_status(slide_name, position, arbitrary_filename)
    #
    #     print 'placeholders', self.placeholders
    #     print 'rescans', self.rescans


    # def gallery_image_updated(self):
    #     sender = self.sender()
    #     if sender == self.orderedSection1_gscene:
    #         self.orderedSection2_gscene.set_active_section(self.orderedSection1_gscene.active_section+1, emit_changed_signal=False)
    #         self.orderedSection3_gscene.set_active_section(self.orderedSection1_gscene.active_section+2, emit_changed_signal=False)
    #     elif sender == self.orderedSection2_gscene:
    #         self.orderedSection1_gscene.set_active_section(self.orderedSection2_gscene.active_section-1, emit_changed_signal=False)
    #         self.orderedSection3_gscene.set_active_section(self.orderedSection2_gscene.active_section+1, emit_changed_signal=False)
    #     elif sender == self.orderedSection3_gscene:
    #         self.orderedSection1_gscene.set_active_section(self.orderedSection3_gscene.active_section-2, emit_changed_signal=False)
    #         self.orderedSection2_gscene.set_active_section(self.orderedSection3_gscene.active_section-1, emit_changed_signal=False)


    def sorted_sections_image_updated(self):
        # print self.sorted_filenames[self.sorted_sections_gscene.active_section]
        filename = self.sorted_filenames[self.sorted_sections_gscene.active_section-1]
        self.label_sorted_sections_filename.setText(filename)
        if filename == 'Placeholder' or filename == 'Rescan':
            return
        assert filename != 'Unknown' and filename != 'Nonexisting'

        # Update slide scene
        slide_name = self.filename_to_slide[filename]
        position = self.slide_position_to_fn[slide_name].keys()[self.slide_position_to_fn[slide_name].values().index(filename)]
        print slide_name, position
        self.slide_gscene.set_active_section(slide_name)

    def set_status(self, slide_name, position, fn):

        # curr_fn = self.slide_position_to_fn[slide_name][position]

        # print curr_fn, fn
        # if curr_fn == fn:
        #     return

        self.slide_position_gscenes[position].set_active_section(fn)
        self.slide_position_to_fn[slide_name][position] = fn
        self.filename_labels[position].setText(fn)

        # if fn in ['Rescan', 'Placeholder', 'Nonexisting']:
        #     # if curr_fn not in ['Rescan', 'Placeholder', 'Nonexisting', 'Unknown'] and self.slide_gscene.active_section == slide_name:
        #     # pixmap = self.slide_position_gscenes[position].pixmapItem
        #     # if pixmap.isVisible() and self.slide_gscene.active_section == slide_name:
        #     #     pixmap.setVisible(False)
        #     self.slide_position_to_fn[slide_name][position] = fn
        #     self.slide_position_gscenes[position].set_active_section(fn)
        # else:
        #     self.slide_position_to_fn[slide_name][position] = fn
        #     self.slide_position_gscenes[position].set_active_section(fn)
        #     print fn

            # if curr_fn not in ['Rescan', 'Placeholder', 'Nonexisting']:
            #     pass
            #     # filenames = list(chain(*self.thumbnail_filenames[slide_name].values()))
            #     # self.slide_position_to_fn[slide_name][position] = os.path.splitext(filenames[0])[0]
            # else:
            #     self.slide_position_to_fn[slide_name][position] = fn
            #     self.slide_position_gscenes[position].set_active_section(fn)


        # combobox = self.status_comboBoxes[position]
        # if fn in ['Rescan', 'Placeholder', 'Nonexisting']:
        #     combobox.setCurrentIndex([combobox.itemText(i) for i in range(combobox.count())].index(fn))
        # else:
        #     combobox.setCurrentIndex(0)


    def slide_image_updated(self):
        self.setWindowTitle('Slide %(slide_index)s' % {'slide_index': self.slide_gscene.active_section})

        slide_name = self.slide_gscene.active_section
        feeder = self.section_image_feeders[slide_name]
        for position, gscene in self.slide_position_gscenes.iteritems():

            gscene.set_data_feeder(feeder)

            # if slide_name not in self.slide_position_to_fn:
            #     self.slide_position_to_fn[slide_name] = {}

            # if position not in self.slide_position_to_fn[slide_name]:
                # Here plug in initial guess of which image should be on this position
                # if position not in self.thumbnail_filenames[slide_name]:
                #     sys.stderr.write('Position %d has no corresponding image.\n' % position)
                #     self.set_status(slide_name, position, 'Nonexisting')

                    # filenames = list(chain(*self.thumbnail_filenames[slide_name].values()))
                    # self.slide_position_to_fn[slide_name][gscene_id] = os.path.splitext(filenames[0])[0]
                # else:

            if self.slide_position_to_fn[slide_name][position] != 'Unknown':
                self.set_status(slide_name, position, self.slide_position_to_fn[slide_name][position])
            else:
                if position in self.thumbnail_filenames[slide_name]:
                    self.set_status(slide_name, position, self.thumbnail_filenames[slide_name][position][0])
                else:
                    self.set_status(slide_name, position, 'Nonexisting')


            # fn = self.slide_position_to_fn[slide_name][position]
            # if fn == 'Rescan' or fn == 'Placeholder' or fn == 'Nonexisting':
            #     gscene.pixmapItem.setVisible(False)
            # else:
            #     gscene.set_active_section(fn)
            # except Exception as e:
                #     sys.stderr.write(e.message + '\n')
                #     gscene.pixmapItem.setVisible(False)

                    # if (self.slide_gscene.active_section, gscene_id) not in self.placeholders and \
                    #     (self.slide_gscene.active_section, gscene_id) not in self.rescans:
                    #     self.status_comboBoxes[gscene_id].setCurrentIndex(3)
                    #     self.filename_labels[gscene_id].setText('Nonexisting')

            # print self.slide_position_to_fn, '\n'

    def show_option_changed(self, index):
        # if text == 'original':
        #     self.section_gscene.set_data_feeder(image_feeders['original'])
        # elif text == 'aligned':
        #     self.section_gscene.set_data_feeder(image_feeders['aligned'])
        show_option_text = str(self.sender().currentText())
        if show_option_text == 'Original Aligned':
            self.currently_showing = 'aligned'
        elif show_option_text == 'Original':
            self.currently_showing = 'original'
        else:
            raise Exception('Not implemented.')

        print self.currently_showing

        self.section_gscene.set_data_feeder(self.image_feeders[self.currently_showing])
        self.section_gscene.update_image()
        # self.setWindowTitle('Section %(sec)d, %(show)s' % {'sec': self.section_gscene.active_section, 'show': self.currently_showing})

    def slide_position_image_updated(self):
        pass
        # slide_position_gscene_id = int(self.sender().id[-1])
        #
        # # ind = self.slide_position_gscenes[slide_position_gscene_id].active_section
        # # filenames = self.thumbnail_filenames[self.slide_gscene.active_section]
        #
        # # if ind in filenames:
        # filename = self.slide_position_gscenes[slide_position_gscene_id].active_section
        # self.filename_labels[slide_position_gscene_id].setText(filename)
        # # else:
        #     self.filename_labels[slide_position_gscene_id].setText('')

    # def active_image_updated(self):
    #     if self.section_gscene.active_section in self.bad_sections:
    #         self.bad_cross.setVisible(True)
    #     else:
    #         self.bad_cross.setVisible(False)
    #
    #     self.setWindowTitle('Section %(sec)d, %(show)s' % {'sec': self.section_gscene.active_section, 'show': self.currently_showing})



    def eventFilter(self, obj, event):

        if event.type() == QEvent.GraphicsSceneMousePress:
            pass
            # if obj in self.slide_position_gscenes.values():
            #     position = self.slide_position_gscenes.keys()[self.slide_position_gscenes.values().index(obj)]
            #     self.active_gscene = self.slide_position_gscenes[position]

        elif event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_1:
                # Record adjusted images
                for i in [1,2,3]:
                    try:
                        if self.slide_position_gscenes[i].active_section is not None:
                            self.slide_position_to_fn[self.slide_gscene.active_section][i] = self.slide_position_gscenes[i].active_section
                    except:
                        pass
                self.slide_gscene.show_previous(cycle=True)
            elif key == Qt.Key_2:
                for i in [1,2,3]:
                    try:
                        if self.slide_position_gscenes[i].active_section is not None:
                            self.slide_position_to_fn[self.slide_gscene.active_section][i] = self.slide_position_gscenes[i].active_section
                    except:
                        pass
                self.slide_gscene.show_next(cycle=True)

            if key == Qt.Key_3:
                self.active_gscene.show_next(cycle=True)
            elif key == Qt.Key_4:
                self.active_gscene.show_previous(cycle=True)
            elif key == Qt.Key_BracketLeft:
                self.sorted_sections_gscene.show_previous()
            elif key == Qt.Key_BracketRight:
                self.sorted_sections_gscene.show_next()

        return False


    # def load(self):
    #     # self.gscenes['sagittal'].load_drawings(username='Lauren', timestamp='latest', annotation_rootdir=annotation_midbrainIncluded_v2_rootdir)
    #     self.section_gscene.load_drawings(username='yuncong', timestamp='latest', annotation_rootdir=cerebellum_masks_rootdir)

    # @pyqtSlot()
    # def save(self):
    #
    #     # if not hasattr(self, 'username') or self.username is None:
    #     #     username, okay = QInputDialog.getText(self, "Username", "Please enter your username:", QLineEdit.Normal, 'anon')
    #     #     if not okay: return
    #     #     self.username = str(username)
    #     #     self.lineEdit_username.setText(self.username)
    #
    #     self.username = 'yuncong'
    #
    #     # labelings_dir = create_if_not_exists('/home/yuncong/CSHL_labelings_new/%(stack)s/' % dict(stack=self.stack))
    #     labelings_dir = create_if_not_exists(os.path.join(cerebellum_masks_rootdir, stack))
    #
    #     timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
    #
    #     self.section_gscene.save_drawings(fn_template=os.path.join(labelings_dir, '%(stack)s_%(orientation)s_downsample%(downsample)d_%(username)s_%(timestamp)s.pkl'), timestamp=timestamp, username=self.username)
    #
    #     self.statusBar().showMessage('Labelings saved to %s.' % labelings_dir)


if __name__ == "__main__":

    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Data Preprocessing GUI.')

    parser.add_argument("stack_name", type=str, help="stack name")
    args = parser.parse_args()

    from sys import argv, exit
    app = QApplication(argv)

    m = PreprocessGUI(stack=args.stack_name)

    # m.show()
    m.showMaximized()
    # m.raise_()
    exit(app.exec_())
