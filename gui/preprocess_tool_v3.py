#! /usr/bin/env python

import sys
import os
from subprocess import check_output
import cPickle as pickle
import argparse

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import pandas

from ui.ui_PreprocessGui_v2 import Ui_PreprocessGui
from ui.ui_AlignmentGui import Ui_AlignmentGui

from widgets.ZoomableBrowsableGraphicsScene import ZoomableBrowsableGraphicsScene, SimpleGraphicsScene2, SimpleGraphicsScene3, SimpleGraphicsScene4
from widgets.ZoomableBrowsableGraphicsSceneWithReadonlyPolygon import ZoomableBrowsableGraphicsSceneWithReadonlyPolygon
from widgets.MultiplePixmapsGraphicsScene import MultiplePixmapsGraphicsScene
from widgets.DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene
from widgets.SignalEmittingItems import *

from DataFeeder import ImageDataFeeder_v2

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *
from preprocess_utilities import *
from gui_utilities import *
from qt_utilities import *

# Use the third method in http://pyqt.sourceforge.net/Docs/PyQt4/designer.html
class PreprocessGUI(QMainWindow, Ui_PreprocessGui):

    def __init__(self, parent=None, stack=None, tb_fmt='png', tb_res='down32', tb_version=None):
        """
        Initialization of preprocessing tool.
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.stack = stack
        self.currently_showing = 'original'
        self.tb_fmt = tb_fmt
        self.tb_res = tb_res
        self.tb_version = tb_version
        self.stack_data_dir = os.path.join(THUMBNAIL_DATA_DIR, stack)

        self.show_valid_only = True

        _, self.section_to_filename = DataManager.load_sorted_filenames(self.stack)
        self.filename_to_accept_decision = {fn: True for sec, fn in self.section_to_filename.iteritems() if not is_invalid(fn)}

        # self.sorted_filenames = [f for s,f in sorted(section_to_filename.items())]

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

        self.first_section = None
        self.last_section = None

        self.sorted_sections_gscene = SimpleGraphicsScene3(id='sorted', gview=self.sorted_sections_gview)
        self.sorted_sections_gview.setScene(self.sorted_sections_gscene)

        self.sorted_sections_gscene.active_image_updated.connect(self.sorted_sections_image_updated)
        self.sorted_sections_gscene.first_section_set.connect(self.set_first_section)
        self.sorted_sections_gscene.last_section_set.connect(self.set_last_section)
        self.sorted_sections_gscene.anchor_set.connect(self.set_anchor)
        self.sorted_sections_gscene.move_down_requested.connect(self.move_down)
        self.sorted_sections_gscene.move_up_requested.connect(self.move_up)

        #######################

        self.installEventFilter(self)

        self.comboBox_show.activated.connect(self.show_option_changed)
        # self.button_sort.clicked.connect(self.sort)
        self.button_load_sorted_filenames.clicked.connect(self.load_sorted_filenames)
        # self.button_save_sorted_filenames.clicked.connect(self.save_sorted_filenames)
        # self.button_confirm_alignment.clicked.connect(self.compose)
        self.button_edit_transform.clicked.connect(self.edit_transform)
        # self.button_crop.clicked.connect(self.crop)
        self.button_save_crop.clicked.connect(self.save_crop)
        self.button_load_crop.clicked.connect(self.load_crop)
        self.button_update_order.clicked.connect(self.update_sorted_sections_gscene_from_sorted_filenames)
        self.button_toggle_show_hide_invalid.clicked.connect(self.toggle_show_hide_invalid)

        ################################

        self.placeholders = set([])
        self.rescans = set([])

        pp_fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_problematic_pairs.txt')
        if os.path.exists(pp_fp):
            sys.stderr.write("Loaded problematic pairs.\n")
            with open(pp_fp, 'r') as f:
                self.problematic_pairs = [tuple(line.split()) for line in f.readlines()]
        else:
            sys.stderr.write("Did not find problematic pairs.\n")
            self.problematic_pairs = []

    def toggle_show_hide_invalid(self):
        self.show_valid_only = not self.show_valid_only
        print self.show_valid_only
        self.button_toggle_show_hide_invalid.setText('Show both valid and invalid' if self.show_valid_only else 'Show valid only')
        self.update_sorted_sections_gscene_from_sorted_filenames()

    def move_down(self):

        curr_fn = self.sorted_sections_gscene.active_section
        next_fn = self.sorted_sections_gscene.data_feeder.sections[self.sorted_sections_gscene.active_i + 1]

        filename_to_section = invert_section_to_filename_mapping(self.section_to_filename)

        curr_fn_section = filename_to_section[curr_fn]
        next_fn_section = filename_to_section[next_fn]

        self.section_to_filename[curr_fn_section] = next_fn
        self.section_to_filename[next_fn_section] = curr_fn

        self.update_sorted_sections_gscene_from_sorted_filenames()
        self.sorted_sections_gscene.set_active_section(curr_fn)

    def move_up(self):

        curr_fn = self.sorted_sections_gscene.active_section
        prev_fn = self.sorted_sections_gscene.data_feeder.sections[self.sorted_sections_gscene.active_i - 1]

        filename_to_section = invert_section_to_filename_mapping(self.section_to_filename)

        curr_fn_section = filename_to_section[curr_fn]
        prev_fn_section = filename_to_section[prev_fn]

        self.section_to_filename[curr_fn_section] = prev_fn
        self.section_to_filename[prev_fn_section] = curr_fn

        self.update_sorted_sections_gscene_from_sorted_filenames()
        self.sorted_sections_gscene.set_active_section(curr_fn)

    def load_crop(self):
        """
        Load crop box.
        """
        self.set_show_option('aligned')
        cropbox_fp = DataManager.get_cropbox_filename(stack=self.stack, anchor_fn=self.anchor_fn)
        with open(cropbox_fp, 'r') as f:
            ul_x, lr_x, ul_y, lr_y, first_section, last_section = map(int, f.readline().split())
            self.first_section = self.section_to_filename[first_section]
            self.last_section = self.section_to_filename[last_section]
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

        filename_to_section = invert_section_to_filename_mapping(self.section_to_filename)
        with open(cropbox_fp, 'w') as f:
            f.write('%d %d %d %d %d %d' % (ul_x, lr_x, ul_y, lr_y, filename_to_section[self.first_section], filename_to_section[self.last_section]))

        upload_to_s3(cropbox_fp)

    def crop(self):
        pass
        ## Note that in cropbox, xmax, ymax are not included, so w = xmax-xmin, instead of xmax-xmin+1

        # self.save_crop()
        #
        # ul_pos = self.sorted_sections_gscene.corners['ul'].scenePos()
        # lr_pos = self.sorted_sections_gscene.corners['lr'].scenePos()
        # ul_x = int(ul_pos.x())
        # ul_y = int(ul_pos.y())
        # lr_x = int(lr_pos.x())
        # lr_y = int(lr_pos.y())
        #
        # if self.stack in all_nissl_stacks:
        #     pad_bg_color = 'white'
        # elif self.stack in all_ntb_stacks:
        #     pad_bg_color = 'black'
        # elif self.stack in all_alt_nissl_ntb_stacks or self.stack in all_alt_nissl_tracing_stacks:
        #     pad_bg_color = 'auto'
        #
        # self.web_service.convert_to_request('crop', stack=self.stack, x=ul_x, y=ul_y, w=lr_x+1-ul_x, h=lr_y+1-ul_y,
        #                                     f=self.first_section, l=self.last_section, anchor_fn=self.anchor_fn,
        #                                     filenames=self.get_valid_sorted_filenames(),
        #                                     first_fn=self.sorted_filenames[self.first_section-1],
        #                                     last_fn=self.sorted_filenames[self.last_section-1],
        #                                     pad_bg_color=pad_bg_color)

    ##################
    # Edit Alignment #
    ##################

    def edit_transform(self):

        sys.stderr.write('Loading Edit Transform GUI...\n')
        self.statusBar().showMessage('Loading Edit Transform GUI...')

        self.alignment_ui = Ui_AlignmentGui()
        self.alignment_gui = QDialog(self)
        self.alignment_gui.setWindowTitle("Edit transform between adjacent sections")
        self.alignment_ui.setupUi(self.alignment_gui)

        self.alignment_ui.button_anchor.clicked.connect(self.add_anchor_pair_clicked)
        self.alignment_ui.button_align.clicked.connect(self.align_using_elastix)
        self.alignment_ui.button_compute.clicked.connect(self.compute_custom_transform)

        param_fps = os.listdir(DataManager.get_elastix_parameters_dir())
        all_parameter_setting_names = ['_'.join(pf[:-4].split('_')[1:]) for pf in param_fps]
        self.alignment_ui.comboBox_parameters.addItems(all_parameter_setting_names)

        section_filenames = self.get_sorted_filenames(valid_only=self.show_valid_only)

        self.curr_gscene = SimpleGraphicsScene4(id='current', gview=self.alignment_ui.curr_gview)
        self.alignment_ui.curr_gview.setScene(self.curr_gscene)
        self.curr_gscene.set_data_feeder(self.ordered_images_feeder)
        self.curr_gscene.set_active_i(1)
        self.curr_gscene.active_image_updated.connect(self.current_section_image_changed)
        self.curr_gscene.anchor_point_added.connect(self.anchor_point_added)

        self.prev_gscene = SimpleGraphicsScene4(id='previous', gview=self.alignment_ui.prev_gview)
        self.alignment_ui.prev_gview.setScene(self.prev_gscene)
        self.prev_gscene.set_data_feeder(self.ordered_images_feeder)
        self.prev_gscene.set_active_i(0)
        self.prev_gscene.active_image_updated.connect(self.previous_section_image_changed)
        self.prev_gscene.anchor_point_added.connect(self.anchor_point_added)

        self.overlay_gscene = MultiplePixmapsGraphicsScene(id='overlay', pixmap_labels=['moving', 'fixed'], gview=self.alignment_ui.aligned_gview)
        self.alignment_ui.aligned_gview.setScene(self.overlay_gscene)
        self.transformed_images_feeder = ImageDataFeeder_v2('overlay image feeder', stack=self.stack,
                                                sections=section_filenames, resolution=self.tb_res)
        self.update_transformed_images_feeder()
        self.overlay_gscene.set_data_feeder(self.transformed_images_feeder, 'moving')
        self.overlay_gscene.set_data_feeder(self.ordered_images_feeder, 'fixed')
        self.overlay_gscene.set_active_indices({'moving': 1, 'fixed': 0})
        self.overlay_gscene.set_opacity('moving', .3)
        self.overlay_gscene.set_opacity('fixed', .3)

        self.alignment_gui.show()

    def update_transformed_images_feeder(self):

        section_filenames = self.get_sorted_filenames(valid_only=self.show_valid_only)

        transformed_image_filenames = []
        for i in xrange(1, len(section_filenames)):
            fp = DataManager.load_image_filepath_warped_to_adjacent_section(stack=self.stack, moving_fn=section_filenames[i], fixed_fn=section_filenames[i-1])
            transformed_image_filenames.append(fp)

        self.transformed_images_feeder.set_images(labels=section_filenames[1:], filenames=transformed_image_filenames, resolution=self.tb_res, load_with_cv2=True)

    def align_using_elastix(self):
        selected_elastix_parameter_name = str(self.alignment_ui.comboBox_parameters.currentText())
        param_fn = os.path.join(REPO_DIR, 'preprocess', 'parameters', 'Parameters_' + selected_elastix_parameter_name + '.txt')

        curr_fn = self.curr_gscene.active_section
        prev_fn = self.prev_gscene.active_section
        out_dir = os.path.join(self.stack_data_dir, self.stack + '_custom_transforms', curr_fn + '_to_' + prev_fn)

        curr_fp = DataManager.get_image_filepath_v2(stack=self.stack, prep_id=None, fn=curr_fn, resol=self.tb_res, version=self.tb_version)
        prev_fp = DataManager.get_image_filepath_v2(stack=self.stack, prep_id=None, fn=prev_fn, resol=self.tb_res, version=self.tb_version )

        # curr_fp = os.path.join(RAW_DATA_DIR, self.stack, curr_fn + '.' + self.tb_fmt)
        # prev_fp = os.path.join(RAW_DATA_DIR, self.stack, prev_fn + '.' + self.tb_fmt)

        execute_command('rm -rf %(out_dir)s; mkdir -p %(out_dir)s; elastix -f %(fixed_fn)s -m %(moving_fn)s -out %(out_dir)s -p %(param_fn)s' % \
        dict(param_fn=param_fn, out_dir=out_dir, fixed_fn=prev_fp, moving_fn=curr_fp))
        # section_filenames = self.get_sorted_filenames(valid_only=self.show_valid_only)

        self.update_transformed_images_feeder()

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
        curr_section_fn = self.curr_gscene.active_section
        prev_section_fn = self.prev_gscene.active_section

        custom_tf_dir = os.path.join(self.stack_data_dir, self.stack + '_custom_transforms', curr_section_fn + '_to_' + prev_section_fn)

        execute_command("rm -rf %(out_dir)s; mkdir -p %(out_dir)s" % dict(out_dir=custom_tf_dir))
        custom_tf_fp = os.path.join(custom_tf_dir, '%(curr_fn)s_to_%(prev_fn)s_customTransform.txt' % \
                    dict(curr_fn=curr_section_fn, prev_fn=prev_section_fn))

        with open(custom_tf_fp, 'w') as f:
            f.write('%f %f %f %f %f %f\n' % (R[0,0], R[0,1], t[0], R[1,0], R[1,1], t[1]))

        self.apply_custom_transform()
        self.update_transformed_images_feeder()

    def apply_custom_transform(self):

        # section_filenames = self.get_sorted_filenames(valid_only=self.show_valid_only)

        # curr_section_fn = section_filenames[self.valid_section_indices[self.curr_gscene.active_i]-1]
        # prev_section_fn = section_filenames[self.valid_section_indices[self.prev_gscene.active_i]-1]
        curr_section_fn = self.curr_gscene.active_section
        prev_section_fn = self.prev_gscene.active_section

        custom_tf_fn = os.path.join(self.stack_data_dir, self.stack+'_custom_transforms', curr_section_fn + '_to_' + prev_section_fn, curr_section_fn + '_to_' + prev_section_fn + '_customTransform.txt')
        with open(custom_tf_fn, 'r') as f:
            t11, t12, t13, t21, t22, t23 = map(float, f.readline().split())

        prev_fp = DataManager.get_image_filepath_v2(stack=self.stack, prep_id=None, fn=prev_section_fn, resol=self.tb_res, version=self.tb_version )
        curr_fp = DataManager.get_image_filepath_v2(stack=self.stack, prep_id=None, fn=curr_section_fn, resol=self.tb_res, version=self.tb_version )
        prev_img_w, prev_img_h = identify_shape(prev_fp)

        output_image_fp = os.path.join(self.stack_data_dir, '%(stack)s_custom_transforms/%(curr_fn)s_to_%(prev_fn)s/%(curr_fn)s_alignedTo_%(prev_fn)s.tif' % \
                        dict(stack=self.stack,
                        curr_fn=curr_section_fn,
                        prev_fn=prev_section_fn) )

        execute_command("convert %(curr_fp)s -virtual-pixel background +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop %(w)sx%(h)s%(x)s%(y)s\! -flatten -compress lzw %(output_fp)s" %\
        dict(curr_fp=curr_fp,
            output_fp=output_image_fp,
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
        section_filenames = self.get_sorted_filenames(valid_only=self.show_valid_only)

        curr_section_fn = self.curr_gscene.active_section
        prev_section_fn = section_filenames[section_filenames.index(curr_section_fn) - 1]

        print self.problematic_pairs

        if (prev_section_fn, curr_section_fn) in self.problematic_pairs:
            self.alignment_ui.label_current_filename.setText('(CHECK)' + str(curr_section_fn))
            self.alignment_ui.label_current_index.setText('(CHECK)' + str(curr_section_fn))
        else:
            self.alignment_ui.label_current_filename.setText(str(curr_section_fn))
            self.alignment_ui.label_current_index.setText(str(curr_section_fn))
        self.prev_gscene.set_active_section(prev_section_fn)
        self.overlay_gscene.set_active_sections({'moving': curr_section_fn, 'fixed': prev_section_fn})

    def previous_section_image_changed(self):
        section_filenames = self.get_sorted_filenames(valid_only=self.show_valid_only)

        prev_section_fn = self.prev_gscene.active_section
        curr_section_fn = section_filenames[section_filenames.index(prev_section_fn) + 1]

        self.alignment_ui.label_previous_filename.setText(prev_section_fn)
        self.alignment_ui.label_previous_index.setText(str(prev_section_fn))
        self.curr_gscene.set_active_section(curr_section_fn)
        self.overlay_gscene.set_active_sections({'moving': curr_section_fn, 'fixed': prev_section_fn})

    ########################## END OF EDIT TRANSFORM ######################################3


    # def sort(self):
    #     """
    #     Sort images.
    #     """
    #     self.update_sorted_sections_gscene_from_sorted_filenames()

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

        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        pickle.dump(info, open(self.stack_data_dir + '/%(stack)s_preprocessInfo_%(timestamp)s.pkl' % {'stack': self.stack, 'timestamp':timestamp}, 'w'))

        execute_command('cd %(stack_data_dir)s && rm -f %(stack)s_preprocessInfo.pkl && ln -s %(stack)s_preprocessInfo_%(timestamp)s.pkl %(stack)s_preprocessInfo.pkl' % {'stack': self.stack, 'timestamp':timestamp, 'stack_data_dir':self.stack_data_dir})

        self.save_crop()
        self.save_sorted_filenames()
        self.save()

    def update_sorted_sections_gscene_from_sorted_filenames(self):

        if not hasattr(self, 'currently_showing'):
            self.currently_showing = 'original'

        # self.sorted_filenames =
        # self.valid_section_filenames = self.get_valid_sorted_filenames()
        # self.valid_section_indices = [self.sorted_filenames.index(fn) + 1 for fn in self.valid_section_filenames]

        if not hasattr(self, 'anchor_fn'):
            anchor_fp = DataManager.get_anchor_filename_filename(self.stack)
            if os.path.exists(anchor_fp):
                with open(anchor_fp) as f:
                    self.set_anchor(f.readline().strip())
            else:
                filenames_to_load = self.get_sorted_filenames(valid_only=self.show_valid_only)
                shapes = \
                    [identify_shape(DataManager.get_image_filepath_v2(stack=self.stack, fn=fn, prep_id=None, version=self.tb_version, resol=self.tb_res))
                    for fn in filenames_to_load]
                largest_idx = np.argmax([h*w for h, w in shapes])
                print 'largest section is ', filenames_to_load[largest_idx]
                self.set_anchor(filenames_to_load[largest_idx])
                print filenames_to_load[largest_idx]

        if self.currently_showing == 'original':

            filenames_to_load = self.get_sorted_filenames(valid_only=self.show_valid_only)
            print filenames_to_load

            if not hasattr(self, 'ordered_images_feeder') or self.ordered_images_feeder is None:
                self.ordered_images_feeder = ImageDataFeeder_v2('ordered image feeder', stack=self.stack,
                                    sections=filenames_to_load, resolution=self.tb_res, use_thread=False, auto_load=False)
                self.ordered_images_feeder.set_images(labels=filenames_to_load,
                                                filenames=[DataManager.get_image_filepath_v2(stack=self.stack, fn=fn, prep_id=None, version=self.tb_version, resol=self.tb_res)
                                                                            for fn in filenames_to_load],
                                                resolution=self.tb_res, load_with_cv2=False)
                self.ordered_images_feeder.set_images(labels=['Placeholder'],
                                                filenames=[self.placeholder_qimage],
                                                resolution=self.tb_res, load_with_cv2=False)
            else:
                self.ordered_images_feeder.set_sections(filenames_to_load)

            self.sorted_sections_gscene.set_data_feeder(self.ordered_images_feeder)

            if self.sorted_sections_gscene.active_i is not None:
                active_i = self.sorted_sections_gscene.active_i
            else:
                active_i = 1
            self.sorted_sections_gscene.set_active_i(active_i)

        elif self.currently_showing == 'aligned':

            filenames_to_load = self.get_sorted_filenames(valid_only=self.show_valid_only)
            print filenames_to_load

            if not hasattr(self, 'aligned_images_feeder') or self.aligned_images_feeder is None:
                self.aligned_images_feeder = ImageDataFeeder_v2('aligned image feeder', stack=self.stack,
                                    sections=filenames_to_load, resolution=self.tb_res, use_thread=False, auto_load=False)
                self.aligned_images_feeder.set_images(labels=filenames_to_load,
                                                filenames=[DataManager.get_image_filepath_v2(stack=self.stack, fn=fn, prep_id=1, version=self.tb_version, resol=self.tb_res)
                                                                            for fn in filenames_to_load],
                                                resolution=self.tb_res, load_with_cv2=False)

                self.aligned_images_feeder.set_images(labels=['Placeholder'],
                                                filenames=[self.placeholder_qimage],
                                                resolution=self.tb_res, load_with_cv2=False)
            else:
                self.aligned_images_feeder.set_sections(filenames_to_load)

            self.sorted_sections_gscene.set_data_feeder(self.aligned_images_feeder)

            if self.sorted_sections_gscene.active_i is not None:
                active_i = self.sorted_sections_gscene.active_i
            else:
                active_i = 1
            self.sorted_sections_gscene.set_active_i(active_i)

        # elif self.currently_showing == 'mask_contour':
        #
        #     self.maskContourViz_images_feeder = ImageDataFeeder_v2('mask contoured image feeder', stack=self.stack,
        #                                         sections=self.valid_section_indices, resolution=self.tb_res)
        #     self.maskContourViz_images_dir = self.stack_data_dir + '/%(stack)s_maskContourViz_unsorted' % {'stack': self.stack}
        #     maskContourViz_image_filenames = [os.path.join(self.maskContourViz_images_dir, '%(fn)s_mask_contour_viz.tif' % {'fn': fn})
        #                                 for fn in self.valid_section_filenames]
        #
        #     self.maskContourViz_images_feeder.set_images(self.valid_section_indices, maskContourViz_image_filenames, resolution=self.tb_res, load_with_cv2=False)
        #     # self.maskContourViz_images_feeder.set_resolution(self.tb_res)
        #
        #     active_i = self.sorted_sections_gscene.active_i
        #     self.sorted_sections_gscene.set_data_feeder(self.maskContourViz_images_feeder)
        #     self.sorted_sections_gscene.set_active_i(active_i)


    def save_sorted_filenames(self):

        sorted_filenames = self.get_sorted_filenames(valid_only=False)

        out_sorted_image_names_fp = DataManager.get_sorted_filenames_filename(stack=self.stack)
        with open(out_sorted_image_names_fp, 'w') as f:
            for i, fn in enumerate(sorted_filenames):
                f.write('%s %03d\n' % (fn, i+1)) # index starts from 1

        upload_to_s3(out_sorted_image_names_fp)

        sys.stderr.write('Sorted filename list saved.\n')
        self.statusBar().showMessage('Sorted filename list saved.')


    def load_sorted_filenames(self):

        # filename_to_section, section_to_filename = DataManager.load_sorted_filenames(self.stack)
        # self.sorted_filenames = [f for s,f in sorted(section_to_filename.items())]

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

    def get_sorted_filenames(self, valid_only=False):
        return [fn for sec, fn in sorted(self.section_to_filename.items())
        if (fn != 'Rescan' and fn != 'Placeholder' and self.filename_to_accept_decision[fn]) or not valid_only]

    def compose(self):
        pass

    def set_first_section(self, fn):
        self.first_section = str(fn)
        self.update_sorted_sections_gscene_label()

    def set_last_section(self, fn):
        self.last_section = str(fn)
        self.update_sorted_sections_gscene_label()

    def set_anchor(self, anchor):
        if isinstance(anchor, int):
            self.anchor_fn = self.sorted_filenames[anchor-1]
        elif isinstance(anchor, str):
            self.anchor_fn = anchor

        with open(os.path.join(THUMBNAIL_DATA_DIR, self.stack, self.stack + '_anchor.txt'), 'w') as f:
            f.write(self.anchor_fn)

        self.update_sorted_sections_gscene_label()

    def sorted_sections_image_updated(self):
        filename = self.sorted_sections_gscene.active_section
        self.label_sorted_sections_filename.setText(filename)
        self.label_sorted_sections_index.setText(str(self.sorted_sections_gscene.active_section))
        if filename == 'Placeholder' or filename == 'Rescan':
            return
        assert filename != 'Unknown' and filename != 'Nonexisting'

        # Update slide scene

        # slide_name = self.filename_to_slide[filename]
        # position = self.slide_position_to_fn[slide_name].keys()[self.slide_position_to_fn[slide_name].values().index(filename)]
        # self.slide_gscene.set_active_section(slide_name)
        self.update_sorted_sections_gscene_label()

    def update_sorted_sections_gscene_label(self):
        """
        Set the label next to sortedSectionGscene to FIRST, LAST or ANCHOR.
        """

        # print self.sorted_sections_gscene.active_section, self.anchor_fn
        # if self.sorted_sections_gscene.active_section is not None:
        #     print self.sorted_filenames[self.sorted_sections_gscene.active_section-1]

        if self.sorted_sections_gscene.active_section == self.first_section:
            self.label_sorted_sections_status.setText('FIRST')
        elif self.sorted_sections_gscene.active_section == self.last_section:
            self.label_sorted_sections_status.setText('LAST')
        elif hasattr(self, 'anchor_fn') and self.sorted_sections_gscene.active_section is not None and \
            self.sorted_sections_gscene.active_section == self.anchor_fn:
            print self.sorted_sections_gscene.active_section
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


    # def slide_image_updated(self):
    #     self.setWindowTitle('Slide %(slide_index)s' % {'slide_index': self.slide_gscene.active_section})
    #
    #     slide_name = self.slide_gscene.active_section
    #     feeder = self.section_image_feeders[slide_name]
    #
    #     if slide_name not in self.slide_position_to_fn:
    #         self.slide_position_to_fn[slide_name] = {p: 'Unknown' for p in [1,2,3]}
    #
    #     for position, gscene in self.slide_position_gscenes.iteritems():
    #
    #         gscene.set_data_feeder(feeder)
    #
    #         if self.slide_position_to_fn[slide_name][position] != 'Unknown':
    #             self.set_status(slide_name, position, self.slide_position_to_fn[slide_name][position])
    #         else:
    #             if position in self.thumbnail_filenames[slide_name]:
    #                 newest_fn = sorted(self.thumbnail_filenames[slide_name][position].items())[-1][1]
    #                 self.set_status(slide_name, position, newest_fn)
    #             else:
    #                 self.set_status(slide_name, position, 'Nonexisting')

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
        self.set_status(self.slide_gscene.active_section, position, self.slide_position_gscenes[position].active_section)

    def eventFilter(self, obj, event):

        if event.type() == QEvent.GraphicsSceneMousePress:
            pass
        elif event.type() == QEvent.KeyPress:
            key = event.key()
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Data Preprocessing GUI.')

    parser.add_argument("stack_name", type=str, help="stack name")
    parser.add_argument("--tb_fmt", type=str, help="thumbnail format", default='png')
    parser.add_argument("--tb_res", type=str, help="resolution of displayed thumbnail images", default='down32')
    parser.add_argument("--tb_version", type=str, help="version of displayed thumbnail images", default=None)
    args = parser.parse_args()
    app = QApplication(sys.argv)

    m = PreprocessGUI(stack=args.stack_name, tb_fmt=args.tb_fmt, tb_res=args.tb_res, tb_version=args.tb_version)

    m.showMaximized()
    sys.exit(app.exec_())
