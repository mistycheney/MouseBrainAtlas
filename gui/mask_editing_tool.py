#! /usr/bin/env python

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui_MaskEditingGui3 import Ui_MaskEditingGui3

import sys, os
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *
from registration_utilities import find_contour_points

import argparse
from sys import argv, exit

from DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene
from ZoomableBrowsableGraphicsScene import ZoomableBrowsableGraphicsScene
from DataFeeder import ImageDataFeeder

from preprocess_utilities import *
from gui_utilities import *
from qt_utilities import *
from mask_editing_utilities import *

from skimage.segmentation import slic, mark_boundaries

import matplotlib.pyplot as plt

class MaskEditingGUI(QMainWindow):
    def __init__(self, parent=None, stack=None):
        QMainWindow.__init__(self, parent)

        self.stack = stack

        self.ui = Ui_MaskEditingGui3()
        self.dialog = QDialog(self)
        self.ui.setupUi(self.dialog)

        self.ui.button_slic.clicked.connect(self.update_slic)
        self.ui.button_submasks.clicked.connect(self.update_submask_image)
        self.ui.button_snake.clicked.connect(self.update_final_mask_image)

        self.ui.slider_threshold.setSingleStep(1)
        self.ui.slider_threshold.setMinimum(1)
        self.ui.slider_threshold.setMaximum(255)
        self.ui.slider_threshold.setValue(200)
        # self.ui.slider_threshold.setEnabled(True)
        self.ui.slider_threshold.valueChanged.connect(self.threshold_changed)

        self.ui.slider_dissimThresh.setSingleStep(1) # unit is 0.01
        self.ui.slider_dissimThresh.setMinimum(0)
        self.ui.slider_dissimThresh.setMaximum(200) # 2
        self.ui.slider_dissimThresh.setValue(50) # .5
        self.ui.slider_dissimThresh.valueChanged.connect(self.dissim_threshold_changed)
        self.ui.button_confirmDissimThresh.clicked.connect(self.dissim_threshold_change_confirmed)

        self.sections_to_filenames = DataManager.load_sorted_filenames(stack)[1]
        self.valid_sections_to_filenames = {sec: fn for sec, fn in self.sections_to_filenames.iteritems() if not is_invalid(fn)}
        self.valid_filenames_to_sections = {fn: sec for sec, fn in self.valid_sections_to_filenames.iteritems() if not is_invalid(fn)}
        # self.valid_filenames = self.valid_filenames_to_sections.keys()
        # self.valid_sections = self.valid_sections_to_filenames.keys()

        q = sorted(self.valid_sections_to_filenames.items())
        self.valid_sections = [sec for sec, fn in q]
        self.valid_filenames = [fn for sec, fn in q]

        # Generate submask review results.
        sys.stderr.write('Load submask review results...\n')

        alg_review_fp = THUMBNAIL_DATA_DIR + "/%(stack)s/%(stack)s_submask_algorithm_review_results.csv" % dict(stack=self.stack)
        user_review_fp = THUMBNAIL_DATA_DIR + "/%(stack)s/%(stack)s_submask_user_review_results.csv" % dict(stack=self.stack)
        if os.path.exists(user_review_fp):
            review_df = pandas.read_csv(user_review_fp, header=0, index_col=0)
            mask_review_results = {fn: {int(submask_i): bool(dec) for submask_i, dec in decisions.dropna().iteritems()} for fn, decisions in review_df.iterrows()}
            self.mask_review_results = {fn: mask_review_results[fn] if fn in mask_review_results else {} for fn in self.valid_filenames}
        elif os.path.exists(alg_review_fp):
            review_df = pandas.read_csv(alg_review_fp, header=0, index_col=0)
            mask_review_results = {fn: {int(submask_i): bool(dec) for submask_i, dec in decisions.dropna().iteritems()} for fn, decisions in review_df.iterrows()}
            self.mask_review_results = {fn: mask_review_results[fn] if fn in mask_review_results else {} for fn in self.valid_filenames}
        else:
            self.mask_review_results = generate_submask_review_results(stack=self.stack, filenames=self.valid_filenames)

        ############################################################

        self.fns_submask_modified = []

        sys.stderr.write('Initialize gscene...\n')
        self.gscene_finalMask = DrawableZoomableBrowsableGraphicsScene(id='finalMask', gview=self.ui.gview_finalMask)
        self.gscene_finalMask.set_default_line_color('g')
        self.gscene_finalMask.set_default_line_width(1)
        self.gscene_finalMask.set_default_vertex_color('b')
        self.gscene_finalMask.set_default_vertex_radius(2)

        # Initialize Mask gview.
        self.final_mask_feeder = ImageDataFeeder(name='finalMask', stack=self.stack, \
                                sections=self.valid_sections, use_data_manager=False,
                                downscale=32)
        # self.final_mask_feeder.set_downsample_factor(32)

        labeled_filenames = {sec: os.path.join(RAW_DATA_DIR, self.stack, fn + ".png")
                            for sec, fn in self.valid_sections_to_filenames.iteritems()}

        self.final_mask_feeder.set_images(labeled_filenames=labeled_filenames)
        self.gscene_finalMask.set_data_feeder(self.final_mask_feeder)

        # Load all submask images.
        self.all_submasks = {}
        for sec, fn in self.valid_sections_to_filenames.iteritems():
            self.all_submasks[fn] = {}
            for mask_ind, decision in self.mask_review_results[fn].iteritems():
                mask_fn = os.path.join(THUMBNAIL_DATA_DIR, "%(stack)s/%(stack)s_submasks/%(img_fn)s/%(img_fn)s_submask_%(mask_ind)d.png") % \
                        dict(stack=self.stack, img_fn=fn, mask_ind=mask_ind)
                if os.path.exists(mask_fn):
                    mask = imread(mask_fn)

                    self.all_submasks[fn][mask_ind] = mask.astype(np.bool)

                    cnts = find_contour_points(mask, sample_every=1)[255]
                    if len(cnts) == 0:
                        raise Exception('ERROR: %s, %d, %d - no contour' % (fn, mask_ind, len(cnts)))
                    elif len(cnts) > 1:
                        sys.stderr.write('WARNING: %s, %d, %d - multiple contours\n' % (fn, mask_ind, len(cnts)))
                        cnt = sorted(cnts, key=lambda c: len(c), reverse=True)[0]
                    else:
                        cnt = cnts[0]

                    if decision:
                        color = 'g'
                    else:
                        color = 'r'
                    self.gscene_finalMask.add_polygon(path=vertices_to_path(cnt), section=sec, linewidth=2, color=color)
                else:
                    sys.stderr.write('WARNING: Review has %s, submask %d, but image file does not exist.\n' % (fn, mask_ind))

        # self.ui.gview_finalMask.setScene(self.gscene_finalMask)

        try:
            self.gscene_finalMask.set_active_section(160, emit_changed_signal=False)
        except:
            pass

        self.gscene_finalMask.active_image_updated.connect(self.final_mask_section_changed)
        self.gscene_finalMask.drawings_updated.connect(self.submask_added)
        self.gscene_finalMask.polygon_pressed.connect(self.submask_clicked)
        self.gscene_finalMask.polygon_deleted.connect(self.submask_deleted)

        #########################################################

        self.ui.comboBox_channel.activated.connect(self.channel_changed)
        self.ui.comboBox_channel.addItems(['Red', 'Green', 'Blue'])

        ########################################################

        self.original_images = {}
        self.selected_channels = {}
        self.selected_thresholds = {sec: 200 for sec in self.valid_sections}
        self.thresholded_images = {}
        self.contrast_stretched_images = {}
        self.slic_labelmaps = {}
        self.slic_boundary_images = {}
        self.ncut_labelmaps = {}
        self.border_dissim_images = {}
        self.selected_dissim_thresholds = {}
        self.submask_images = {}
        self.sp_dissim_maps = {}
        self.submasks = {}
        self.final_masks = {}
        self.final_mask_images = {}

        #########################################################

        self.gscene_thresholded = ZoomableBrowsableGraphicsScene(id='thresholded', gview=self.ui.gview_thresholded)
        # self.ui.gview_thresholded.setScene(self.gscene_thresholded)

        self.thresholded_image_feeder = ImageDataFeeder(name='thresholded', stack=self.stack, \
                                                        sections=self.valid_sections, use_data_manager=False,
                                                        downscale=32)
        # self.thresholded_image_feeder.set_downsample_factor(32)
        self.gscene_thresholded.set_data_feeder(self.thresholded_image_feeder)

        #########################################################

        self.gscene_slic = ZoomableBrowsableGraphicsScene(id='slic', gview=self.ui.gview_slic)
        # self.ui.gview_slic.setScene(self.gscene_slic)

        self.slic_image_feeder = ImageDataFeeder(name='slic', stack=self.stack, \
                                                sections=self.valid_sections, use_data_manager=False,
                                                downscale=32)
        # self.slic_image_feeder.set_downsample_factor(32)
        self.gscene_slic.set_data_feeder(self.slic_image_feeder)

        #########################################################

        self.gscene_dissimmap = ZoomableBrowsableGraphicsScene(id='dissimmap', gview=self.ui.gview_dissimmap)
        # self.ui.gview_dissimmap.setScene(self.gscene_dissimmap)

        self.dissim_image_feeder = ImageDataFeeder(name='dissimmap', stack=self.stack, \
                                                sections=self.valid_sections, use_data_manager=False,
                                                downscale=32)
        # self.dissim_image_feeder.set_downsample_factor(32)
        self.gscene_dissimmap.set_data_feeder(self.dissim_image_feeder)

        #########################################################

        self.gscene_submasks = ZoomableBrowsableGraphicsScene(id='submasks', gview=self.ui.gview_submasks)
        self.submask_image_feeder = ImageDataFeeder(name='submasks', stack=self.stack, \
                                                sections=self.valid_sections, use_data_manager=False,
                                                downscale=32)
        self.gscene_submasks.set_data_feeder(self.submask_image_feeder)

        #########################################################

        self.gscene_snake = ZoomableBrowsableGraphicsScene(id='snake', gview=self.ui.gview_snake)
        self.snake_image_feeder = ImageDataFeeder(name='snake', stack=self.stack, \
                                                sections=self.valid_sections, use_data_manager=False,
                                                downscale=32)
        self.gscene_snake.set_data_feeder(self.snake_image_feeder)

        #########################################################

        self.dialog.showMaximized()

    def update_slic(self):
        sec = self.gscene_finalMask.active_section

        t = time.time()
        self.slic_labelmaps[sec] = slic(self.contrast_stretched_images[sec].astype(np.float),
                                    sigma=SLIC_SIGMA, compactness=SLIC_COMPACTNESS,
                                    n_segments=SLIC_N_SEGMENTS, multichannel=False, max_iter=SLIC_MAXITER)
        sys.stderr.write('SLIC: %.2f seconds.\n' % (time.time() - t)) # 10 seconds, iter=100, nseg=1000;

        self.slic_boundary_images[sec] = img_as_ubyte(mark_boundaries(self.contrast_stretched_images[sec],
                                            label_img=self.slic_labelmaps[sec],
                                            background_label=-1, color=(1,0,0)))

        self.slic_image_feeder.set_image(sec=sec, numpy_image=self.slic_boundary_images[sec])
        self.gscene_slic.update_image(sec=sec)

        ####

        self.ncut_labelmaps[sec] = normalized_cut_superpixels(self.contrast_stretched_images[sec], self.slic_labelmaps[sec])
        self.sp_dissim_maps[sec] = compute_sp_dissims_to_border(self.contrast_stretched_images[sec], self.ncut_labelmaps[sec])
        self.border_dissim_images[sec] = generate_dissim_viz(self.sp_dissim_maps[sec], self.ncut_labelmaps[sec])
        self.dissim_image_feeder.set_image(sec=sec, numpy_image=self.border_dissim_images[sec])
        self.gscene_dissimmap.update_image(sec=sec)

        self.selected_dissim_thresholds[sec] = determine_dissim_threshold(self.sp_dissim_maps[sec], self.ncut_labelmaps[sec])


    def update_submask_image(self):

        sec = self.gscene_finalMask.active_section

        self.submasks[sec] = get_submasks(ncut_labels=self.ncut_labelmaps[sec], sp_dissims=self.sp_dissim_maps[sec], dissim_thresh=self.selected_dissim_thresholds[sec])
        self.submask_images[sec] = generate_submasks_viz(self.contrast_stretched_images[sec], self.submasks[sec], color=(255,0,0))
        self.submask_image_feeder.set_image(sec=sec, numpy_image=self.submask_images[sec])
        self.gscene_submasks.update_image(sec=sec)

    def update_final_mask_image(self):

        sec = self.gscene_finalMask.active_section

        # self.final_masks[sec] = snake(img=self.contrast_stretched_images[sec], submasks=self.submasks[sec])
        self.final_masks[sec] = snake(img=self.thresholded_images[sec], submasks=self.submasks[sec])
        self.final_mask_images[sec] = generate_submasks_viz(self.original_images[sec], self.final_masks[sec], color=(0,0,255))

        self.snake_image_feeder.set_image(sec=sec, numpy_image=self.final_mask_images[sec])
        self.gscene_snake.update_image(sec=sec)

    def change_channel(self, channel):
        print 'Changed to', channel
        sec = self.gscene_finalMask.active_section
        self.contrast_stretched_images[sec] = contrast_stretch_image(self.original_images[sec], channel=channel)
        self.update_thresholded_image()

    def channel_changed(self, index):
        self.selected_channels[self.gscene_finalMask.active_section] = index

        channel_text = str(self.sender().currentText())
        if channel_text == 'Red':
            self.change_channel(0)
        elif channel_text == 'Green':
            self.change_channel(1)
        elif channel_text == 'Blue':
            self.change_channel(2)

    def dissim_threshold_change_confirmed(self):
        self.selected_dissim_thresholds[self.gscene_finalMask.active_section] = self.ui.slider_dissimThresh.value() * 0.01
        self.update_submask_image()

    def dissim_threshold_changed(self, value):
        self.ui.label_dissimThresh.setText(str(value * 0.01))

    def threshold_changed(self, value):
        self.ui.label_threshold.setText(str(value))
        self.selected_thresholds[self.gscene_finalMask.active_section] = value
        self.update_thresholded_image()

    def update_thresholded_image(self):
        print "update_thresholded_image"

        sec = self.gscene_finalMask.active_section
        thresholded_image = (self.contrast_stretched_images[sec] > self.selected_thresholds[sec]).astype(np.uint8)*255
        self.thresholded_images[sec] = thresholded_image

        self.thresholded_image_feeder.set_image(sec=sec, qimage=numpy_to_qimage(thresholded_image))

        self.gscene_thresholded.update_image(sec=sec)

    def final_mask_section_changed(self):

        self.update_mask_gui_window_title()

        sec = self.gscene_finalMask.active_section

        if sec not in self.contrast_stretched_images:
            if sec not in self.original_images:
                self.original_images[sec] = imread(DataManager.get_image_filepath(stack=self.stack, section=sec, resol='thumbnail', version='original_png'))

        if sec not in self.selected_channels:
            self.selected_channels[sec] = 0

        self.ui.comboBox_channel.setCurrentIndex(self.selected_channels[sec])
        self.change_channel(self.selected_channels[sec])

        self.ui.slider_threshold.setValue(self.selected_thresholds[sec])

        try:
            self.gscene_thresholded.set_active_section(sec)
        except: # The first time this will complain "Image not loaded" yet. But will not once update_thresholded_image() loads the image.
            pass

        self.update_thresholded_image()

        try:
            self.gscene_slic.set_active_section(sec)
        except:
            pass

        try:
            self.gscene_dissimmap.set_active_section(sec)
        except:
            pass

        try:
            self.gscene_submasks.set_active_section(sec)
        except:
            pass

        if sec in self.selected_dissim_thresholds:
            self.ui.slider_dissimThresh.setValue(int(self.selected_dissim_thresholds[sec]/0.01))

        try:
            self.gscene_snake.set_active_section(sec)
        except:
            pass


    def submask_added(self, polygon):
        curr_fn = self.valid_sections_to_filenames[self.gscene_finalMask.active_section]
        existing_submask_indices = self.mask_review_results[curr_fn].keys()
        if len(existing_submask_indices) == 0:
            first_available_submask_ind = 1
        else:
            available_submask_indices = [i for i in range(1, np.max(existing_submask_indices) + 2) if i not in existing_submask_indices]
            first_available_submask_ind = available_submask_indices[0]
        new_submask_ind = first_available_submask_ind

        # Update submask review result
        self.mask_review_results[curr_fn][new_submask_ind] = True

        # Update submask list
        new_submask_contours = vertices_from_polygon(polygon)

        bbox = self.gscene_finalMask.pixmapItem.boundingRect()
        image_shape = (int(bbox.height()), int(bbox.width()))
        new_submask = contours_to_mask([new_submask_contours], image_shape)
        self.all_submasks[curr_fn][new_submask_ind] = new_submask

        if curr_fn not in self.fns_submask_modified:
            self.fns_submask_modified.append(curr_fn)

        sys.stderr.write('Submask %d added.\n' % new_submask_ind)

    def submask_deleted(self, polygon):
        submask_ind = self.gscene_finalMask.drawings[self.gscene_finalMask.active_i].index(polygon) + 1
        curr_fn = self.valid_sections_to_filenames[self.gscene_finalMask.active_section]

        # Update submask review result
        del self.mask_review_results[curr_fn][submask_ind]

        # Update submask list
        del self.all_submasks[curr_fn][submask_ind]

        sys.stderr.write('Submask %d removed.\n' % submask_ind)
        self.update_mask_gui_window_title()

    def submask_clicked(self, polygon):
        print polygon
        print self.gscene_finalMask.drawings[self.gscene_finalMask.active_i]
        submask_ind = self.gscene_finalMask.drawings[self.gscene_finalMask.active_i].index(polygon) + 1
        sys.stderr.write('Submask %d clicked.\n' % submask_ind)

        curr_fn = self.valid_sections_to_filenames[self.gscene_finalMask.active_section]

        print self.mask_review_results[curr_fn]

        curr_decision = self.mask_review_results[curr_fn][submask_ind]
        self.update_submask_decision(submask_ind=submask_ind, decision=not curr_decision, fn=curr_fn)

        if curr_fn not in self.fns_submask_modified:
            self.fns_submask_modified.append(curr_fn)

        self.update_mask_gui_window_title()


    def get_mask_sec_fn(self, sec=None, fn=None):
        if sec is None:
            if fn is None:
                sec = self.gscene_finalMask.active_section
                fn = self.valid_sections_to_filenames[sec]
            else:
                sec = self.valid_filenames_to_sections[fn]
        else:
            if fn is None:
                fn = self.valid_sections_to_filenames[sec]
            else:
                assert fn == self.valid_sections_to_filenames[sec]
        return sec, fn

    def update_submask_decisions_all_sections(self, decisions_all_sections):
        for fn, decisions in decisions_all_sections.iteritems():
            self.update_submask_decisions(decisions, fn=fn)

    def update_submask_decisions(self, decisions, sec=None, fn=None):
        """
        decisions: dict {fn: bool}
        """
        _, fn = self.get_mask_sec_fn(sec=sec, fn=fn)
        for submask_ind, decision in decisions.iteritems():
            self.update_submask_decision(submask_ind, decision, fn=fn)

    def update_submask_decision(self, submask_ind, decision, sec=None, fn=None):

        sec, fn = self.get_mask_sec_fn(sec=sec, fn=fn)

        if submask_ind in self.mask_review_results[fn]:
            curr_decision = self.mask_review_results[fn][submask_ind]
            if curr_decision == decision:
                return

        self.mask_review_results[fn][submask_ind] = decision
        self.update_mask_gui_window_title()

        if decision:
            pen = QPen(Qt.green)
        else:
            pen = QPen(Qt.red)
        pen.setWidth(2)

        curr_i, _ = self.gscene_finalMask.get_requested_index_and_section(sec=sec)
        self.gscene_finalMask.drawings[curr_i][submask_ind-1].setPen(pen)

    def update_mask_gui_window_title(self):
        curr_sec = self.gscene_finalMask.active_section
        curr_fn = self.valid_sections_to_filenames[curr_sec]
        self.dialog.setWindowTitle('%s (%d) %s' % (curr_fn, curr_sec, self.mask_review_results[curr_fn]))
        print '%s (%d) %s' % (curr_fn, curr_sec, self.mask_review_results[curr_fn])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Mask Editing GUI')
    parser.add_argument("stack", type=str, help="stack name")
    args = parser.parse_args()
    stack = args.stack

    app = QApplication(argv)

    m = MaskEditingGUI(stack=stack)
    # m.showMaximized()
    m.show()
    exit(app.exec_())
