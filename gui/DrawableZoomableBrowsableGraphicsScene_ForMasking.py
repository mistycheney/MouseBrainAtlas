from PyQt4.QtCore import *
from PyQt4.QtGui import *

from gui_utilities import *

import sys, os
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from data_manager import DataManager
from metadata import *

from DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene

from registration_utilities import find_contour_points

class DrawableZoomableBrowsableGraphicsScene_ForMasking(DrawableZoomableBrowsableGraphicsScene):
    """
    Extends base class by:
    - defining specific signal handlers
    """

    def __init__(self, id, gview=None, parent=None):
        super(DrawableZoomableBrowsableGraphicsScene_ForMasking, self).__init__(id=id, gview=gview, parent=parent)

        self.set_default_line_color('g')
        self.set_default_line_width(1)
        self.set_default_vertex_color('b')
        self.set_default_vertex_radius(2)

        self.drawings_updated.connect(self.submask_added)
        self.polygon_pressed.connect(self.submask_clicked)
        self.polygon_deleted.connect(self.submask_deleted)

        self.submasks = {}
        self.submask_decisions = {}

    def remove_submask_and_decisions_for_one_section(self, sec):
        if sec in self.submasks:
            del self.submasks[sec]
        if sec in self.submask_decisions:
            del self.submask_decisions[sec]
        self.delete_all_polygons_one_section(section=sec)

    def add_submask_and_decision_for_one_section(self, submasks, submask_decisions, sec):
        self.submasks[sec] = submasks
        self.submask_decisions[sec] = submask_decisions

        for submask_ind, decision in enumerate(submask_decisions):

            mask = submasks[submask_ind]

            # Compute contour of mask
            cnts = find_contour_points(mask, sample_every=1)[mask.max()]
            if len(cnts) == 0:
                raise Exception('ERROR: %d, %d - no contour' % (mask_ind, len(cnts)))
            elif len(cnts) > 1:
                sys.stderr.write('WARNING: %d, %d - multiple contours\n' % (mask_ind, len(cnts)))
                cnt = sorted(cnts, key=lambda c: len(c), reverse=True)[0]
            else:
                cnt = cnts[0]

            if decision:
                color = 'g'
            else:
                color = 'r'

            self.add_polygon(path=vertices_to_path(cnt), section=sec, linewidth=2, color=color)

    def set_submasks_and_decisions(self, submasks_allFiles, submask_decisions_allFiles):

        self.submasks = submasks_allFiles
        self.submask_decisions = submask_decisions_allFiles

        for sec, submasks in submasks_allFiles.iteritems():
            self.add_submask_and_decision_for_one_section(submasks, submask_decisions_allFiles[sec], sec=sec)

    def submask_added(self, polygon):
        existing_submask_indices = self.submask_decisions[self.active_section].keys()
        if len(existing_submask_indices) == 0:
            first_available_submask_ind = 0
        else:
            available_submask_indices = [i for i in range(np.max(existing_submask_indices) + 1) if i not in existing_submask_indices]
            first_available_submask_ind = available_submask_indices[0]
        new_submask_ind = first_available_submask_ind

        # Update submask decisions
        self.submask_decisions[self.active_section][new_submask_ind] = True

        # Update submask list
        new_submask_contours = vertices_from_polygon(polygon)

        bbox = self.pixmapItem.boundingRect()
        image_shape = (int(bbox.height()), int(bbox.width()))
        new_submask = contours_to_mask([new_submask_contours], image_shape)
        self.submasks[self.active_section][new_submask_ind] = new_submask

        # if curr_fn not in self.fns_submask_modified:
        #     self.fns_submask_modified.append(curr_fn)

        sys.stderr.write('Submask %d added.\n' % new_submask_ind)

    def submask_deleted(self, polygon):
        submask_ind = self.drawings[self.active_i].index(polygon)

        # Update submask decisions
        del self.submask_decisions[self.active_section][submask_ind]

        # Update submask list
        del self.submasks[self.active_section][submask_ind]

        sys.stderr.write('Submask %d removed.\n' % submask_ind)
        # self.update_mask_gui_window_title()

    def submask_clicked(self, polygon):
        submask_ind = self.drawings[self.active_i].index(polygon)
        sys.stderr.write('Submask %d clicked.\n' % submask_ind)

        curr_decision = self.submask_decisions[self.active_section][submask_ind]
        self.update_submask_decision(submask_ind=submask_ind, decision=not curr_decision, sec=self.active_section)

        # if curr_fn not in self.fns_submask_modified:
        #     self.fns_submask_modified.append(curr_fn)

        # self.update_mask_gui_window_title()


    def update_submask_decision(self, submask_ind, decision, sec):

        if submask_ind in self.submask_decisions[sec]:
            curr_decision = self.submask_decisions[sec][submask_ind]
            if curr_decision == decision:
                return

        self.submask_decisions[sec][submask_ind] = decision

        if decision:
            pen = QPen(Qt.green)
        else:
            pen = QPen(Qt.red)
        pen.setWidth(2)

        curr_i, _ = self.get_requested_index_and_section(sec=sec)
        self.drawings[curr_i][submask_ind].setPen(pen)
