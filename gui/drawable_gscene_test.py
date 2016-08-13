#! /usr/bin/env python

import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

import sys
import os
import datetime
from random import random
import subprocess
import time
import json
from pprint import pprint
import cPickle as pickle
from itertools import groupby
from operator import itemgetter

import numpy as np

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

from ui_BrainLabelingGui_v14 import Ui_BrainLabelingGui
from rectification_tool import *

from matplotlib.colors import ListedColormap, NoNorm, ColorConverter

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import LinearRing as ShapelyLineRing

from skimage.color import label2rgb

from gui_utilities import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import DataManager
from metadata import *

from collections import defaultdict, OrderedDict, deque

from operator import attrgetter

import requests

from joblib import Parallel, delayed

# from LabelingPainter import LabelingPainter
from custom_widgets import *
from SignalEmittingItems import *


from drawable_gscene import *

#######################################################################

class BrainLabelingGUI(QMainWindow, Ui_BrainLabelingGui):
	def __init__(self, parent=None, stack=None):
		"""
		Initialization of BrainLabelingGUI.
		"""
		# self.app = QApplication(sys.argv)
		QMainWindow.__init__(self, parent)

		self.stack = stack
		# self.initialize_brain_labeling_gui()
		self.setupUi(self)

		self.active_gscene_id = None

		self.section1_gscene = DrawableGraphicsScene(id=1, gui=self, gview=self.section1_gview)
		self.section2_gscene = DrawableGraphicsScene(id=2, gui=self, gview=self.section2_gview)
		self.section3_gscene = DrawableGraphicsScene(id=3, gui=self, gview=self.section3_gview)

		self.section1_gscene.drawings_updated.connect(self.drawings_updated)
		self.section2_gscene.drawings_updated.connect(self.drawings_updated)
		self.section3_gscene.drawings_updated.connect(self.drawings_updated)

		from collections import defaultdict
		self.drawings = defaultdict(list)

		self.section1_gscene.drawings = self.drawings
		self.section2_gscene.drawings = self.drawings
		self.section3_gscene.drawings = self.drawings

		self.section1_gview.setScene(self.section1_gscene)
		self.section2_gview.setScene(self.section2_gscene)
		self.section3_gview.setScene(self.section3_gscene)

		self.sections = [180,181,182]

		self.qimages = {sec_ind: QImage(DataManager.get_image_filepath(stack=self.stack, section=sec_ind, version='rgb-jpg', data_dir=data_dir))
						for sec_ind in self.sections}

		self.section1_gscene.set_qimages(self.qimages)
		self.section2_gscene.set_qimages(self.qimages)
		self.section3_gscene.set_qimages(self.qimages)
		self.section1_gscene.set_active_section(180)
		self.section2_gscene.set_active_section(181)
		self.section3_gscene.set_active_section(182)

		self.contextMenu_set = True

		self.recent_labels = []

		self.structure_names = load_structure_names(os.environ['REPO_DIR']+'/gui/structure_names.txt')
		self.new_labelnames = load_structure_names(os.environ['REPO_DIR']+'/gui/newStructureNames.txt')
		self.structure_names = OrderedDict(sorted(self.new_labelnames.items()) + sorted(self.structure_names.items()))

		self.installEventFilter(self)

		self.keyPressEvent = self.key_pressed

	@pyqtSlot(object)
	def drawings_updated(self, polygon):
		pass
		# self.drawings[polygon.section].append(polygon)
		# if polygon.gscene.id == 1:
		# 	[polygon.section].append(polygon)
		# 	self.section2_gscene.drawings[]

	def key_pressed(self, event):
		# if event.type() == Qt.keyPressEvent:
		key = event.key()
		if key == Qt.Key_3:
			self.section2_gscene.show_previous()
		elif key == Qt.Key_4:
			self.section2_gscene.show_next()
		elif key == Qt.Key_1:
			self.section1_gscene.show_previous()
		elif key == Qt.Key_2:
			self.section1_gscene.show_next()

	def eventFilter(self, obj, event):
		# print obj.metaObject().className(), event.type()

		return False




def load_structure_names(fn):
	names = {}
	with open(fn, 'r') as f:
		for ln in f.readlines():
			abbr, fullname = ln.split('\t')
			names[abbr] = fullname.strip()
	return names


if __name__ == "__main__":

	import argparse
	import sys
	import time

	parser = argparse.ArgumentParser(
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description='Launch brain labeling GUI.')

	parser.add_argument("stack_name", type=str, help="stack name")
	parser.add_argument("-n", "--num_neighbors", type=int, help="number of neighbor sections to preload, default %(default)d", default=1)
	args = parser.parse_args()

	from sys import argv, exit
	appl = QApplication(argv)

	stack = args.stack_name
	NUM_NEIGHBORS_PRELOAD = args.num_neighbors
	m = BrainLabelingGUI(stack=stack)

	m.showMaximized()
	m.raise_()
	exit(appl.exec_())
