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

from matplotlib.backend_bases import key_press_handler, MouseEvent, KeyEvent
from matplotlib.backends.backend_qt4agg import (
	FigureCanvasQTAgg as FigureCanvas,
	NavigationToolbar2QT as NavigationToolbar)
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

from ui_BrainLabelingGui_v11 import Ui_BrainLabelingGui

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib.colors import ListedColormap, NoNorm, ColorConverter
from matplotlib.path import Path
from matplotlib.text import Text

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import LinearRing as ShapelyLineRing

from skimage.color import label2rgb

from visualization_utilities import *

sys.path.append(os.environ['LOCAL_REPO_DIR'] + '/utilities')
from utilities2015 import *

from collections import defaultdict, OrderedDict, deque
from scipy.spatial.distance import cdist

from operator import attrgetter

import requests

from joblib import Parallel, delayed

from enum import Enum
class Mode(Enum):
	REVIEW_PROPOSAL = 'review proposal'
	IDLE = 'idle'
	MOVING_POLYGON = 'moving polygon'
	MOVING_VERTEX = 'moving vertex'
	CREATING_NEW_POLYGON = 'create new polygon'
	ADDING_VERTICES_CONSECUTIVELY = 'adding vertices consecutively'
	ADDING_VERTICES_RANDOMLY = 'adding vertices randomly'
	KEEP_SELECTION = 'keep selection'
	SELECT_UNCERTAIN_SEGMENT = 'select uncertain segment'
	DELETE_ROI_MERGE = 'delete roi (merge)'
	DELETE_ROI_DUPLICATE = 'delete roi (duplicate)'
	DELETE_BETWEEN = 'delete edges between two vertices'
	CONNECT_VERTICES = 'connect two vertices'

class ProposalType(Enum):
	GLOBAL = 'global'
	LOCAL = 'local'
	FREEFORM = 'freeform'
	ALGORITHM = 'algorithm'

class PolygonType(Enum):
	CLOSED = 'closed'
	OPEN = 'open'
	TEXTURE = 'textured'
	TEXTURE_WITH_CONTOUR = 'texture with contour'
	DIRECTION = 'directionality'

SELECTED_POLYGON_LINEWIDTH = 5
UNSELECTED_POLYGON_LINEWIDTH = 3
SELECTED_CIRCLE_SIZE = 30
UNSELECTED_CIRCLE_SIZE = 5
CIRCLE_PICK_THRESH = 1000.
PAN_THRESHOLD = 10

HISTORY_LEN = 20

AUTO_EXTEND_VIEW_TOLERANCE = 200

# NUM_NEIGHBORS_PRELOAD = 1 # preload neighbor sections before and after this number
VERTEX_CIRCLE_RADIUS = 10

class LabelingPainter(object):
	'''
	Control the output to the QGraphicsView and QGraphicsScene, 
	according to input
	'''
	def __init__(self, gview, gscene, pixmap):
		self.gview = gview
		self.gscene = gscene
		self.pixmap = pixmap
		self.accepted_proposals = {}

	def set_scene(self, gscene):
		self.gview.setScene(gscene)

	# def set_section(self, section):
	# 	self.section = section

	# def add_vertex(self, ):
