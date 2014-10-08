import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *


# QFileSystemModel might be useful http://qt-project.org/doc/qt-4.8/qfilesystemmodel.html

def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

def convert_to_QStandardItem(d, root_key):

    root_item = QStandardItem(root_key)
    if isinstance(d, list):
        for k in d:
            leaf_item = QStandardItem(k)
            root_item.appendRow(leaf_item)
    elif isinstance(d, dict):
        for k, v in d.iteritems():
            sub_item = convert_to_QStandardItem(v, k)
            root_item.appendRow(sub_item)

    return root_item

def convert_to_QStandardItemModel(d):

    m = QStandardItemModel()
    if isinstance(d, list):
        for k in d:
            leaf_item = QStandardItem(k)
            m.appendRow(leaf_item)
    elif isinstance(d, dict):
        for k, v in d.iteritems():
            m.appendRow(convert_to_QStandardItem(v, k))

    return m