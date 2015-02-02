import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *

# QFileSystemModel might be useful http://qt-project.org/doc/qt-4.8/qfilesystemmodel.html

def paths_to_dict(paths):

    level_counts = [len(p.split('_')) for p in paths]
    n_levels = max(level_counts) + 1
    paths_same_level = [[] for _ in range(n_levels)]
    for p, lc in zip(paths, level_counts):
        paths_same_level[lc].append(p)
        
    res_dict = {}
    for l in range(n_levels):
        for p in paths_same_level[l]:
            elements = p.split('/')
            parent = reduce(dict.get, elements[:-1], res_dict)
            parent[elements[-1]] = {}

    return res_dict


def pad_paths(paths, level, key_list):
    for p in paths:
        elements = p.split('/')
        if len(elements) == level:
            prefix = '/'.join(elements[:level])
            paths += [prefix + '/' + k for k in key_list]
    return list(set(paths))


def pad_dict(d, level, key_list):
    if level == 0:
        for p in key_list:
            if p not in d:
                d[p] = None
    else:
        for k, v in d.iteritems():
            pad_dict(v, level=level-1, key_list=key_list)
    return d


def dict_to_paths(d, prefix=None):
    paths = []
    for k, v in d.iteritems():
        if prefix is None:
            full_key = k
        else:
            full_key = prefix + '/' + k
        paths.append(full_key)
        if v is not None: # if this is not a file, but is a directory
            paths += dict_to_paths(v, prefix=full_key)
    return paths


class Node(object):
    def __init__(self, parent=None, name='', value=None):
        self.parent = parent
        self.children = []
        self.name = name
        self.value = value

    # def convert_to_dict(self):


def find_child_by_name(node, name):
    return [c for c in node.children if c.name == name][0]


def paths_to_tree(paths):

    level_counts = [len(p.split('/')) for p in paths]
    n_levels = max(level_counts) + 1
    paths_same_level = [[] for _ in range(n_levels)]
    for pv_tuple, lc in zip(paths, level_counts):
        paths_same_level[lc].append(pv_tuple)
        
    root = Node('root', None)
    for l in range(n_levels):
        for p in paths_same_level[l]:
            elements = p.split('/')
            parent = reduce(find_child_by_name, elements[:-1], root)
            parent.children.append(Node(parent, elements[-1]))

    return root

# def labeled_paths_to_tree(paths):

#     level_counts = [len(p.split('/')) for p, v in paths]
#     n_levels = max(level_counts) + 1
#     paths_same_level = [[] for _ in range(n_levels)]
#     for pv_tuple, lc in zip(paths, level_counts):
#         paths_same_level[lc].append(pv_tuple)
        
#     root = Node('root', None)
#     for l in range(n_levels):
#         for p, v in paths_same_level[l]:
#             elements = p.split('/')
#             parent = reduce(find_child_by_name, elements[:-1], root)
#             parent.children.append(Node(parent, elements[-1], v))

#     return root

def paths_to_QStandardModel(paths):
    tree = paths_to_tree(paths)
    model = tree_to_QStandardItemModel(tree)
    return model


def tree_to_QStandardItem(node):
    # label = '%s(%s)' % (node.name, node.value)
    label = node.name
    root = QStandardItem(label)
    if len(node.children) > 0:
        for c in node.children:
            sub_tree = tree_to_QStandardItem(c)
            root.appendRow(sub_tree)
    return root


def tree_to_QStandardItemModel(node):
    m = QStandardItemModel()
    for c in node.children:
        m.appendRow(tree_to_QStandardItem(c))
    return m


# def convert_to_QStandardItem(d, root_key):
#     root_item = QStandardItem(root_key)
#     if isinstance(d, list):
#         for k in d:
#             leaf_item = QStandardItem(k)
#             root_item.appendRow(leaf_item)
#     elif isinstance(d, dict):
#         for k, v in d.iteritems():
#             sub_item = convert_to_QStandardItem(v, k)
#             root_item.appendRow(sub_item)

#     return root_item

# def convert_to_QStandardItemModel(d):

#     m = QStandardItemModel()
#     if isinstance(d, list):
#         for k in d:
#             leaf_item = QStandardItem(k)
#             m.appendRow(leaf_item)
#     elif isinstance(d, dict):
#         for k, v in d.iteritems():
#             m.appendRow(convert_to_QStandardItem(v, k))

#     return m