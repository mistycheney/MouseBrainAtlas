import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, LineString, Point
import numpy as np


def polygon_goto(polygon, x, y):

    path = polygon.path()

    if path.elementCount() == 0:
        path.moveTo(x, y)
        # print 'move to', x, y
    else:
        path.lineTo(x, y)
        # print 'line to', x, y

    polygon.setPath(path)


def path_goto(path, x, y):

    if path.elementCount() == 0:
        path.moveTo(x, y)
        # print 'move to', x, y
    else:
        path.lineTo(x, y)
        # print 'line to', x, y
    # return path


def vertices_from_polygon(polygon=None, path=None, closed=None):

    if path is None:
        path = polygon.path()

    if closed is None:
       closed = polygon_is_closed(polygon=polygon, path=path)

    if closed:
        vertices = [(int(path.elementAt(i).x), int(path.elementAt(i).y)) for i in range(path.elementCount()-1)]
    else:
        vertices = [(int(path.elementAt(i).x), int(path.elementAt(i).y)) for i in range(path.elementCount())]

    return np.array(vertices)

def polygon_to_shapely(polygon=None, path=None):
    if path is None:
        path = polygon.path()

    closed = polygon_is_closed(polygon=polygon, path=path)   
    vertices = vertices_from_polygon(polygon=polygon, path=path, closed=closed)

    if len(vertices) == 1:
        return Point(vertices[0])

    if closed:
        return Polygon(vertices)
    else:
        return LineString(vertices)


def polygon_is_closed(polygon=None, path=None):

    if path is None:
        path = polygon.path()

    assert path.elementCount() > 0, 'Polygon has no point.'

    if path.elementCount() == 1: # if polygon has only one vertex, count as open
        return False

    e0 = path.elementAt(0)
    e1 = path.elementAt(path.elementCount()-1)
    is_closed = (e0.x == e1.x and e0.y == e1.y)

    return is_closed

def polygon_num_vertices(polygon=None, path=None, closed=None):
    if path is None:
        path = polygon.path()

    if path.elementCount() == 0:
        return 0

    if closed is None:
        closed = polygon_is_closed(path=path)

    n = path.elementCount() - 1 if closed else path.elementCount()
    return n


def vertices_to_path(vertices, closed=True):
    '''
    Generate QPainterPath from vertices.

    Args:
        vertices (n x 2 numpy array): vertices
        closed (bool): whether the polygon is closed; if polygon has only one vertex, count as open

    Returns:
        QPainterPath: the output path
    '''

    path = QPainterPath()

    for i, (x, y) in enumerate(vertices):
        if i == 0:
            path.moveTo(x,y)
        else:
            path.lineTo(x,y)

    if len(vertices) == 1:
        closed = False  # if polygon has only one vertex, count as open

    if closed:
        path.closeSubpath()

    return path

def find_vertex_insert_position(polygon, x, y):

    path = polygon.path()

    if path.elementCount() == 0:
        return 0

    is_closed = polygon_is_closed(path=path)

    pos = (x,y)

    xys = vertices_from_polygon(path=path, closed=is_closed)

    n = len(xys)
    if n == 1:
        return 1

    xys_homo = np.column_stack([xys, np.ones(n,)])

    if is_closed:
        edges = np.array([np.cross(xys_homo[i], xys_homo[(i+1)%n]) for i in range(n)])
    else:
        edges = np.array([np.cross(xys_homo[i], xys_homo[i+1]) for i in range(n-1)])

    edges_normalized = edges/np.sqrt(np.sum(edges[:,:2]**2, axis=1))[:, np.newaxis]

    signed_dists = np.dot(edges_normalized, np.r_[pos,1])
    dists = np.abs(signed_dists)
    # sides = np.sign(signed_dists)

    projections = pos - signed_dists[:, np.newaxis] * edges_normalized[:,:2]

    endpoint = [None for _ in projections]
    for i, (px, py) in enumerate(projections):
        if (px > xys[i][0] and px > xys[(i+1)%n][0]) or (px < xys[i][0] and px < xys[(i+1)%n][0]):
            endpoint[i] = [i, (i+1)%n][np.argmin(np.squeeze(cdist([pos], [xys[i], xys[(i+1)%n]])))]
            dists[i] = np.min(np.squeeze(cdist([pos], [xys[i], xys[(i+1)%n]])))

    # print edges_normalized[:,:2]
    # print projections                
    # print dists
    # print endpoint
    nearest_edge_begins_at = np.argsort(dists)[0]

    if nearest_edge_begins_at == 0 and not is_closed and endpoint[0] == 0:
        new_vertex_ind = 0
    elif nearest_edge_begins_at == n-2 and not is_closed and endpoint[-1] == n-1:
        new_vertex_ind = n
    else:
        new_vertex_ind = nearest_edge_begins_at + 1  

    print 'nearest_edge_begins_at', nearest_edge_begins_at, 'new_vertex_ind', new_vertex_ind

    return new_vertex_ind


################################################

# QFileSystemModel might be useful http://qt-project.org/doc/qt-4.8/qfilesystemmodel.html

# def paths_to_dict(paths):

#     level_counts = [len(p.split('_')) for p in paths]
#     n_levels = max(level_counts) + 1
#     paths_same_level = [[] for _ in range(n_levels)]
#     for p, lc in zip(paths, level_counts):
#         paths_same_level[lc].append(p)
        
#     res_dict = {}
#     for l in range(n_levels):
#         for p in paths_same_level[l]:
#             elements = p.split('/')
#             parent = reduce(dict.get, elements[:-1], res_dict)
#             parent[elements[-1]] = {}

#     return res_dict


# def pad_paths(paths, level, key_list):
#     for p in paths:
#         elements = p.split('/')
#         if len(elements) == level:
#             prefix = '/'.join(elements[:level])
#             paths += [prefix + '/' + k for k in key_list]
#     return list(set(paths))


# def pad_dict(d, level, key_list):
#     if level == 0:
#         for p in key_list:
#             if p not in d:
#                 d[p] = None
#     else:
#         for k, v in d.iteritems():
#             pad_dict(v, level=level-1, key_list=key_list)
#     return d


# def dict_to_paths(d, prefix=None):
#     paths = []
#     for k, v in d.iteritems():
#         if prefix is None:
#             full_key = k
#         else:
#             full_key = prefix + '/' + k
#         paths.append(full_key)
#         if v is not None: # if this is not a file, but is a directory
#             paths += dict_to_paths(v, prefix=full_key)
#     return paths


# class Node(object):
#     def __init__(self, parent=None, name='', value=None):
#         self.parent = parent
#         self.children = []
#         self.name = name
#         self.value = value

#     # def convert_to_dict(self):


# def find_child_by_name(node, name):
#     return [c for c in node.children if c.name == name][0]


# def paths_to_tree(paths):

#     level_counts = [len(p.split('/')) for p in paths]
#     n_levels = max(level_counts) + 1
#     paths_same_level = [[] for _ in range(n_levels)]
#     for pv_tuple, lc in zip(paths, level_counts):
#         paths_same_level[lc].append(pv_tuple)
        
#     root = Node('root', None)
#     for l in range(n_levels):
#         for p in paths_same_level[l]:
#             elements = p.split('/')
#             parent = reduce(find_child_by_name, elements[:-1], root)
#             parent.children.append(Node(parent, elements[-1]))

#     return root

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

# def paths_to_QStandardModel(paths):
#     tree = paths_to_tree(paths)
#     model = tree_to_QStandardItemModel(tree)
#     return model


# def tree_to_QStandardItem(node):
#     # label = '%s(%s)' % (node.name, node.value)
#     label = node.name
#     root = QStandardItem(label)
#     if len(node.children) > 0:
#         for c in node.children:
#             sub_tree = tree_to_QStandardItem(c)
#             root.appendRow(sub_tree)
#     return root


# def tree_to_QStandardItemModel(node):
#     m = QStandardItemModel()
#     for c in node.children:
#         m.appendRow(tree_to_QStandardItem(c))
#     return m

####################################################

