import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, LineString, Point
import numpy as np

import re

def get_list_from_model(model):
    return [str(model.data(model.index(index, 0))) for index in range(model.rowCount())]

def fill_item_to_tree_widget(item, value):
    # http://stackoverflow.com/questions/21805047/qtreewidget-to-mirror-python-dictionary
    # http://stackoverflow.com/questions/31342228/pyqt-tree-widget-adding-check-boxes-for-dynamic-removal
    # http://stackoverflow.com/questions/27521391/signal-a-qtreewidgetitem-toggled-checkbox

    item.setExpanded(True)
    if type(value) is dict:
        for key, val in sorted(value.iteritems()):
            child = QTreeWidgetItem()
            child.setFlags(child.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            child.setText(0, key)
            child.setCheckState(0, Qt.Checked)
            item.addChild(child)
            fill_item_to_tree_widget(child, val)

def decide_whether_enable(node, valid_abbrs):
    # http://stackoverflow.com/questions/8961449/pyqt-qtreewidget-iterating

    abbr = re.findall('^.*?(\((.*)\))?$', str(node.text(0)))[0][1]
    should_enable = abbr in valid_abbrs

    child_count = node.childCount()
    if child_count > 0:
        should_enable = should_enable | any([decide_whether_enable(node.child(i), valid_abbrs) for i in range(child_count)])

    node.setDisabled(not should_enable)
    return should_enable

def fill_tree_widget(widget, value, valid_abbrs):
    """
    Invalid names are greyed out.
    """
    # http://stackoverflow.com/questions/21805047/qtreewidget-to-mirror-python-dictionary
    widget.clear()
    fill_item_to_tree_widget(widget.invisibleRootItem(), value)
    decide_whether_enable(widget.invisibleRootItem(), valid_abbrs)

def subpath(path, begin, end):

    new_path = QPainterPath()

    is_closed = polygon_is_closed(path=path)
    n = path.elementCount() - 1 if is_closed else path.elementCount()

    if not is_closed:
        assert end >= begin
        begin = max(0, begin)
        end = min(n-1, end)
    else:
        assert end != begin # cannot handle this, because there is no way a path can have the same first and last points but is not closed
        if end < begin:
            end = end + n

    for i in range(begin, end + 1):
        elem = path.elementAt(i % n)
        if new_path.elementCount() == 0:
            new_path.moveTo(elem.x, elem.y)
        else:
            new_path.lineTo(elem.x, elem.y)

    assert new_path.elementCount() > 0

    return new_path


def split_path(path, vertex_indices):
    '''
    Split path.

    Args:
        path (QPainterPath): input path
        vertex_indices (list of tuples or n x 2 numpy array): indices in the cut box

    Returns:
        list of QPainterPath: paths in cut box
        list of QPainterPath: paths outside of cut box

    '''

    is_closed = polygon_is_closed(path=path)
    n = polygon_num_vertices(path=path, closed=is_closed)

    segs_in, segs_out = split_array(vertex_indices, n, is_closed)

    print segs_in, segs_out

    in_paths = []
    out_paths = []

    for b, e in segs_in:
        in_path = subpath(path, b, e)
        in_paths.append(in_path)

    for b, e in segs_out:
        out_path = subpath(path, b-1, e+1)
        out_paths.append(out_path)

    return in_paths, out_paths


def split_array(vertex_indices, n, is_closed):

    cache = [i in vertex_indices for i in range(n)]

    i = 0

    sec_outs = []
    sec_ins = []

    sec_in = [None,None]
    sec_out = [None,None]

    while i != (n+1 if is_closed else n):

        if cache[i%n] and not cache[(i+1)%n]:
            sec_in[1] = i%n
            sec_ins.append(sec_in)
            sec_in = [None,None]

            sec_out[0] = (i+1)%n
        elif not cache[i%n] and cache[(i+1)%n]:
            sec_out[1] = i%n
            sec_outs.append(sec_out)
            sec_out = [None,None]

            sec_in[0] = (i+1)%n

        i += 1

    if sec_in[0] is not None or sec_in[1] is not None:
        sec_ins.append(sec_in)

    if sec_out[0] is not None or sec_out[1] is not None:
        sec_outs.append(sec_out)

    tmp = [None, None]
    for sec in sec_ins:
        if sec[0] is None and sec[1] is not None:
            tmp[1] = sec[1]
        elif sec[0] is not None and sec[1] is None:
            tmp[0] = sec[0]
    if tmp[0] is not None and tmp[1] is not None:
        sec_ins = [s for s in sec_ins if s[0] is not None and s[1] is not None] + [tmp]
    else:
        sec_ins = [s for s in sec_ins if s[0] is not None and s[1] is not None]

    tmp = [None, None]
    for sec in sec_outs:
        if sec[0] is None and sec[1] is not None:
            tmp[1] = sec[1]
        elif sec[0] is not None and sec[1] is None:
            tmp[0] = sec[0]
    if tmp[0] is not None and tmp[1] is not None:
        sec_outs = [s for s in sec_outs if s[0] is not None and s[1] is not None] + [tmp]
    else:
        sec_outs = [s for s in sec_outs if s[0] is not None and s[1] is not None]

    if not is_closed:
        sec_ins2 = []
        for sec in sec_ins:
            if sec[0] > sec[1]:
                sec_ins2.append([sec[0], n-1])
                sec_ins2.append([0, sec[1]])
            else:
                sec_ins2.append(sec)

        sec_outs2 = []
        for sec in sec_outs:
            if sec[0] > sec[1]:
                sec_outs2.append([sec[0], n-1])
                sec_outs2.append([0, sec[1]])
            else:
                sec_outs2.append(sec)

        return sec_ins2, sec_outs2

    else:
        return sec_ins, sec_outs

def insert_vertex(path, x, y, new_index):

    new_path = QPainterPath()
    for i in range(path.elementCount()+1): # +1 is important, because the new_index can be after the last vertex
        if i == new_index:
            path_goto(new_path, x, y)
        if i < path.elementCount():
            elem = path.elementAt(i)
            path_goto(new_path, elem.x, elem.y)
    return new_path


def delete_between(path, first_index, second_index):

    print first_index, second_index

    if second_index < first_index:    # ensure first_index is smaller than second_index
        temp = first_index
        first_index = second_index
        second_index = temp

    n = polygon_num_vertices(path=path)

    if (second_index - first_index > first_index + n - second_index):
        indices_to_remove = range(second_index, n+1) + range(0, first_index+1)
    else:
        indices_to_remove = range(first_index, second_index+1)

    print indices_to_remove

    paths_to_remove, paths_to_keep = split_path(path, indices_to_remove)
    assert len(paths_to_keep) == 1

    return paths_to_keep[0]


def delete_vertices(path, indices_to_remove, merge=False):
    if merge:
        new_path = delete_vertices_merge(path, indices_to_remove)
    else:
        paths_to_remove, paths_to_keep = split_path(path, indices_to_remove)


def delete_vertices_merge(path, indices_to_remove):

    is_closed = polygon_is_closed(path=path)
    n = polygon_num_vertices(path=path, closed=is_closed)

    segs_to_remove, segs_to_keep = split_array(indices_to_remove, n, is_closed)
    print "segs_to_remove:", segs_to_remove, "segs_to_keep:", segs_to_keep

    new_path = QPainterPath()
    for b, e in sorted(segs_to_keep):
        if e < b: e = e + n
        for i in range(b, e + 1):
            elem = path.elementAt(i % n)
            if new_path.elementCount() == 0:
                new_path.moveTo(elem.x, elem.y)
            else:
                new_path.lineTo(elem.x, elem.y)

    if is_closed:
        new_path.closeSubpath()

    return new_path


def polygon_goto(polygon, x, y):
    """Modifies polygon object inline.
    """

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
        vertices (n x 2 numpy array): vertices, (x,y)
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
