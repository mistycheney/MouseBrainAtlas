import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, LineString, Point
import numpy as np


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
