import numpy as np
from joblib import Parallel, delayed
from itertools import product
import os

import sys
sys.path.append(os.environ['GORDON_REPO_DIR'] + '/pipeline_scripts')
from utilities2014 import *

import matplotlib.pyplot as plt

from tables import *

import compute_pie

dm = DataManager(generate_hierarchy=False, stack='RS141', resol='x5', section=17)
dm._load_image()

textonmap = dm.load_pipeline_result('texMap', 'npy')
textonmap_viz = dm.load_pipeline_result('texMap', 'png')
n_texton = len(np.unique(textonmap)) - 1

image = dm.image_rgb
mask = dm.mask

mys, mxs = np.where(mask)
mys = mys.astype(np.int16)
mxs = mxs.astype(np.int16)

height, width = image.shape[:2]

G_nonmaxsup = np.load('/home/yuncong/csd395/G_nonmaxsup.npy')

r = 5
rho = .5

conns_ij_y = []
conns_ij_x = []
circle_j = []
for y, x in product(range(-r, r+1), range(-r, r+1)):
    d = np.sqrt(y**2+x**2)
    if d < r and not (y==0 and x==0):
        pts_conn_ij_y = np.linspace(0,y,d).astype(np.int)
        pts_conn_ij_x = np.linspace(0,x,d).astype(np.int)
        circle_j.append((y,x))
        conns_ij_y.append(pts_conn_ij_y)
        conns_ij_x.append(pts_conn_ij_x)

circle_j = np.asarray(circle_j, dtype=np.int16)

b = time.time()

# A = compute_pie.compute_connection_weight(G_nonmaxsup, circle_j, conns_ij_y, conns_ij_x,
#                                           mys[:100], mxs[:100], height, width, mask)

A = Parallel(n_jobs=16)(delayed(compute_pie.compute_connection_weight)(G_nonmaxsup, circle_j,
                                                                       conns_ij_y, conns_ij_x,
                                                                       mys[s], mxs[s], height, width, mask)
                        for s in np.array_split(range(len(mys)), 16))


A = np.vstack(A)

print time.time()-b


filters = Filters(complevel=9, complib='blosc')

with open_file('/home/yuncong/csd395/A.hdf', mode="w") as f:
    _ = f.create_carray('/', 'data', FloatAtom(), filters=filters, obj=A)

