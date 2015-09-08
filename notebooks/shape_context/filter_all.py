import os

import numpy as np
from matplotlib import pyplot as pp

import texture as t
import brainstem as b

RESULTS = os.path.expanduser("~/devel/results/filtered")

NAMES = b.get_filenames()
N_NAMES = len(NAMES)
ANGLE = 20

for i, name in enumerate(NAMES):
    print 'processing {} of {}: {}'.format(i, N_NAMES, name)
    img = b.get_cutout(name, rlevel=3)
    img = b.make_grey(img)
    filtered, _ = t.filter_img(img, angle=ANGLE)

    filter_file = os.path.splitext(name)[0] + "_filtered.npy"
    filter_path = os.path.join(RESULTS, filter_file)
    np.save(filter_path, filtered)
    
