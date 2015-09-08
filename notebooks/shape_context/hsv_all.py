import os

import numpy as np
from matplotlib import pyplot as pp
from skimage.color import hsv2rgb

import texture as t

ANGLE = 20

filtered = os.path.expanduser("~/devel/results/filtered")
hsv_results = os.path.expanduser("~/devel/results/hsv")
magnitude_results = os.path.expanduser("~/devel/results/magnitude")
angle_results = os.path.expanduser("~/devel/results/angle")

filenames = list(f for f in os.listdir(filtered)
                 if os.path.splitext(f)[1] == ".npy")
n_files = len(filenames)

for i, filename in enumerate(filenames):
    print "processing {} of {}".format(i, n_files)
    arr = np.load(os.path.join(filtered, filename))
    magnitude, angle = t.directionality_filter(arr, angle=ANGLE)

    magnitude_file = os.path.splitext(filename)[0] + ".png"
    magnitude_path = os.path.join(magnitude_results, magnitude_file)
    pp.imsave(magnitude_path, magnitude, cmap=pp.cm.gray)

    angle_file = os.path.splitext(filename)[0] + ".png"
    angle_path = os.path.join(angle_results, angle_file)
    pp.imsave(angle_path, angle)

    hsv_file = os.path.splitext(filename)[0] + ".png"
    hsv_path = os.path.join(hsv_results, hsv_file)
    hsv = hsv2rgb(t.make_hsv(magnitude, angle))
    pp.imsave(hsv_path, hsv)
