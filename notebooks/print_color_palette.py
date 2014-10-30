# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

hc_colors = np.loadtxt('../visualization/high_contrast_colors.txt', skiprows=1)/255.

fig, axes = plt.subplots(ncols=len(hc_colors))
fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)

for i, (ax, c) in enumerate(zip(axes, hc_colors)):
    color_box = np.empty((1,1,3))
    color_box[:,:] = c
    ax.imshow(color_box, aspect='equal')
    pos = list(ax.get_position().bounds)
    x_text = pos[0] + pos[2]/2.
    y_text = pos[1] + pos[3]/2.
    fig.text(x_text, y_text, i-1, va='center', ha='center', fontsize=10)

for ax in axes:
    ax.set_axis_off()
    
plt.savefig('../visualization/hc_colors.tif', bbox_inches='tight')
# plt.show()

