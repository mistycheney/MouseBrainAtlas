"""
Interactive Matplotlib Color Selector

Written by Jake Vanderplas <jakevdp@cs.washington.edu>
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, AxesWidget
from matplotlib.colors import rgb2hex

# Initialize labels and image
labels=np.zeros((16,16))
im_arr = np.ones((256, 256, 3), dtype=np.float32)
#draw grid
for i in range(0,256,16):
    im_arr[i,:]=0
    im_arr[:,i]=0

def paint(i,j):
    " change the color at location i,j and redraw "
    labels[i,j]=(labels[i,j]+1) % 3
    color=[0,0,0]; color[int(labels[i,j])]=1.0
    im_arr[16*i:16*(i+1),16*j:16*(j+1),:] = color

    im = main_ax.imshow(im_arr, extent=[0, 1, 0, 1])


fig = plt.figure(figsize=(8, 8))
main_ax = plt.axes([0.1, 0.15, 0.8, 0.8], xticks=[], yticks=[])
cbox_ax = plt.axes([0.1, 0.04, 0.04, 0.04], xticks=[], yticks=[])


im = main_ax.imshow(im_arr, extent=[0, 1, 0, 1], picker=True)
cbox_im = cbox_ax.imshow(np.ones((1, 1, 3)))
cbox_txt = cbox_ax.text(1.5, 0.5, "", transform=cbox_ax.transAxes,
                        fontsize=20, va='center', ha='left')
def on_pick(event):
    x = event.mouseevent.xdata
    y = event.mouseevent.ydata
    print 'x=',int(x*16),'y=',int(y*16)
    paint(int((1-y)*16),int(x*16))
    cbox_txt.set_text('x='+str(x/16)+', y='+str(y/16))
    fig.canvas.draw()
    
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
