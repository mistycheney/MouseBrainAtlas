"""
Paint by numbers GUI
Written by Yoav Freund
"""
import numpy as np
import pylab
from matplotlib import pyplot as plt
import matplotlib as mpl
import Image
from time import time

#subsample image (this is put here to improve reaction time)
from scipy.signal import convolve2d
def subsample(I,smooth=False,k=4):
    h,w=np.shape(I)
    if smooth:
        kernel=np.ones([k,k])/(float(k*k))
        smoothed=convolve2d(I,kernel,mode='same',boundary='symm')
        return smoothed[k/2:h:k,k/2:w:k]
    else:
        return I[k/2:h:k,k/2:w:k]
        
def boundaries(x,t=10):
    """ Find segmentation boundaries
    x = segmentation matrix (different integer for each segment)
    t = thickness of boundaries
    output: a matrix whose dimensions are smaller than those of x by t (in each coordinate)
    """
    y1=abs(x[:-t,:-t]-x[t:,:-t])
    y2=abs(x[:-t,:-t]-x[:-t,t:])
    y=np.zeros(np.shape(x))
    l=t/2; r=t-l
    y[l:-r,l:-r]=((y1+y2)>0)*1
    return y
    
def mark_super_pixels(axes,seg,seg_no,indexes,colors=['r'],alpha=0.2):
    """
    color a list of super-pixels:
    axes = matplotlib.axes.Axes in which to draw the overlay
    seg = the segmentation map
    seg_no = the values in seg are integers in range(seg_no) 
    indexes = the indexes of the super-pixels
    colors : used to paint the segments. 
            This list can be shorter or longer than indexes,
            trimmed if longer
            repeated if shorter.
            
            Can use any color in:
            b : blue
            g : green
            r : red
            c : cyan
            m : magenta
            y : yellow
            k : black
            w : white
            or http://www.w3schools.com/html/html_colornames.asp
    """
    
    t0=time()
    Clist=['none']*seg_no
    nc=len(colors)
    for i in range(len(indexes)):
        Clist[indexes[i]]=colors[i % nc]
    Cmap = mpl.colors.ListedColormap(Clist)
    t1=time()
    axes.imshow(seg, cmap=Cmap,alpha=0.2)
    t2=time()
    #draw()
    t3=time()
    print 'times for mark_super_pixels:',t1-t0,t2-t1,t3-t2
    
def on_pick(event):
    x = event.mouseevent.xdata
    y = event.mouseevent.ydata
    s = seg[y,x]
    print 'x=',x,'y=',y,'seg=',s
    cbox_txt.set_text('x='+str(x)+', y='+str(y)+' seg='+str(seg[y,x]))
    mark_super_pixels(main_ax,seg,seg_no,[s],colors=['r'])
    fig.canvas.draw()

#read data
setName='PMD1305_reduce2_region0_0244_param5'
stem='../YuncongBrain/'+setName+'_data'

seg=np.load(stem+'/'+setName+'_segmentation.npy')
direction=np.load(stem+'/'+setName+'_sp_dir_hist_normalized.npy')
texton=np.load(stem+'/'+setName+'_sp_texton_hist_normalized.npy')

seg_no=np.shape(texton)[0]
print 'seg_no=',seg_no

# Initialize image
im=Image.open(stem+'/PMD1305_reduce2_region0_0244.tif')
aim=np.array(im)/256.0
bwaim=np.mean(aim,axis=2)  # make into BW picture

borders=boundaries(seg,t=3)
print np.shape(seg),np.shape(borders),np.shape(bwaim)

bwaim=subsample(bwaim,smooth=True,k=1)
seg=subsample(seg,smooth=False,k=1)
print np.shape(bwaim)

colormap=pylab.get_cmap('gray')
# trim im to size of seg
h1,w1=np.shape(bwaim)
h2,w2=np.shape(seg)
top=(h1-h2)/2; bottom=(h1-h2)-top
left=(w1-w2)/2; right=(w1-w2)-left

bwaim=bwaim[bottom:-top,left:-right]
h,w=h2,w2

fig = plt.figure(figsize=(8,8.0*h/w))
main_ax = plt.axes([0.05, 0.1, 0.9,0.9], xticks=[], yticks=[])
cbox_ax = plt.axes([0.0, 0.04, 0.04, 0.04], xticks=[], yticks=[])
cbox_txt = cbox_ax.text(1.5, 0.5, "", transform=cbox_ax.transAxes,
                        fontsize=20, va='center', ha='left')

main_ax.imshow(bwaim,cmap=colormap,aspect='equal',picker=True)
main_ax.imshow(borders,cmap=mpl.colors.ListedColormap(['w','m']),alpha=0.5)
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()

