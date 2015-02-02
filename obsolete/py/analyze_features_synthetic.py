# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

%autosave 0

# <codecell>

from preamble import *

# <codecell>

pattern = np.ones((100,100))
pattern[:,0:50:5] = 0
pattern[:,50::10] = 0
plt.imshow(pattern, cmap=plt.cm.Greys_r)

# <codecell>

from skimage.transform import rotate

w = 10
h = 10

im_stripes = np.zeros((h, w))

stripe_spacing = 4
stripe_width = 2

for row in range(0, h, stripe_spacing):
    im_stripes[row:row+stripe_width] = 1

thetas = np.random.randint(0, 6, 8)*30
ims = [rotate(im_stripes, -theta) > 0.5 for theta in thetas]
    
pattern = np.vstack([np.hstack(ims[:3]), np.hstack([ims[3], np.zeros((h, w)), ims[4]]), np.hstack(ims[5:])])

plt.imshow(pattern, cmap=plt.cm.Greys_r)

# <codecell>

from joblib import Parallel, delayed
from scipy.signal import fftconvolve

def convolve_per_proc(i):
    return fftconvolve(pattern, dm.kernels[i], 'same')

filtered = Parallel(n_jobs=16)(delayed(convolve_per_proc)(i) for i in range(dm.n_kernel))
features = np.array(filtered)

# <codecell>

vmin = features.min()
vmax = features.max()

# <codecell>

plt.matshow(features[:,20,20].reshape(dm.n_freq, dm.n_angle), vmin=vmin, vmax=vmax)
plt.colorbar();

# <codecell>

plt.matshow(features[:,80,80].reshape(dm.n_freq, dm.n_angle), vmin=vmin, vmax=vmax)
plt.colorbar();

# <codecell>

from skimage.util import pad
padded_kernels = [None] * dm.n_kernel
for i, kern in enumerate(dm.kernels):
    ksize = kern.shape[0]
    a = (dm.max_kern_size - ksize)/2
    padded_kern = pad(kern, [a, a], mode='constant', constant_values=0)
    padded_kernels[i] = padded_kern

# <codecell>

F = np.vstack([k.flatten() for k in padded_kernels])

# <codecell>

import itertools
f = np.column_stack([features[:,i,j] for i,j in itertools.product(range(pattern.shape[0]), range(pattern.shape[1]))])

# <codecell>

from numpy.linalg import lstsq
b, r, _, _ = lstsq(F, features[:,25,25])

c = b.reshape((dm.max_kern_size,dm.max_kern_size))

plt.imshow(c[dm.max_kern_size/2-10:dm.max_kern_size/2+10,
             dm.max_kern_size/2-10:dm.max_kern_size/2+10], cmap=plt.cm.Greys_r)

