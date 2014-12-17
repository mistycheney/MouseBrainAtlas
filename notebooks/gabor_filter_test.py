# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

from preamble import *

# <codecell>

features = dm.load_pipeline_result('features', 'npy')
features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

# <codecell>

f1 = features[:,817,4480]
f2 = features[:,835,4502]
f3 = features[:,847,4522]
f4 = features[:,1130,3741]

patch_min = min(f1.min(), f2.min(), f3.min(), f4.min())
patch_max = max(f1.max(), f2.max(), f3.max(), f4.max())

plt.matshow(f1.reshape(dm.n_freq, dm.n_angle), vmin=patch_min, vmax=patch_max)
plt.show()
plt.matshow(f2.reshape(dm.n_freq, dm.n_angle), vmin=patch_min, vmax=patch_max)
plt.show()
plt.matshow(f3.reshape(dm.n_freq, dm.n_angle), vmin=patch_min, vmax=patch_max)
plt.show()
plt.matshow(f4.reshape(dm.n_freq, dm.n_angle), vmin=patch_min, vmax=patch_max)

