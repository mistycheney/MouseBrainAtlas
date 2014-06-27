# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

cluster_data1, _ = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=.5, center_box=(5.0, 5.0), shuffle=True, random_state=None)
cluster_data2, _ = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=.5, center_box=(-5.0, -5.0), shuffle=True, random_state=None)
cluster_data = np.r_[cluster_data1, cluster_data2]

# bg_data, _ = make_blobs(n_samples=200, n_features=2, centers=1, cluster_std=5.0, center_box=(0., 0.), shuffle=True, random_state=None)

bg_data = np.random.random(size=(200,2))
bg_data = (bg_data-0.5)*20

data = np.r_[cluster_data, bg_data]
n_data = data.shape[0]

scatter(data[:,0], data[:,1]);
plt.show()
scatter(cluster_data[:,0], cluster_data[:,1], c='r')
scatter(bg_data[:,0], bg_data[:,1], c='b')
plt.show()

# <codecell>

from scipy.spatial.distance import pdist, squareform
D = pdist(data)
D = squareform(D)

# <codecell>

seed = np.random.randint(0, n_data)
scatter(data[:,0], data[:,1]);

nn = D.argsort(axis=1)[:,:50]
scatter(data[nn[seed],0], data[nn[seed],1], c='g');

scatter(data[seed,0], data[seed,1], c='r');

plt.show()

# <codecell>

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# <codecell>

import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal


scatter(data[:,0], data[:,1]);

nn = D.argsort(axis=1)[:,:10]
scatter(data[nn[seed],0], data[nn[seed],1], c='g');

scatter(data[seed,0], data[seed,1], c='r');


mean = data[nn[seed]].mean(axis=0)
cov = np.array([[1.,0],[0,1.]])
# cov = np.cov(data[nn[seed]].T)

x = np.arange(-15, 15, 0.1)
y = np.arange(-15, 15, 0.1)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, cov[0,0], cov[1,1], mean[0], mean[1], cov[1,0])
plt.contour(X, Y, Z, levels=[.22*Z.max()], colors='g');

data_prob = multivariate_normal.pdf(data, mean=mean, cov=cov);

# <codecell>

weights = np.ones((n_data))

# <codecell>

null = 1./n_data*np.ones((n_data,))

# <codecell>

mean

# <codecell>

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

weights = null/data_prob
plt.scatter(data[:,0], data[:,1], s=20*np.tanh(weights), cmap=plt.cm.binary_r)
plt.show()

# <codecell>

matched = (data_prob > null).nonzero()
plt.scatter(data[matched,0], data[matched,1], c=ratio[matched], cmap=plt.cm.binary_r)
plt.show()

