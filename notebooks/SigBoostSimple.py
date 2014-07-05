# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal

# <codecell>

cluster_data1, _ = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=.5, center_box=(5.0, 5.0), shuffle=True, random_state=None)
cluster_data2, _ = make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=.5, center_box=(-5.0, -5.0), shuffle=True, random_state=None)
cluster_data = np.r_[cluster_data1, cluster_data2]

bg_data, _ = make_blobs(n_samples=200, n_features=2, centers=1, cluster_std=5.0, center_box=(0., 0.), shuffle=True, random_state=None)

# bg_data = np.random.random(size=(200,2))
# bg_data = (bg_data-0.5)*20

data = np.r_[cluster_data, bg_data]
n_data = data.shape[0]

# <codecell>

scatter(data[:,0], data[:,1]);
plt.show()

scatter(bg_data[:,0], bg_data[:,1], c='b')
scatter(cluster_data[:,0], cluster_data[:,1], c='b')
centers = np.array([[5,5],[-5,-5]])

data_prob = multivariate_normal.pdf(data, mean=[0,0], cov=cov);
x = np.arange(-15, 15, 0.1)
y = np.arange(-15, 15, 0.1)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, 5, 5, 0, 0, 0)
plt.contour(X, Y, Z, levels=[.1*Z.max()], colors='g');


for m in range(2):
    data_prob = multivariate_normal.pdf(data, mean=centers[m], cov=cov);
    x = np.arange(-15, 15, 0.1)
    y = np.arange(-15, 15, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, cov[0,0], cov[1,1], centers[m,0], centers[m,1], cov[1,0])
    plt.contour(X, Y, Z, levels=[.1*Z.max()], colors='g');

plt.title('a large Gaussian + two small Gaussians')    
plt.show()

# <codecell>

mu0 = data.mean(axis=0)
cov0 = np.cov(data.T)

D = pdist(data)
D = squareform(D)
nn = D.argsort(axis=1)[:,:50]

n_models = 3

cov = np.array([[0.5, 0], [0, 0.5]])
weights = np.ones((n_data, ))/n_data
# models = np.zeros((5, 2))
model_idx = np.zeros((n_models, ), dtype=np.int)
scores = np.zeros((n_data, n_models))

null_probs = multivariate_normal.pdf(data, mean=mu0, cov=cov0)

probs = np.array([multivariate_normal.pdf(data, mean=x, cov=cov) for x in data])
for t in range(n_models):
#     ratio = np.empty((n_data, t+1, n_data))
#     for i in range(n_data):
#         ratio[i,:,:] = [np.log(probs[i]/probs[model_idx[tt]]) for tt in range(t+1)]

#     overall_score = np.zeros((n_data,))
#     for i in range(n_data):
#         overall_score[i] = np.sum(ratio[i].min(axis=0)[nn[i,:20]]*weights[nn[i,:20]])

    ratio = np.empty((n_data, n_data))
    for i in range(n_data):
        ratio[i,:] = np.log(probs[i]/null_probs)

    overall_score = np.zeros((n_data,))
    for i in range(n_data):
        overall_score[i] = np.sum(ratio[i][nn[i,:20]]*weights[nn[i,:20]])
    
    model_next = overall_score.argsort()[-1]    
    
    scatter(data[:,0], data[:,1], s=10000*np.tanh(weights), c=overall_score/overall_score.max(), cmap=plt.cm.binary_r);
    scatter(data[model_next,0], data[model_next,1], 
            s=10000*np.tanh(weights[model_next]), c='r');
    
    data_prob = multivariate_normal.pdf(data, mean=data[model_next], cov=cov);
    x = np.arange(-15, 15, 0.1)
    y = np.arange(-15, 15, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, cov[0,0], cov[1,1], data[model_next,0], data[model_next,1], cov[1,0])
    plt.contour(X, Y, Z, levels=[.1*Z.max()], colors='g');

    weights[nn[model_next, :20]] = weights[nn[model_next, :20]] * 1./ratio[model_next][nn[model_next, :20]]
    weights = weights/weights.sum()

    model_idx[t] = model_next
    plt.title('SigBoost, round %d' %(t+1))
    plt.show()
    
scatter(data[:,0], data[:,1]);
for m in model_idx:
    scatter(data[m,0], data[m,1], s=20, c='r');
    
    data_prob = multivariate_normal.pdf(data, mean=data[m], cov=cov);
    x = np.arange(-15, 15, 0.1)
    y = np.arange(-15, 15, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, cov[0,0], cov[1,1], data[m,0], data[m,1], cov[1,0])
    plt.contour(X, Y, Z, levels=[.1*Z.max()], colors='g');
plt.title('SigBoost')
plt.show()

# <codecell>

from sklearn.cluster import MiniBatchKMeans
n_clusters = 2
kmeans = MiniBatchKMeans(n_clusters)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

scatter(data[:,0], data[:,1], s=20, c='b');
for l in range(n_clusters):
    scatter(centroids[:,0], centroids[:,1], s=20, c='r');
    
    data_prob = multivariate_normal.pdf(data, mean=data[m], cov=cov);
    x = np.arange(-15, 15, 0.1)
    y = np.arange(-15, 15, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, cov[0,0], cov[1,1], centroids[l,0], centroids[l,1], cov[1,0])
    plt.contour(X, Y, Z, levels=[.1*Z.max()], colors='g');

plt.title('kmeans')

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

