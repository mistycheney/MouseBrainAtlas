Brainstem project
=================

Some useful code for the brainstem project. Parts of this project will be incorporated into [this project](https://github.com/mistycheney/registration).

Right now there are three main modules. The first reads images, thresholds them, and clusters objects found by thresholding. It also optionally caches files as TIFs, which take up more space than JPEG2000 but are faster to read. Here is an example of how to use it:

```python
import brainstem as b
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as pp

# get a list of all available image files
filenames = b.get_filenames()

# get a cropped version of the first few files
samples = b.sample_many(filenames[:5])

# segment the cells in each image
segmented, n_objects = zip(*list(b.segment_cells(i) for i in samples))

# cluster the resulting objects
X = b.all_object_features(segmented)
model = MiniBatchKMeans()
model.fit(X)

# visualize results, projecting onto first two principal components
pca = PCA(2)
Xp = pca.fit_transform(X)
pp.figure()
pp.scatter(Xp[:, 0], Xp[:, 1], c=model.labels_, s=30)

# visualize clusters in the first image
label_vecs = b.split_labels(segmented, model.labels_, n_objects=n_objects)
clustered_img = b.assign_clusters(segmented[0], label_vecs[0], rgb=True)
pp.figure()
pp.imshow(clustered_img)

````

You will have to modify ``brainstem.DATADIR`` to point to a directory containing jp2 files. If you do not want the caching functionality, set ``brainstem.USE_CACHE = False``.

The second module implements shape context descriptors, shape distance metrics via dynamic programming, and context-sensitive shape similarity via graph transduction. Before using it, it is necessary to build the Cython modules in place:

    python setup.py build_ext --inplace

Here is an example of how to use it:

```python
import skimage.data
import shape_context as sc

# get a binary horse shape
img = skimage.data.horse()
img = skimage.color.rgb2gray(img)
t = skimage.filter.threshold_otsu(img)
binary_img = img < t

# make a copy and cut off its legs
legless = binary_img.copy()
legless[200:, 200:] = 0

# compute the distance between them
sc.full_shape_distance(binary_img, legless)

```

Finally, the third module, "texture.py", implementes filtering with Gabor filters, filter visualization, and texture segmentation.