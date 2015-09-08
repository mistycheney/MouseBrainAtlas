import texture as t
import brainstem as b
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as pp

names = b.get_filenames()
img = b.get_cutout(names[0])
img = b.make_grey(img)
freqs = t.get_freqs(img)
thetas = np.deg2rad([0, 45, 90, 135])
kernels, all_freqs = t.make_filter_bank(freqs[-5:], thetas)
kernels = list(np.real(k) for k in kernels)
filtered, all_freqs = t.filter_image(img, kernels, all_freqs, select=False)
features = t.compute_features(filtered, all_freqs)
feats_coords = t.add_coordinates(features, 1.5)
model = MiniBatchKMeans(6)
model.fit(feats_coords.reshape(-1, feats_coords.shape[-1]))
pp.imshow(model.labels_.reshape(img.shape))
