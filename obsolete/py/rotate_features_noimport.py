# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# try:
#     features_rotated = dm.load_pipeline_result('features_rotated', 'npy')

# except Exception as e:

features = dm.load_pipeline_result('features', 'npy')

valid_features = features[:, dm.mask].T
n_valid = len(valid_features)

del features

def rotate_features(fs):
    features_tabular = fs.reshape((fs.shape[0], dm.n_freq, dm.n_angle))
    max_angle_indices = features_tabular.max(axis=1).argmax(axis=-1)
    features_rotated = np.reshape([np.roll(features_tabular[i], -ai, axis=-1) 
                               for i, ai in enumerate(max_angle_indices)], (fs.shape[0], dm.n_freq * dm.n_angle))

    return features_rotated

from joblib import Parallel, delayed

n_splits = 1000
features_rotated_list = Parallel(n_jobs=16)(delayed(rotate_features)(fs) for fs in np.array_split(valid_features, n_splits))
features_rotated = np.vstack(features_rotated_list)

del valid_features

dm.save_pipeline_result(features_rotated, 'features_rotated', 'npy')

