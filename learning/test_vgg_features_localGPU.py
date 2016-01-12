import os
import sys

from FeatureExtractor import *

model_dir = os.environ['MODEL_DIR']+'/vgg16'

fe = FeatureExtractor(model_dir, batch_size=1, ctx='gpu')

files = os.listdir('patches')

for f in files[30:31]:

	print f

	all_patches_arr = np.load('patches/'+f)[:10]
	print all_patches_arr.shape

	# MXNetError: [00:18:25] src/storage/./gpu_device_storage.h:39: Check failed: e == cudaSuccess || e == cudaErrorCudartUnloading CUDA: out of memory
	# https://github.com/dmlc/mxnet/issues/675
	# changing batch_size does not help
	# Restaring notebook, this error is no more !!!

	images = all_patches_arr.transpose(0, 3, 1, 2)
	print images.shape

	try:
		features = fe.extract(images)
		print features.shape
		np.save('patches/'+f[:-4]+'_features.npy', features)

	except Exception as e:

		print f, 'failed'
		print e