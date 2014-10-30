import sys
import os
from pprint import pprint
import cPickle as pickle
import numpy as np
import datetime
import shutil

# old_data_dir = '/home/yuncong/BrainLocal/DavidData_v2'
# new_data_dir = '/home/yuncong/BrainLocal/DavidData_v3'

# old_data_dir = '/oasis/projects/nsf/csd181/yuncong/DavidData2014v2'
# new_data_dir = '/oasis/projects/nsf/csd181/yuncong/DavidData2014v3'

old_data_dir = os.path.realpath(sys.argv[1])
new_data_dir = os.path.realpath(sys.argv[2])

# start fresh
if os.path.exists(new_data_dir):
	shutil.rmtree(new_data_dir)

external_nlevel = len(old_data_dir.split('/'))

if not os.path.exists(new_data_dir):
	os.makedirs(new_data_dir)

for path, folders, files in os.walk(old_data_dir):
	all_segments = path.split('/')
	nlevel = len(all_segments) - external_nlevel
	internal_segments = '/'.join(all_segments[external_nlevel:])
	image_name = '_'.join(all_segments[external_nlevel:])
	new_path = new_data_dir + '/' + internal_segments
	
	if nlevel == 3:
		
		try:
			os.makedirs(new_path)
		except:
			continue

		if not os.path.exists(path+'/'+image_name+'.tif'):
			continue

		print 'cp', path+'/'+image_name+'.tif', new_path+'/'+image_name+'.tif'
		shutil.copyfile(path+'/'+image_name+'.tif', new_path+'/'+image_name+'.tif')
			
		if not os.path.exists(path+'/'+image_name+'_mask.png'):
			continue

		print 'cp', path+'/'+image_name+'_mask.png', new_path+'/'+image_name+'_mask.png'
		shutil.copyfile(path+'/'+image_name+'_mask.png', new_path+'/'+image_name+'_mask.png')
