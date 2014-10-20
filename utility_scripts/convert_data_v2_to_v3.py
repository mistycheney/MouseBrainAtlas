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

old_data_dir = sys.argv[1]
new_data_dir = sys.argv[2]

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
		if not os.path.exists(new_path+'/labelings'):
			os.makedirs(new_path+'/labelings')

		for paramSet in folders: # paramSets

			# print internal_segments
			# print path, new_path, folders
			# print paramSet

			print 'cp', path+'/'+image_name+'.tif', new_path+'/'+image_name+'.tif'
			shutil.copyfile(path+'/'+image_name+'.tif', new_path+'/'+image_name+'.tif')
			
			print 'cp', path+'/'+paramSet+'/pipelineResults', new_path+'/'+paramSet+'_pipelineResults'
			shutil.copytree(path+'/'+paramSet+'/pipelineResults', new_path+'/'+paramSet+'_pipelineResults')
			

			if os.path.exists(path+'/'+paramSet+'/labelings'):
				for f in os.listdir(path+'/'+paramSet+'/labelings'):
					new_labeling_name = '_'.join([x for x in f.split('_') if x != paramSet])
					print 'cp', path+'/'+paramSet+'/labelings/'+f, new_path+'/labelings/'+new_labeling_name
					shutil.copyfile(path+'/'+paramSet+'/labelings/'+f, new_path+'/labelings/'+new_labeling_name)