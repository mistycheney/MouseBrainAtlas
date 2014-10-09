import os 

def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

def get_local_data_structure(data_dir):
    local_structure = get_directory_structure(data_dir)
    
    local_data = {}
    for stack, stack_content in local_structure.values()[0].iteritems():
        if stack_content is None: continue
	local_data[stack] = {}
        for resol, resol_content in stack_content.iteritems():
            local_data[stack][resol] = {}
            for slice, slice_content in resol_content.iteritems():
                local_data[stack][resol][slice] = {}
                for param, param_content in slice_content.iteritems():
                    if param_content is not None and 'labeling' in param_content:
                        labelings = ['_'.join(labeling[:-4].split('_')[-2:]) for labeling in param_content['labelings'].keys() 
                                if labeling.endswith('.pkl')]
                        local_data[stack][resol][slice][param] = labelings

    return local_data

import sys
import cPickle as pickle
# dir_dict = get_local_data_structure(sys.argv[1])
dir_dict = get_directory_structure(sys.argv[1])
pickle.dump(dir_dict, open('remote_directory_structure.pkl', 'w'))
