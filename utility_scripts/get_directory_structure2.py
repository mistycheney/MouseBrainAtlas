import os
import sys
import cPickle as pickle


def generate_json(data_dir):

    d = get_directory_structure(data_dir)

    q = []
    for stack_name, stack_content in d.values()[0].items():
        if stack_content is None or len(stack_content) == 0: continue
        sec_items = stack_content['x5'].items()
        stack_info = {'name': stack_name,
                    'available_res': stack_content.keys()
                    }

        sec_infos = []
        for sec_ind, sec_content in sec_items:
            if sec_content is None: continue
            # stack_info['available_sections'].append(int(sec_ind))
            sec_info = {'index': int(sec_ind)}
            if 'labelings' in sec_content.keys():
                sec_info['labelings'] = [k for k in sec_content['labelings'].keys() if k.endswith('pkl')]
            if 'pipelineResults' in sec_content.keys():
                sec_info['available_results'] = sec_content['pipelineResults'].keys()
            sec_infos.append(sec_info)
        sec_infos = sorted(sec_infos, key=lambda x: x['index'])

        sec_infos = sorted(sec_infos, key=lambda x: x['index'])
        stack_info['sections'] = sec_infos

        q.append(stack_info)

    return q


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

if __name__ == '__main__':

    dir_dict = generate_json(sys.argv[1])
    repo_dir = sys.argv[2]
    pickle.dump(dir_dict, open(repo_dir+'/remote_directory_structure.pkl', 'w'))