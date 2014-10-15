import sys
import os
import re

os.chdir(sys.argv[1])

stack = None
resolution = None
slice = None
params = None

rename_dict = {'img_cropped': 'cropImg',
'sp_dir_hist_normalized' : 'dirHist',
'sp_texton_hist_normalized': 'texHist',
'textonmap': 'texMap'}


for fn in os.listdir('.'):
    res = re.findall('(.*?)_(.*?)_(.*?)_param_(.*?)_(.*)\.(.*)', fn)
    #res = re.findall('(.*?)_(.*?)_(.*?)_(.*?)_(.*)\.(.*)', fn)
    if len(res) == 0: continue
    stack, resolution, slice, params, objname, ext = res[0]
    if objname in rename_dict:
        new_objname = rename_dict[objname]
    else:
        new_objname = objname
    
    new_name = '_'.join([stack, resolution, slice, params, new_objname]) + '.' + ext
    print fn, '->', new_name
    os.rename(fn, new_name)
