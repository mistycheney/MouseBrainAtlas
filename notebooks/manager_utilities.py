# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, json
from IPython.display import FileLink, Image, FileLinks
from pprint import pprint
import cv2
import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import gray2rgb

# <codecell>

def load_array(suffix, img_name, param_id, output_dir):
    result_name = img_name + '_param' + str(param_id)
    arr_file = os.path.join(output_dir, result_name, '%s_%s.npy'%(result_name, suffix))
    arr = np.load(arr_file)
    print 'load %s' % (arr_file)
    return arr

def save_array(arr, suffix, img_name, param_id, cache_dir='scratch'):
    result_name = img_name + '_param' + str(param_id)
    arr_file = os.path.join(cache_dir, result_name, '%s_%s.npy'%(result_name, suffix))
#     if not os.path.exists(arr_file):
    np.save(arr_file, arr)
    print '%s saved to %s' % (suffix, arr_file)
#     else:
#         print '%s already exists' % (arr_file)
        
def regulate_images(imgs):
    return np.array(map(regulate_img, imgs))
        
def regulate_img(img):
    if not np.issubsctype(img, np.uint8):
        try:
            img = img_as_ubyte(img)
        except:
            img_norm = (img-img.min()).astype(np.float)/(img.max() - img.min())    
            img = img_as_ubyte(img_norm)
            
    if img.ndim == 2:
        img = gray2rgb(img)
    
    return img
        
def save_img(img, suffix, img_name, param_id, 
             cache_dir='scratch', overwrite=True):
    '''
    img is in uint8 type or float type
    '''
    img = regulate_img(img)
        
    img_fn = get_img_filename(suffix, img_name, param_id, cache_dir, ext='tif')
    if not os.path.exists(img_fn) or overwrite:
        cv2.imwrite(img_fn, img)
        print '%s saved to %s' % (suffix, img_fn)
    else:
        print '%s already exists' % (img_fn)
        
    img_fn = get_img_filename(suffix, img_name, param_id, cache_dir, ext='png')
    if not os.path.exists(img_fn) or overwrite:
        cv2.imwrite(img_fn, img)
        print '%s saved to %s' % (suffix, img_fn)
    else:
        print '%s already exists' % (img_fn)

def get_img_filename(suffix, img_name, param_id, cache_dir='scratch', ext='tif'):
    result_name = img_name + '_param' + str(param_id)
    img_fn = os.path.join(cache_dir, result_name, '%s_%s.%s'%(result_name, suffix, ext))
    return img_fn

# <codecell>

def load_parameters(params_file, dump_dir, redownload=False):

    import csv
    
    if redownload:
        import gspread
        import getpass
        
        username = "cyc3700@gmail.com"
        password = getpass.getpass()

        docid = "1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE"

        client = gspread.login(username, password)
        spreadsheet = client.open_by_key(docid)
        for i, worksheet in enumerate(spreadsheet.worksheets()):
            with open(params_file, 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(worksheet.get_all_values())

    parameters = dict([])
    with open(params_file, 'r') as f:
        param_reader = csv.DictReader(f)
        for param in param_reader:
            for k in param.iterkeys():
                if param[k] != '':
                    try:
                        param[k] = int(param[k])
                    except ValueError:
                        param[k] = float(param[k])
            if param['param_id'] == 0:
                default_param = param
            else:
                for k, v in param.iteritems():
                    if v == '':
                        param[k] = default_param[k]
            parameters[param['param_id']] = param
        
            param_file = os.path.join(dump_dir, 'param%s.json'%param['param_id'])
            json.dump(param, open(param_file, 'w'))
            
    return parameters

