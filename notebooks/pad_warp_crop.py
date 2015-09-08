import sys
import os
from subprocess import check_output
from skimage.util import pad
from skimage.io import imread, imsave
import numpy as np
import cPickle as pickle
import cv2
import time

from joblib import Parallel, delayed

input_dir = '/home/yuncong/csd395/CSHL_data/MD579_renamed'
filenames = os.listdir(input_dir)
suffix = '_lossless.tif'
# suffix = '_thumbnail.tif'
all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) 
                         for img_fn in filenames if img_fn.endswith(suffix)]))


def identify_shape(img_fn):
    return map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(input_dir, img_fn), 
                                 shell=True).split('x'))

img_shapes_arr = np.array(Parallel(n_jobs=16)(delayed(identify_shape)(img_fn) for img_fn in all_files.itervalues()))
max_width = img_shapes_arr[:,0].max()
max_height = img_shapes_arr[:,1].max()

margin = 0

canvas_width = max_width + 2 * margin
canvas_height = max_height + 2 * margin

padded_dir = '/home/yuncong/csd395/CSHL_data/MD579_padded'
if not os.path.exists(padded_dir):
    os.makedirs(padded_dir)

warped_dir = '/home/yuncong/csd395/CSHL_data/MD579_warped'
if not os.path.exists(warped_dir):
    os.makedirs(warped_dir)
    
cropped_dir = '/home/yuncong/csd395/CSHL_data/MD579_cropped'
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)
    
shrinked_dir = '/home/yuncong/csd395/CSHL_data/MD579_shrinked'
if not os.path.exists(shrinked_dir):
    os.makedirs(shrinked_dir)
    
with open('/home/yuncong/csd395/MD579_finalTransfParams.pkl', 'r') as f:
    Ts = pickle.load(f)
    
# the section that correponds to Ts[0]
init_sec = 59

# the resolution of lossless is "scale factor" times the resolution on which transform parameters are obtained
scale_factor = 32
    
Ts_lossless = {}
for sec, T in Ts.iteritems():
    T_lossless = T.copy()
    T_lossless[:2, 2] = T[:2, 2] * scale_factor
    Ts_lossless[sec + init_sec] = T_lossless
    
bg_color = (230,232,235)


from subprocess import check_output, call

def execute_command(cmd):
    print cmd

    try:
        retcode = call(cmd, shell=True)
        if retcode < 0:
            print >>sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >>sys.stderr, "Child returned", retcode
    except OSError as e:
        print >>sys.stderr, "Execution failed:", e
        raise e
        
def pad_and_warp_and_crop(img_fn, T, input_dir=input_dir, padded_dir=padded_dir, 
                                      warped_dir=warped_dir, cropped_dir=cropped_dir,
                          ext='tif'):
    
    img_fn = os.path.basename(img_fn)
    
    secind = int(img_fn[:-4].split('_')[1])
    
    warped_fn = img_fn[:-4] + '_padded_warped.' + ext
    padded_fn = img_fn[:-4] + '_padded.' + ext
    
    d = {'input_fn': os.path.join(input_dir, img_fn),
        'output_fn': os.path.join(padded_dir, padded_fn),
         'bg_r': bg_color[0],
         'bg_g': bg_color[1],
         'bg_b': bg_color[2],
         'width': canvas_width,
         'height': canvas_height,
         'offset_x': '+0',
         'offset_y': '+0'
         }
        
    convert_cmd = 'convert %(input_fn)s -background "rgb(%(bg_r)d,%(bg_g)d,%(bg_b)d)" -gravity center -geometry %(offset_x)s%(offset_y)s -extent %(width)dx%(height)d -compress lzw %(output_fn)s'%d
    execute_command(convert_cmd)
    
    # special treatment for MD579
    if secind in range(59, 67):
        d = {
            'input_fn': os.path.join(padded_dir, padded_fn),
            'output_fn': os.path.join(padded_dir, padded_fn),
            'bg_r': bg_color[0],
            'bg_g': bg_color[1],
            'bg_b': bg_color[2],
            'offset_y': '-' + str(300*scale_factor)    
        }
        execute_command('convert -page +0%(offset_y)s %(input_fn)s -background "rgb(%(bg_r)d,%(bg_g)d,%(bg_b)d)" -flatten %(output_fn)s'%d)

    d = {'sx':T[0,0],
         'sy':T[1,1],
         'rx':T[1,0],
         'ry':T[0,1],
         'tx':T[0,2],
         'ty':T[1,2],
         'input_fn': os.path.join(padded_dir, padded_fn),
         'output_fn': os.path.join(warped_dir, warped_fn)
        }

    affine_cmd = "convert %(input_fn)s -distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' %(output_fn)s"%d
    execute_command(affine_cmd)
    
    cropped_fn = img_fn[:-4] + '_padded_warped_cropped.' + ext

    d = {
    'x': 576 * scale_factor,
    'y': 413 * scale_factor,
    'w': 403 * scale_factor,
    'h': 280 * scale_factor,
    'input_fn': os.path.join(warped_dir, warped_fn),
    'output_fn': os.path.join(cropped_dir, cropped_fn)
    }

    crop_cmd = 'convert %(input_fn)s -crop %(w)dx%(h)d+%(x)d+%(y)d %(output_fn)s'%d
    execute_command(crop_cmd)
    
    d = {
        'input_fn': os.path.join(warped_dir, warped_fn),
        'shrinked_fn': os.path.join(shrinked_dir, img_fn[:-4] + '_padded_warped_shrinked.' + ext)
    }
        
    shrink_cmd = 'convert %(input_fn)s -scale 12.5%% %(shrinked_fn)s'%d
    execute_command(shrink_cmd)
    
    
secind = int(sys.argv[1])
# t = time.time()
pad_and_warp_and_crop(all_files[secind], np.linalg.inv(Ts_lossless[secind]))
# print time.time()-t
    
    