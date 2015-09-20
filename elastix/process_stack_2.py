#! /usr/bin/env python

import os
import sys
import time
from preprocess_utility import run_distributed3	


stack = sys.argv[1]
input_dir = sys.argv[2]
first_sec = int(sys.argv[3])
last_sec = int(sys.argv[4])
x,y,w,h = map(int, sys.argv[5:9])

DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'

d = {'script_dir': os.path.join(os.environ['GORDON_REPO_DIR'], 'elastix'),
     'stack': stack,
     'first_sec': first_sec,
     'last_sec': last_sec,

     'input_dir': input_dir,
     'renamed_dir': os.path.join(DATAPROC_DIR, stack+'_renamed'),
	 'padded_dir': os.path.join(DATAPROC_DIR, stack+'_lossless_padded'),
	 'warped_dir': os.path.join(DATAPROC_DIR, stack+'_lossless_warped'),
	 'suffix': 'lossless',

	 'cropped_dir': os.path.join(DATAPROC_DIR, stack+'_lossless_cropped'),
    }


# tmp_dir = DATAPROC_DIR + '/tmp'
# if not os.path.exists(tmp_dir):
# 	os.makedirs(tmp_dir)

# filenames = os.listdir(input_dir)
# tuple_sorted = sorted([(int(fn[:-4].split('_')[-1]), fn) for fn in filenames if fn.endswith('tif')])
# indices_sorted, fn_sorted = zip(*tuple_sorted)

# fn_correctInd_tuples = zip(fn_sorted, range(1, len(indices_sorted)+1))

# n_host = 16
# last_sec = len(indices_sorted)
# first_sec = 1
# secs_per_job = (last_sec - first_sec + 1)/float(n_host)
# first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]
# filenames_per_node = [fn_correctInd_tuples[f-1:l] for f,l in first_last_tuples]

# for i, args in enumerate(filenames_per_node):
# 	with open(tmp_dir + '/' + stack + '_tmp%d'%i, 'w') as f:
# 		f.write('\n'.join([fn + ' ' + str(ind) for fn, ind in args]))

# t = time.time()
# print 'renaming and expanding...',
# run_distributed3('%(script_dir)s/rename.py'%d,
#                 [(stack, input_dir, d['renamed_dir'], tmp_dir+'/' + stack + '_tmp%d'%i, d['suffix']) for i in range(n_host)],
#                 stdout=open('/tmp/log', 'ab+'))
# print 'done in', time.time() - t, 'seconds'


# t = time.time()
# print 'trim and fill lossless...',
# # os.system("cat " + os.path.join(tmp_dir, stack + '_bgColor_argsfile') + """| parallel -q --colsep ' ' -j 1 --filter-hosts -S %(all_servers_str)s convert %(output_dir)s/{1}_thumbnail.tif -trim -fuzz 4%% -fill "rgb({2},{3},{4})" -opaque white -compress lzw %(output_dir)s/{1}_thumbnail.tif"""%d)
# run_distributed3('%(script_dir)s/trim_fill.py'%d,
#                 [(stack, d['renamed_dir'], d['renamed_dir'], tmp_dir+'/'+stack + '_bgColor_argsfile_%d'%i,  d['suffix']) for i in range(n_host)],
#                 stdout=open('/tmp/log', 'ab+'))
# print 'done in', time.time() - t, 'seconds'


t = time.time()
print 'padding...',
os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/pad_IM.py %(stack)s %(renamed_dir)s %(padded_dir)s %(first_sec)d %(last_sec)d"%d + " lossless")
print 'done in', time.time() - t, 'seconds'

# t = time.time()
# print 'warping...',
# run_distributed3('%(script_dir)s/warp_IM.py'%d, 
# 				[(stack, d['padded_dir'], d['warped_dir'], f, l, d['suffix']) for f, l in first_last_tuples],
# 				stdout=open('/tmp/log', 'ab+'))
# print 'done in', time.time() - t, 'seconds'


# t = time.time()
# print 'warping...',
# run_distributed3('%(script_dir)s/crop_IM.py'%d, 
# 				[(stack, d['padded_dir'], d['cropped_dir'], f, l, d['suffix'], x, y, w, h) for f, l in first_last_tuples],
# 				stdout=open('/tmp/log', 'ab+'))
# print 'done in', time.time() - t, 'seconds'