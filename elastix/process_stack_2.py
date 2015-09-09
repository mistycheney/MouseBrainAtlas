#! /usr/bin/env python

import os

# stack = 'MD579'
# first_sec = 59
# last_sec = 188

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

d = {'all_sections_str': ' '.join(map(str, range(first_sec, last_sec+1))),
     'all_servers_str': ','.join(['gcn-20-%d.sdsc.edu'%i for i in range(31,39)+range(41,49)]),
     'script_dir': '/home/yuncong/csd395/elastix',
     'stack': stack,
     'x': 576,
     'y': 413,
     'w': 403,
     'h': 280
    }

os.system("parallel -j 1 --filter-hosts -S %(all_servers_str)s %(script_dir)s/pad_warp_crop_lossless.py %(stack)s %(x)s %(y)s %(w)s %(h)s ::: %(all_sections_str)s"%d)