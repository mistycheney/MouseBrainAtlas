#! /usr/bin/env python

import os
import sys

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], '/notebooks'))
from utilities2014 import execute_command

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

d = {'all_sections_str': ' '.join(map(str, range(first_sec, last_sec+1))),
     'all_sections_str2': ' '.join(map(str, range(first_sec+1, last_sec+1))),
     'all_servers_str': ','.join(['gcn-20-%d.sdsc.edu'%i for i in range(31,39)+range(41,49)]),
     'script_dir': '/home/yuncong/csd395/Brain/elastix',
     'stack': stack,
     'first_sec': first_sec,
     'last_sec': last_sec
    }

execute_command("ssh gcn-20-33.sdsc.edu %(script_dir)s/pad_thumbnails.py %(stack)s %(first_sec)s %(last_sec)s"%d)

execute_command("parallel -j 1 --filter-hosts -S %(all_servers_str)s %(script_dir)s/elastix_consecutive.py %(stack)s ::: %(all_sections_str2)s"%d)

execute_command("ssh gcn-20-33.sdsc.edu %(script_dir)s/elastix_consecutive2.py %(stack)s %(first_sec)s %(last_sec)s"%d)