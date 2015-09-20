#!/usr/bin/env python

import os
import sys
from preprocess_utility import run_distributed3	
from subprocess import check_output
import time
import numpy as np


stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])


n_host = 16
secs_per_job = (last_sec - first_sec + 1)/float(n_host)
first_last_tuples = [(int(first_sec+i*secs_per_job), int(first_sec+(i+1)*secs_per_job-1) if i != n_host - 1 else last_sec) for i in range(n_host)]

d = {
     'script_dir': os.path.join(os.environ['GORDON_REPO_DIR'], 'distributed'),
     'stack': stack,
     'first_sec': first_sec,
     'last_sec': last_sec,
    }

t = time.time()
print 'computing features...'

run_distributed3('%(script_dir)s/analyze_CSHL_data_1.py'%d, 
                [(stack, f, l) for f, l in first_last_tuples],
                stdout=open('/tmp/log', 'ab+'))

print 'done in', time.time() - t, 'seconds'
