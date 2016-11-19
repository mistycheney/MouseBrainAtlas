import os
import sys
import time
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from preprocess_utility import *
from data_manager import *
from metadata import *
# Align
all_stacks = ['MD602', 'MD603']
t = time.time()
sys.stderr.write('align all subjects to atlas ...')
exclude_nodes = [33]

run_distributed_aws(command='%(script_path)s %%(stack)s 1 1 atlasV2' % \
                {'script_path': os.path.join(os.environ['REPO_DIR'], 'registration') + '/align_subject_brain_to_atlas_v2.py'},
                kwargs_list=dict(stack=all_stacks),
                exclude_nodes=exclude_nodes,
                argument_type='single')

sys.stderr.write('done in %f seconds\n' % (time.time() - t))  # 526 seconds
