import argparse
import sys
import os
from subprocess import call

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Execute feature extraction pipeline',
epilog="""
The following command processes image RS141_x5_0001.tif using the specified parameters.
python %s RS141 1 -g blueNisslWide -s blueNisslRegular -v blueNissl
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("resolution", type=str, help="resolution string")
parser.add_argument("slice_ind", type=int, help="slice index")
parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

def execute_command(cmd):
	try:
	    retcode = call(cmd, shell=True)
	    if retcode < 0:
	        print >>sys.stderr, "Child was terminated by signal", -retcode
	    else:
	        print >>sys.stderr, "Child returned", retcode
	except OSError as e:
	    print >>sys.stderr, "Execution failed:", e
	    raise e

execute_command('source ../setup.sh')

print '======== gabor filtering ======='
execute_command('python gabor_filter.py %s %d -g %s -s %s -v %s' % (args.stack_name, args.slice_ind, args.gabor_params_id, args.segm_params_id, args.vq_params_id))

print '======== segmentation ======='
execute_command('python segmentation.py %s %d -g %s -s %s -v %s' % (args.stack_name, args.slice_ind, args.gabor_params_id, args.segm_params_id, args.vq_params_id))

print '======== rotate features ======='
execute_command('python rotate_features.py %s %d -g %s -s %s -v %s' % (args.stack_name, args.slice_ind, args.gabor_params_id, args.segm_params_id, args.vq_params_id))

print '======== generate textons ======='
execute_command('python generate_textons.py %s %d -g %s -s %s -v %s' % (args.stack_name, args.slice_ind, args.gabor_params_id, args.segm_params_id, args.vq_params_id))

print '======== assign textons ======='
execute_command('python assign_textons.py %s %d -g %s -s %s -v %s' % (args.stack_name, args.slice_ind, args.gabor_params_id, args.segm_params_id, args.vq_params_id))

print '======== compute_texton_histograms ======='
execute_command('python compute_texton_histograms.py %s %d -g %s -s %s -v %s' % (args.stack_name, args.slice_ind, args.gabor_params_id, args.segm_params_id, args.vq_params_id))

print '======== grow_regions ======='
execute_command('python grow_regions.py %s %d -g %s -s %s -v %s' % (args.stack_name, args.slice_ind, args.gabor_params_id, args.segm_params_id, args.vq_params_id))
