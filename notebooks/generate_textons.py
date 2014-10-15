# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

parser = argparse.ArgumentParser(
formatter_class=argparse.RawDescriptionHelpFormatter,
description='Generate textons from a set of filtered images',
epilog="""%s
"""%(os.path.basename(sys.argv[0]), ))

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("resolution", type=str, help="resolution string")
parser.add_argument("begin_slice", type=str, help="slice number to begin, zero-padding to 4 digits")
parser.add_argument("end_slice", type=str, help="slice number to end, zero-padding to 4 digits")
parser.add_argument("param_id", type=str, help="parameter identification name")
args = parser.parse_args()

data_dir = '/oasis/projects/nsf/csd181/yuncong/DavidData2014v2'
repo_dir = '/oasis/projects/nsf/csd181/yuncong/Brain/'
params_dir = os.path.join(repo_dir, 'params')

class args:
    stack_name = 'RS141'
    resolution = 'x5'
    begin_slice = '0001'
    end_slice = '0005'
    param_id = 'redNissl'

