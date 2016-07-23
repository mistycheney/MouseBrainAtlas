import os
import sys
import time

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *

from learning_utilities import *

# Run once - Generate landmark indices and dump to files
for stack in ['MD589', 'MD594', 'MD585']:
    print stack

    t = time.time()
    sys.stderr.write('Generating landmark indices and dump to file ...')

    indices_allLandmarks_allSections = locate_annotated_patches(stack, force=True)
    fn = os.path.join(patch_rootdir, '%(stack)s_indices_allLandmarks_allSection.h5' % {'stack':stack})
    indices_allLandmarks_allSections.to_hdf(fn, 'framewise_indices')

    sys.stderr.write('done in %f seconds\n' % (time.time() - t))
