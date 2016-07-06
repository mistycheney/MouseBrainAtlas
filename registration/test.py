import numpy as np

import sys
import os

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from registration_utilities import A
a = A(int(1e4))
a.func_parallel()
