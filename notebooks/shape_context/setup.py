from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = "brainstem",
    ext_modules = cythonize('_shape_context.pyx'), # accepts a glob pattern
    include_dirs = [np.get_include()],
)
