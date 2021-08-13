from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import eigency


__eigen_dir__ = '/usr/local/include/eigen3'
INCLUDE_DIRS = ['.', '../../baysis', '/usr/local/include', __eigen_dir__] + eigency.get_includes()
LIBRARY_DIRS = ['/usr/local/lib']
args = ['-stdlib=libc++', '-std=c++17']#, '-fopenmp']
CYTHON_DIRECTIVES = {"embedsignature": True}

extensions = [
    Extension("eigency.conversions", ["eigency/conversions.pyx"],
              include_dirs = INCLUDE_DIRS,
              language="c++17"
              ),
    Extension("eigency.core", ["eigency/core.pyx"],
              include_dirs = INCLUDE_DIRS,
              language="c++17"
              ),
    Extension('PyBaysis',
              sources=['pyapi.pyx'],
              language='c++17',
              include_dirs=INCLUDE_DIRS,
              library_dirs=LIBRARY_DIRS,
              extra_compile_args=args,
              # extra_link_args=['-lomp'],
              define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
]

setup(ext_modules=cythonize(extensions, annotate=True))