import os
import platform
import subprocess
import sysconfig

import six
import numpy
from numpy.distutils.misc_util import Configuration

if platform.system() == 'Windows':
    CXX_COMPILER = os.environ['CXX'] if 'CXX' in os.environ else None
else:
    CXX_COMPILER = sysconfig.get_config_var('CXX')

# detect OpenMP support
if platform.system() == 'Darwin':
    c_ver = subprocess.check_output([CXX_COMPILER, '--version']).decode('ascii')
    if c_ver.find('clang') >= 0:  # Xcode clang does not support OpenMP
        OPENMP_SUPPORT = False
    else:  # GCC supports OpenMP
        OPENMP_SUPPORT = True
else:
    OPENMP_SUPPORT = True


def configuration(parent_package="", top_path=None):
    config = Configuration("compiledtrees", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    extra_link_args = []
    if OPENMP_SUPPORT:
        extra_link_args.append("-fopenmp")
    if six.PY2 and platform.system() == 'Windows':  # HACK: A bug with linking
        extra_link_args.append("-Wl,--allow-multiple-definition")

    extra_compile_args = ["-O3", "-Wno-unused-function"]
    if OPENMP_SUPPORT:
        extra_compile_args.append("-fopenmp")

    config.add_extension("_compiled",
                         sources=["_compiled.c"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_link_args=extra_link_args,
                         extra_compile_args=extra_compile_args,
                         )
    config.add_subpackage("tests")
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
