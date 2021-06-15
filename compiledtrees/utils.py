import os
import platform
import tempfile
import numpy as np

QUASI_FLOAT = 1000000


def convert_to_quasi_float(f):
    return np.int32(f * QUASI_FLOAT)


def convert_from_quasi_float(q):
    return float(q) / QUASI_FLOAT


class TemporaryFileFactory(object):
    def __init__(self, scratch_dir=None, delete_files=False):
        self.__mode = 'w+b'
        self.__scratch_dir = scratch_dir
        self.__delete_files = delete_files

    def get_file(self, prefix='compiledtrees_', suffix=None):
        return tempfile.NamedTemporaryFile(
            prefix=prefix,
            suffix=suffix,
            mode=self.__mode,
            dir=self.__scratch_dir,
            delete=self.__delete_files
        )


def get_temp_file_factory():
    scratch_dir = os.environ.get('COMPILED_TREES_SCRATCH')  # default is /tmp
    if scratch_dir and not os.path.exists(scratch_dir):
        raise Exception('Scratch directory {scratch_dir} does not exist'.format(scratch_dir=scratch_dir))
    delete_files = platform.system() != 'Windows'

    return TemporaryFileFactory(scratch_dir, delete_files)


def get_opt_level():
    opt_level = os.environ.get('COMPILED_TREES_OPT_LEVEL', '-O3')
    if opt_level not in ('-O0', '-O1', '-O2', '-O3', '-Os'):
        raise Exception('Invalid GCC optimization level: {opt_level}'.format(opt_level=opt_level))

    return opt_level


temp_file_factory = get_temp_file_factory()

gcc_opt_level = get_opt_level()

