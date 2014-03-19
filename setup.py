from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import os


def readme():
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except (IOError, ImportError), e:
        return open('README.md').read()

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('compiledtrees')
    return config

metadata = {
    "name": 'sklearn-compiledtrees',
    "version": '1.0',
    "description": 'Compiled scikit-learn decision trees for faster evaluation',
    "author": 'Andrew Tulloch',
    "author_email": 'andrew@tullo.ch',
    "url": 'sklearn-compiledtrees',
    "packages": ['sklearn-compiledtrees'],
    "license": 'BSD License',
    "platforms": 'Any',
    "configuration": configuration,
    "long_description": readme(),
}

setup(**metadata)
