from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import os


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

setup(
    name='sklearn-compiledtrees',
    version='1.2',
    author='Andrew Tulloch',
    author_email='andrew@tullo.ch',
    maintainer='Andrew Tulloch',
    maintainer_email='andrew@tullo.ch',
    url='https://github.com/ajtulloch/sklearn-compiledtrees',
    configuration=configuration,
    description='Compiled scikit-learn decision trees for faster evaluation',
    packages=['compiledtrees'],
    license='MIT License',
    platforms='Any',
    long_description=open('README.rst').read(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',])
