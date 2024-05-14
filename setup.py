from distutils.core import setup, Extension
import os

import numpy as np

# Configure extension
libraries = []
if os.name == 'posix':
    libraries.append('m')

compiledtrees_ext = Extension(
    'compiledtrees._compiled',
    ['compiledtrees/_compiled.c'],
    include_dirs=[np.get_include()],
    libraries=libraries,
    extra_link_args=["-Wl,--allow-multiple-definition"],
    extra_compile_args=["-O3", "-Wno-unused-function"]
)

setup(
    name='sklearn-compiledtrees',
    version='1.4',
    author='Andrew Tulloch',
    author_email='andrew@tullo.ch',
    maintainer='Andrew Tulloch',
    maintainer_email='andrew@tullo.ch',
    url='https://github.com/ajtulloch/sklearn-compiledtrees',
    description='Compiled scikit-learn decision trees for faster evaluation',
    packages=['compiledtrees'],
    ext_modules=[compiledtrees_ext],
    install_requires=['joblib', 'scikit-learn', 'six', 'numpy'],
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
