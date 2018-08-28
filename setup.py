"""Installation script for distorthsv module."""
import distutils.core
import os
import setuptools


# NOTE(brendan): E.g., DISTORTHSV_EXTRA_LIBS=omp
DISTORTHSV_EXTRA_LIBS = os.getenv('DISTORTHSV_EXTRA_LIBS')


distorthsv_module = distutils.core.Extension(
    '_distorthsv',
    extra_compile_args=['-fopenmp'],
    define_macros=[('MAJOR_VERSION', '0'), ('MINOR_VERSION', '0')],
    undef_macros=['NDEBUG'],
    libraries=[] if DISTORTHSV_EXTRA_LIBS is None
                 else DISTORTHSV_EXTRA_LIBS.split(','),
    sources=['distorthsv/distorthsvmodule.c'])

setuptools.setup(author='Brendan Duke',
                 author_email='brendanw.duke@gmail.com',
                 name='DistortHSV',
                 description='Image distortion package.',
                 long_description="""
                    Extension for HSV distortion, left-right image/video
                    flipping, and contrast distortion.
                """,
                 entry_points="""
                     [console_scripts]
                     distorthsv_test=distorthsv.test.distorthsv_test:distorthsv_test
                 """,
                 install_requires=['Click', 'matplotlib', 'numpy'],
                 ext_modules=[distorthsv_module],
                 packages=setuptools.find_packages(),
                 py_modules=['distorthsv.test.distorthsv_test'],
                 url='https://brendanduke.ca',
                 version='0.0')
