import sys
import shutil
import os
from os import path
from subprocess import call as execute

from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension


PACKAGE_NAME = "MulticoreTSNE"

VERSION = '0.1'


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        if 0 != os.system('cmake --version'):
            sys.exit('\nError: Cannot find cmake. Install cmake, e.g. `pip install cmake`.')
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        SOURCE_DIR = ext.sourcedir
        EXT_DIR = path.abspath(path.dirname(self.get_ext_fullpath(ext.name)))
        BUILD_TEMP = self.build_temp

        shutil.rmtree(BUILD_TEMP, ignore_errors=True)
        os.makedirs(BUILD_TEMP)

        # Run cmake
        if 0 != execute(['cmake',
                         '-DCMAKE_BUILD_TYPE={}'.format('Debug' if self.debug else 'Release'),
                         '-DCMAKE_VERBOSE_MAKEFILE={}'.format(int(self.verbose)),
                         "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY='{}'".format(EXT_DIR),
                         SOURCE_DIR], cwd=BUILD_TEMP):
            sys.exit('\nERROR: Cannot generate Makefile. See above errors.')

        # Run make
        if 0 != execute('cmake --build . -- -j4', shell=True, cwd=BUILD_TEMP):
            sys.exit('\nERROR: Cannot find make? See above errors.')


if __name__ == '__main__':
    EXT_MODULES = []
    if 'test' not in sys.argv:
        EXT_MODULES = [CMakeExtension('MulticoreTSNE.MulticoreTSNE',
                                      sourcedir='multicore_tsne')]
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description='Multicore version of t-SNE algorithm.',
        author="Dmitry Ulyanov (based on L. Van der Maaten's code)",
        author_email='dmitry.ulyanov.msu@gmail.com',
        url='https://github.com/DmitryUlyanov/Multicore-TSNE',
        install_requires=[
            'numpy',
            'psutil',
            'cffi'
        ],
        packages=find_packages(),
        include_package_data=True,

        ext_modules=EXT_MODULES,
        cmdclass={'build_ext': CMakeBuild},
    )
