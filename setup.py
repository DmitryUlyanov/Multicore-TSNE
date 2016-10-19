import numpy
from setuptools import setup
from distutils.command.install import install as DistutilsInstall
import sys
import os

'''
    Just because I wanted .so file to be built same way for python and torch
    we exec cmake from cmd here.
'''


class MyInstall(DistutilsInstall):
    def run(self):
        os.system('mkdir -p multicore_tsne/release ; rm -r multicore_tsne/release/* ; cd multicore_tsne/release ; cmake -DCMAKE_BUILD_TYPE=RELEASE .. ; make VERBOSE=1')
        os.system(
            'cp multicore_tsne/release/libtsne_multicore.so python/libtsne_multicore.so')
        DistutilsInstall.run(self)


setup(
    name="MulticoreTSNE",
    version="0.1",
    description='Multicore version of t-SNE algorithm.',
    author="Dmitry Ulyanov (based on L. Van der Maaten's code)",
    author_email='dmitry.ulyanov.msu@gmail.com',
    url='https://github.com/DmitryUlyanov/Multicore-TSNE',
    install_requires=[
        'numpy',
        'psutil',
        'cffi',
    ],

    packages=['MulticoreTSNE'],
    package_dir={'MulticoreTSNE': 'python'},
    package_data={'MulticoreTSNE': ['multicore_tsne.so']},
    include_package_data=True,

    cmdclass={"install": MyInstall},
)
