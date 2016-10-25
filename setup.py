#! /usr/bin/env python

from setuptools import setup

setup(
    name='sco',
    version='0.1.0',
    description='Library for the prediction of cortical responses to visual stimuli: the Standard Cortical Observer',
    keywords='neuroscience vision cortex ',
    author='Noah C. Benson',
    author_email='nben@nyu.edu',
    url='https://github.com/noahbenson/sco/',
    license='GPLv3',
    packages=['sco',
              'sco.anatomy',
              'sco.contrast',
              'sco.stimulus',
              'sco.pRF',
              'sco.normalization',
              'sco.model_comparison'],
    package_data={'': ['LICENSE.txt']},
    install_requires=['numpy>=1.2',
                      'scipy>=0.7',
                      #'scikit-image>=0.12',
                      'nibabel>=2.0',
                      'pysistence>=0.4',
                      'decorator>=4.0',
                      'neuropythy>=0.1'])
