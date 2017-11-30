#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='phoenixAtariAI',
    version='0.0.1',
    description='Phoenix Game on Atari 2600 based on RAM state',
    author='Riccardo, Alberto',
    author_email='phoenixstanford2017@gmail.com',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.4',
        'atari_py>=0.1.1',
        'Pillow',
        'PyOpenGL',
        'Keras>=2.0.9',
        'tensorflow>=1.4.0'
    ],
)



