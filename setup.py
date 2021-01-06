# -*- coding: utf-8 -*-
"""
Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
Created on: Nov 23, 2020
"""

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='cBrainMRIPrePro',
    version='0.0.1',
    description="Conventional Brain Magnetic Resonance Images Pre-Processing for NIfTI files",
    long_description=readme,
    author='Alexandre CARRE',
    author_email='alexandre.carre@gustaveroussy.fr',
    url='https://github.com/Alxaline/cBrainMRIPrePro',
    license=license,
    packages=find_packages(exclude=['docs']),
    package_data={
        'cBrainMRIPrePro.utils.Atlas_SRI': ['*.nii.gz'],  # add template
    },
    python_requires='>=3.6',
    keywords="brain Conventional mri preprocessing",
)

setup(install_requires=['torch',
                        'simpleitk',
                        'scikit-image',
                        'numba',
                        'argparse',
                        'pdoc3',
                        'antspyx @ git+https://github.com/ANTsX/ANTsPy.git@v0.2.6#egg=antspyx',
                        'HD-BET @ git+https://github.com/MIC-DKFZ/HD-BET.git#egg=HD-BET',
                        ], **args)
