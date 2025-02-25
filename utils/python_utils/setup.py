# -*- coding: utf-8 -*-

import os
from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='python_utils',
    version='0.0.1',
    author='Michael Skowronek',
    author_email='michael.skowronek.91@gmail.com',
    maintainer='Michael Skowronek',
    maintainer_email='michael.skowronek.91@gmail.com',
    description="Python utils package.",
    long_description="Python utils package.",
    install_requires=requirements,
    python_requires=">=3.12",
    packages=find_packages(
        where='.',
        include=[
            'python_utils',
            # e.g. 'python_utils.example',
        ],
        exclude=[
            'python_utils.tests',
            'python_utils.example',
        ],
    ),
)