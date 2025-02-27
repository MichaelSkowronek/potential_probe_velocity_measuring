# -*- coding: utf-8 -*-

from setuptools import setup

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
    package_dir = {
        'python_utils': 'src/python_utils',
    }
)