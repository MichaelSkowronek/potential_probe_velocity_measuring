# -*- coding: utf-8 -*-

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ppvm_py',
    version='0.0.1',
    author='Michael Skowronek',
    author_email='michael.skowronek.91@gmail.com',
    maintainer='Michael Skowronek',
    maintainer_email='michael.skowronek.91@gmail.com',
    description="Python package for potential_probe_velocity_measuring.",
    long_description="Python package for potential_probe_velocity_measuring.",
    install_requires=requirements,
    python_requires=">=3.12",
    package_dir = {
        'ppvm_py': 'src/ppvm_py',
    }
)