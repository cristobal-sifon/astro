from __future__ import absolute_import, print_function

from setuptools import setup

# download from https://github.com/cristobal-sifon/myhelpers
from myhelpers.setup_helpers import *


setup(
    name='astro',
    version=find_version('astro/__init__.py'),
    description='Custom astronomical and astrophysical utilities',
    long_description=read('README.md'),
    author='Cristobal Sifon',
    author_email='sifon@astro.princeton.edu',
    url='https://github.com/cristobal-sifon/astro',
    packages=['astro', 'astro.clusters', 'astro.footprint', 'astro.ggl'],
    include_package_data=True,
    zip_safe=False
    )
