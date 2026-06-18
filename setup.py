import os
import re
from setuptools import setup


def read_requirements(reqfile):
    return [i for i in open(reqfile).read().split("\n") if i]


def find_location(file):
    return os.path.abspath(os.path.dirname(file))


def find_version(filename):
    """Find package version

    The version should be defined as variable `__version__` (typically
    in an `__init__.py`). This way it only needs to be specified in
    one place.

    Copied from pip's setup.py,
    https://github.com/pypa/pip/blob/1.5.6/setup.py
    """
    version_file = read(filename)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read(filename):
    """Utility function to read the entire contents of a file

    Used mostly to pass the contents of a README file to
    `long_description` in the setup script.

    Copied from the Python docs
    """
    return open(filename).read()


setup(
    name="astro",
    version=find_version("astro/__init__.py"),
    description="Custom astronomical and astrophysical utilities",
    long_description=read("README.md"),
    author="Cristobal Sifon",
    author_email="sifon@astro.princeton.edu",
    url="https://github.com/cristobal-sifon/astro",
    packages=["astro", "astro.clusters", "astro.footprint", "astro.ggl"],
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    zip_safe=False,
)
