# -*- coding: utf-8 -*-
"""
Astronomical and astrophysical utilities

Contains various useful calculations in the framework of astronomy in
general and galaxy clusters in particular, plus many recurrent physical
and astrophysical constants, and unit conversions.

Some of these classes are superseded by various astropy modules (e.g.,
units), but others remain useful in their own right.

Classes:
    clusters -- Calculations relevant to clusters of galaxies (e.g. mass)
    cosmology -- Cosmological parameters of astronomical objects (e.g.
                 distances)
    constants -- Various physical and astrophysical constants in cgs
               units
    footprint -- Working with footprints
    photometry -- Calculations and utilities for photometric data
    units -- Unit conversions from cgs

    See each class for more help

"""

from . import *


__version__ = "0.6.0a3"
