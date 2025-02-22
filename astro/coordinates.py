from __future__ import absolute_import, division, print_function

import numpy
import sys
from astLib import astCoords
from astLib import astWCS
from astropy.io import fits
from numpy import arcsin, arctan, cos, sin

if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip

"""
Various coordinates utilities

Variables
---------
  epoch      : Epoch. Right now only 'J2000' is implemented
  aNGP       : Right Ascension of the North Galactic Pole, for the
               chosen epoch
  dNGP       : Declination of the North Galactic Pole, for the chosen
               epoch

"""

epoch = 'J2000'

if epoch == 'J2000':
    aNGP = astCoords.hms2decimal('12:51:26.28', ':')
    dNGP = astCoords.dms2decimal('+27:07:42.0', ':')


def torad(x):
    """
    Input: coordinate in degrees
    Returns : coordinate in radians
    """
    return numpy.pi * x / 180



def todeg(x):
    """
    Input: coordinate in radians
    Returns: coordinate in degrees
    """
    return 180 * x / numpy.pi


def eq2gal(ra, dec):
    """
    WARNING: the output galactic longitude has a ~4 arcmin difference
    with the NED output. Will correct later

    Convert from Equatorial to Galactic coordinates. Currently only
    implemented for epoch=J2000

    Parameters
    ----------
        ra        : float
                    Right Ascension in decimal degrees
        dec       : float
                    Declination in decimal degrees

    Returns
    -------
        b         : float
                    Galactic latitude, in decimal degrees
        l         : float
                    Galactic longitude, in decimal degrees

    """
    a0 = torad(aNGP)
    d0 = torad(dNGP)
    a = torad(ra)
    d = torad(dec)
    # Galactic latitude
    b = cos(d) * cos(d0) * cos(a - a0) + sin(d) * sin(d0)
    b = arcsin(b)
    # Galactic longitude
    l0 = torad(33)
    l = (sin(d) - sin(b) * sin(d0)) / (cos(d) * sin(a - a0) * cos(d0))
    l = arctan(l) + l0
    return todeg(l) + 180, todeg(b)


def zhelio(z, ra, dec):
    c = 299792.458
    # from NED
    Vapex = 232.3 # km/s
    rad = numpy.pi / 180
    lapex = 87.8 * rad
    bapex = 1.7 * rad
    gal = [astCoords.convertCoords('J2000', 'GALACTIC', x, y, 2000.0)
           for x, y in izip(ra, dec)]
    l, b = numpy.transpose(gal)
    l *= rad
    b *= rad
    Vh = numpy.sin(b) * numpy.sin(bapex) - \
         numpy.cos(b) * numpy.cos(bapex)*numpy.cos(l-lapex)
    Vh *= Vapex
    return (c * z - Vh) / c

