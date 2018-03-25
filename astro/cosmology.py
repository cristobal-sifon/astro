# -*- coding: utf-8 -*-
"""
a module for calculation of cosmological distances and related quantities

Contains functions to compute cosmological information of objects as a
function of redshift. See Hogg (1999) for a description of all these
quantities.

The default cosmological parameters are consistent with WMAP-7 data (e.g.
Larson et al. 2011, ApJS, 192, 16):
    h = 0.7
    Omega L = 0.7
    Omega M = 0.3
    Omega_k = 0.00

"""
from __future__ import absolute_import, division, print_function

import scipy
from scipy import integrate
#from astropy import constants
#from astropy import units as u

# local modules
#import constants
#import units
from . import constants, units


h = 0.7
Omega_L = 0.7
Omega_M = 1 - Omega_L
Omega_k = 0.


def E(z):
    """
    Evolution factor E(z) at a given redshift.
    """
    try:
        return (Omega_M * (1+z)**3 + Omega_k * (1+z)**2 + Omega_L) ** 0.5
    except AttributeError:
        from uncertainties import unumpy
        return unumpy.sqrt(Omega_M * (1+z)**3 + Omega_k * (1+z)**2 + Omega_L)

def H(z, unit='default'):
    """
    Hubble constant at a given redshift.
    The units can be km·s⁻¹·Mpc⁻¹ ('default') or s⁻¹ ('s')
    """
    if unit == 's':
        return 100 * h * E(z) * (units.km / units.Mpc)
    return 100 * h * E(z)

def tH(z, unit='Gyr'):
    """
    Hubble time at a given redshift.
    The units can be 's', 'yr' or 'Gyr' (default unit='Gyr').
    """
    f = lambda t: 1. / ((1. + t) * E(t))
    i = integrate.quad(f, z, scipy.inf)[0]
    if unit == 'Gyr':
        return 1 / H(0, unit='s') * i / units.Gyr
    if unit == 'yr':
        return 1 / H(0, unit='s') * i / units.yr
    if unit == 's':
        return 1 / H(0, unit='s') * i
def tHubble(z, unit='Gyr'):
  """
  See cosmology.tH()
  """
  return tH(z, unit)

def lookback(z, unit='Gyr'):
    """
    Lookback time at a given redshift.
    The units can be 's', 'yr' or 'Gyr' (default units='Gyr').
    """
    f = lambda t: 1. / ((1. + t) * E(t))
    i = integrate.quad(f, 0., z, epsrel=1e-6, epsabs=0)[0]
    if unit == 'Gyr':
        return 1 / H(0, unit='s') * i / units.Gyr
    if unit == 'yr':
        return 1 / H(0, unit='s') * i / units.yr
    if unit == 's':
        return 1 / H(0, unit='s') * i

def dH(z=0, unit='Mpc'):
    """
    Hubble distance at a given redshift.
    The units can be 'cm', 'ly', or 'Mpc' (default units='Mpc').
    """
    if unit == 'Mpc':
        return constants.c / H(z) / units.km
    if unit == 'kpc':
        return constants.c / H(z, unit='s') / units.kpc
    if unit == 'cm':
        return constants.c / H(z, unit='s')
    if unit == 'm':
        return constants.c / H(z, unit='s') / units.m
    if unit == 'ly':
        return constants.c / H(z, unit='s') / units.ly

def dHubble(z=0, unit='Mpc'):
    """
    See cosmology.dH()
    """
    return dH(z, unit)

def dC(z1, unit='Mpc', **kwargs):
    """
    Comoving distance at a given redshift or betwen two redshifts.
    The units can be 'cm', 'ly' or 'Mpc' (default units='Mpc').
    """
    f = lambda t: 1 / E(t)
    i = integrate.romberg(f, 0., z1, **kwargs)
    return i * dH(0, unit)

def dM(z, dist, input_unit='deg', unit='Mpc'):
    """
    Transverse comoving distance at redshift z.

    input_unit can be 'arcsec', 'arcmin', 'deg', or 'rad'
    """
    if type(dist) in (list, tuple):
        dist = scipy.array(dist)
    if input_unit == 'arcsec':
        dist /= 3600.
        input_unit = 'deg'
    if input_unit == 'arcmin':
        dist /= 60
        input_unit = 'deg'
    if input_unit == 'deg':
        dist *= scipy.pi / 180
    if Omega_k == 0:
        return dist * dC(z, unit)
    elif Omega_k > 0:
        return dist * dH / scipy.sqrt(Omega_k) * \
               scipy.sinh(scipy.sqrt(Omega_k) * dC(z, unit) / dH(0, unit))
    else:
        return dist * dH / scipy.sqrt(-Omega_k) * \
               scipy.sin(scipy.sqrt(Omega_k) * dC(z, unit) / dH(0, unit))

def dA(z1, z2=0, unit='Mpc'):
    """
    Returns the angular diameter distance between two redshifts (from zero to
    z1, by default). If z2>0, then z2 should be greater than z1.

    The units can be 'cm', 'ly' or 'Mpc' (default units='Mpc').
    """
    if z2 == 0:
        return dC(z1, unit) / (1 + z1)
    elif Omega_k >= 0:
        pseudo_dist = dC(z2, unit) * \
                         scipy.sqrt(1 + Omega_k * \
                                    (dC(z1, unit) / dH(0, unit)) ** 2)
        pseudo_dist -= dC(z1, unit) * \
                       scipy.sqrt(1 + Omega_k * \
                                  (dC(z2, unit) / dH(0, unit)) ** 2)
        return pseudo_dist / (1 + z2)
    else:
        msg = 'Angular Diameter Distance not implemented for values Omega_k < 0'
        raise ValueError(msg)

def dL(z, unit='Mpc'):
    """
    Luminosity distance at a given redshift.
    The units can be 'cm', 'ly' or 'Mpc' (default units='Mpc').
    """
    return dC(z, unit) * (1 + z)

def dProj(z, dist, input_unit='deg', unit='Mpc'):
    """
    Projected distance, physical or angular, depending on the input units (if
    input_unit is physical, returns angular, and vice-versa).
    The units can be 'cm', 'ly' or 'Mpc' (default units='Mpc').
    """
    if input_unit in ('deg', 'arcmin', 'arcsec'):
        Da = dA(z, unit=unit)
    else:
        Da = dA(z, unit=input_unit)

    # from angular to physical
    if input_unit == 'deg':
        dist = Da * scipy.pi * dist / 180
    elif input_unit == 'arcmin':
        dist = Da * scipy.pi * dist / (180 * 60)
    elif input_unit == 'arcsec':
        dist = Da * scipy.pi * dist / (180 * 3600)
    # from physical to angular
    if unit == 'deg':
        dist = dist * 180 / (scipy.pi * Da)
    elif unit == 'arcmin':
        dist = dist * 180 * 60 / (scipy.pi * Da)
    elif unit == 'arcsec':
        dist = dist * 180 * 3600 / (scipy.pi * Da)
    return dist

def angular_separation(ra1, dec1, ra2, dec2, unit='deg'):
    # copied from astLib.astCoords.calcAngSepDeg
    # gives the same result to <1e-2 arcsec
    if unit == 'deg':
        ra1 *= scipy.pi / 180
        dec1 *= scipy.pi / 180
        ra2 *= scipy.pi / 180
        dec2 *= scipy.pi / 180
    dra = ra1 - ra2
    cos = scipy.sin(dec1) * scipy.sin(dec2) + \
          scipy.cos(dec1) * scipy.cos(dec2) * scipy.cos(dra)
    x = scipy.cos(dec2) * scipy.sin(dra) / cos
    y = scipy.cos(dec2) * scipy.sin(dec1) - \
        scipy.sin(dec2) * scipy.cos(dec1) * scipy.cos(dra)
    y /= cos
    return scipy.degrees(scipy.hypot(x, y))

def density(z=0, ref='critical', unit='astro'):
    """
    Returns the critical or average density at redshift z.
        *Units: Msun·Mpc⁻³ if unit=='astro' (default); else g·cm⁻³
    """
    assert ref[0] in 'acm'
    if ref[0] == 'c':
        d = 3 * H(z, unit='s')**2 / (8*scipy.pi*constants.G)
    elif ref[0] in ('a', 'm'):
        d = 3 * H(0., unit='s')**2 * Omega_M * (1+z)**3 / \
            (8 * scipy.pi * constants.G)
    if unit == 'astro':
        return d / constants.Msun * units.Mpc ** 3
    return d
