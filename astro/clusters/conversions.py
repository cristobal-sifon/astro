# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy
import uncertainties
from astro import constants, cosmology, units

# from the KiDS GGL pipeline
# should maybe include the NFW file in this package
#from kids_ggl_pipeline.halomodel import nfw as nfw_func

def Msph(r, z, dr=0., ref='200c', unit='Msun', r_unit='kpc'):
    """
    Calculates the mass of a cluster with radius r, assuming it is
    spherical.

    Supported units are Msun (for M) and {Mpc,kpc} for r.

    """
    contrast = int(ref[:-1])
    # [rho] = Msun·Mpc⁻³
    rho = contrast * cosmology.density(z, ref=ref[-1], unit='astro')
    #        m = 4*pi/3 * rho*r**3
    # --> r**3 = m / (4*pi/3) / rho
    if r_unit == 'kpc':
        r = r / 1000
    m = rho * 4 * numpy.pi / 3 * r**3 # [M] = Msun

    if dr != 0.:
        dm = rho * 4 * numpy.pi * r**2 # [dM] = Msun
    # later: include alternative units
    if unit == 'Msun':
        if dr != 0:
            return m, dm
        else:
            return m


def rsph(m, z, dm=0., ref='200c', unit='Mpc'):
    """
    Returns r, the radius that contains a density *contrast* times
    the critical density, assuming a spherical distribution, given a
    mass m,
    
    m = (4·pi/3)·r³·(contrast·rho_c)

    If an error on the mass is provided, also returns the error in r.
        *Input:
    m: the mass of the cluster, in solar masses. Can be an
        uncertainties.ufloat as well, in which case *r* will be
        returned with the same type.
    z: the redshift of the cluster
    dm: the uncertainty on the mass, in solar masses
    contrast: the density contrast with respect to the critical (or
        average) density of the Universe at redshift z.
        *Returns:
    r: the spherical radius, in kpc
    dr: the uncertainty on r, in kpc -- if dm is given

    """
    contrast = int(ref[:-1])
    # [rho] = Msun·Mpc⁻³
    rho = contrast * cosmology.density(z, ref=ref[-1], unit='astro')
    #        m = 4*pi/3 * rho*r**3
    # --> r**3 = m / (4*pi/3) / rho
    r = (m / (4 * numpy.pi / 3) / rho) ** (1. / 3.) # [r] = Mpc
    if dm != 0. and type(m) != uncertainties.Variable:
        dr = 1 / (4 * numpy.pi) * 1 / (rho * r ** 2) * dm # [dr] = Mpc
        if unit == 'kpc':
            return 1e3 * r, 1e3 * dr
        if unit == 'Mpc':
            return r, dr
    if unit == 'kpc':
        return 1e3 * r
    if unit == 'Mpc':
        return r


def nfw(parameters, new, z=0):
    """
    Convert properties from one NFW profile to another one taking some
    other fixed values.
    
    For instance, find the mass of an NFW with a different concentration
    but the same normalization rho_s

    Parameters
    ----------
        parameters : array-like, shape = 2x2
                  the name and value of the original NFW parameters.
                  Possible names are:
                        'm' or 'm200' -- virial mass
                        'c' -- concentration
                        'rs' -- scale radius
                        'r200' -- virial radius
                        'rho' or 'rho_s' -- central density
                                            (i.e., normalization)
        output  : value to 

    """
    # check for wrong inputs later
    
    return




