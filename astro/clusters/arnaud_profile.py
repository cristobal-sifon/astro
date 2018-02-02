#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import scipy
import scipy.integrate

#from .mass_conversion_felipe import convert_mass

# local
#from astro import constants, cosmology, units
#from astro.clusters import conversions
from . import conversions
from .. import constants, cosmology, units

cosmology.h = 0.7
h70 = 1.

"""
see M. Arnaud et al., 2010, A&A, 517, A92
"""

# Eq. 7
alpha_p = 0.12
# Eq. B.2, self-similar values
P0    = 8.130 / h70 ** 1.5
c500  = 1.156
gamma = 0.3292
alpha = 1.0620
beta  = 5.4807


# for self-similar
#alpha_p = 0
def alpha_prime(x):
  """
  Eq. 8
  """
  # for self-similar
  return 0
  f = (2 * x) ** 3
  f = f / (1 + f)
  return 0.10 - (alpha_p + 0.10) * f


def P(x, M500, z):
  """
  Universal pressure profile, Eq. 13
  """
  return P500(M500, z) * (M500 / 3e14) ** (alpha_p + alpha_prime(x)) * p(x)


def P500(M500, z, unit = 'astro'):
  """
  Eq. 5
  """
  # in units of keV·cm⁻³
  #pp = 1.65e-3 * (cosmology.h * cosmology.E(z)) ** (8./3.) * (h70 * M500 / 3e14) ** (2./3.) * h70 ** 2
  pp = 1.65e-3 * cosmology.E(z) ** (8./3.) * (M500 / 3e14) ** (2./3.) * h70 ** 2
  # in units of erg·cm⁻³ = g·cm⁻¹·s⁻²
  pp = pp * units.keV
  if unit == 'cgs':
    return pp
  # in units of kg·m⁻¹·s⁻²
  if unit == 'mks':
    return pp / units.kg * units.m
  # in units of Msun·Mpc⁻¹·s⁻²
  if unit == 'astro':
    return pp / units.Msun * units.Mpc
  return pp


def p(x):
  """
  Eq. 11
  """
  cx = c500 * x
  exp = (beta - gamma) / alpha
  return P0 / (cx ** gamma * (1 + cx ** alpha) ** exp)


def Ysph(R, R500, M500, z):
  f = lambda r: P(r / R500, M500, z) * r ** 2
  i = 4 * scipy.pi * scipy.integrate.quad(f, 0, R)[0]
  return i * constants.Msun * constants.sigmaT / (constants.me * constants.c ** 2)


def Ycyl(R, R500, M500, z):
  """
  Y_{SZ}·D_A**2, Eq. 15
  """
  Rb = 5 * R500
  f = lambda x, r: 4 * scipy.pi * r * P(x / R500, M500, z) * x / scipy.sqrt(x ** 2 - r ** 2)
  i = scipy.integrate.dblquad(f, 0, R, lambda r: r, lambda r: Rb)
  return i[0] * constants.Msun * constants.sigmaT / (constants.me * constants.c ** 2)


def I(x):
  """
  Eq. 24
  """
  f = lambda u: p(u) * u ** 2
  return 3 * scipy.integrate.quad(f, 0, x)[0]


def J(x):
  """
  Eq. 27
  """
  f = lambda u: p(u) * scipy.sqrt(u ** 2 - x ** 2) * u
  return I(5) - 3 * scipy.integrate.quad(f, x, 5)[0]


def conversion_r500(m500, z, slope = 0.6):

  r500 = conversions.rsph(m500, z, ref = '500c', unit = 'Mpc')
  #m200 = m500 * transform_M500c(m500, z, (0.27, 0.73, 0.7))
  m200 = profiles.nfw(m500, z, ref_in='500c', ref_out='200c')
  r200 = conversions.rsph(m200, z, ref = '200c', unit = 'Mpc')

  y500_sph = Ysph(r500, r500, m500, z)
  y200_sph = Ysph(r200, r500, m500, z)
  y500_cyl = Ycyl(r500, r500, m500, z)
  y200_cyl = Ycyl(r200, r500, m500, z)

  # checking the Arnaud et al. conversion
  y5r500 = Ysph(5 * r500, r500, m500, z)
  #print(y5r500/ Ycyl(5 * r500, r500, m500, z))
  #exit()

  r52 = r500 / r200; m52 = m500 / m200
  y52s = y500_sph / y200_sph; y52c = y500_cyl / y200_cyl
  #ysc = y200_sph / y200_cyl
  print()
  print('z = {0:.3f}'.format(z))
  print('r500 = {0:.3f}'.format(r500))
  print('r500/r200 = {0:.3f} ({1:.3f})'.format(r52, 1 / r52))
  print('m500/m200 = {0:.3f} ({1:.3f})'.format(m52, 1 / m52))
  print('(y500/y200)_sph = {0:.3f} ({1:.3f})'.format(y52s, 1 / y52s))
  print('(y500/y200)_cyl = {0:.3f} ({1:.3f})'.format(y52c, 1 / y52c))
  print(' For y_sph: r500 to r200 --> factor {0:.3f} for a' \
        ' slope {1:.3f}'.format(y52s**slope / m52, slope))
  print(' For y_cyl: r500 to r200 --> factor {0:.3f} for a'
        ' slope {1:.3f}'.format(y52c**slope / m52, slope))
  print('\ny_5x500 = {0:.8f}'.format(y5r500))
  print('y_500 = {0:.8f}'.format(y500_sph))
  print('y200_cyl = {0:.8f}'.format(y200_cyl))
  print('(y_5x500/y500)_sph = {0:.3f}'.format(y5r500 / y500_sph))
  print('y200_sph/y200_cyl = {0:.3f} ({1:.3f})'.format(
            y200_sph/y200_cyl, y200_cyl/y200_sph))
  return


def conversion_r200a(m200a, z, slope = 0.6):

  r200a = conversions.rsph(m200a, z, ref = '200a')
  #m200, m500a, m500, m2500 = convert_mass(m200a, z, (0.27, 0.73, 0.7))
  m200 = profiles.nfw(m200a, z, ref_in='200a', ref_out='200c')
  m500 = profiles.nfw(m200a, z, ref_in='200a', ref_out='500c')
  r500  = conversions.rsph(m500, z, ref = '500c')
  r200  = conversions.rsph(m200, z, ref = '200c')

  y200a_sph = Ysph(r200a, r500, m500, z)
  y200_sph  = Ysph(r200,  r500, m500, z)
  y200a_cyl = Ycyl(r200a, r500, m500, z)
  y200_cyl  = Ycyl(r200,  r500, m500, z)

  rac = r200a / r200; mac = m200a / m200
  yacs = y200a_sph / y200_sph; yacc = y200a_cyl / y200_cyl
  print()
  print('z = {0:.3f}'.format(z))
  print('r200a/r200 = {0:.3f} ({1:.3f})'.format(rac, 1 / rac))
  print('m200a/m200 = {0:.3f} ({1:.3f})'.format(mac, 1 / mac))
  print('(y200a/y200)_sph = {0:.3f} ({1:.3f})'.format(yacs, 1 / yacs))
  print('(y200a/y200)_cyl = {0:.3f} ({1:.3f})'.format(yacc, 1 / yacc))
  print(' For y_sph: r200a to r200 --> factor {0:.3f} for a' \
        ' slope {1:.3f}'.format(yacs**slope / mac, slope))
  print(' For y_cyl: r200a to r200 --> factor {0:.3f} for a' \
        ' slope {1:.3f}'.format(yacc**slope / mac, slope))
  return


def profile_sph(M500, r, z, ref = 'absolute', verbose = 'no'):
  """
  Eqs. 22-24
  """
  if ref == 'absolute':
    r500 = conversions.rsph(M500, z, ref = '500c', unit = 'Mpc')
    x = r / r500
  if ref == 'relative':
    x = r
  if verbose == 'yes':
    print('Calculating Ysph at %.2f*r500 (z=%.3f)'.format(x, z))
  Ax = 2.925e-5 * I(x) / h70
  y = Ax * (M500 / 3e14) ** 1.78
  #if z > 0:
    #y = y / cosmology.dA(z) ** 2
  return y


def profile_cyl(M500, r, z, ref = 'absolute', verbose = 'no'):
  """
  Eqs. 25-27
  """
  r500 = conversions.rsph(M500, z, ref = '500c', unit = 'Mpc')
  if ref == 'absolute':
    x = r / r500
  if ref == 'relative':
    x = r
  if verbose == 'yes':
    print('Calculating Ycyl at {0:.2f}*r500 (z={1:.3f})'.format(x, z))
  Bx = 2.925e-5 * J(x) / h70
  y = Bx * (M500 / 3e14) ** 1.78
  #if z > 0:
    #y = y / cosmology.dA(z) ** 2
  return y


def Yratio(ref1, ref2, proj = 'sph'):
  """
  return the ratio of Y@ref2 to Y@ref1 (i.e., should multiply the value 
originally given, at ref1)
  """
  m = 1e15; z = 0.5 # unimportant for the ratio
  r1 = conversions.rsph(m, z, ref = ref1)
  r2 = conversions.rsph(m, z, ref = ref2)
  if proj == 'sph':
    y1 = profile_sph(m, r1 / 1e3, z)
    y2 = profile_sph(m, r2 / 1e3, z)
  elif proj == 'cyl':
    y1 = profile_cyl(m, r1 / 1e3, z)
    y2 = profile_cyl(m, r2 / 1e3, z)
  return y2 / y1

