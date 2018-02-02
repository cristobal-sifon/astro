#!/usr/bin/env python
import itertools
import pylab
import readfile
import scipy
from astLib import astCoords
from astro import cosmology
from scipy import ndimage

def asymetry(x, y, lum=None, center=None, z=None, r200=1, dx=0.02, width=4,
              profile='gauss', rweight='wen13'):
  """
  measure the assymetry as defined by Wen & Han (2013)

  See smooth_map() for parameters.

  """
  I, b = smooth_map(x, y, lum, center, z, r200, dx, width, profile, rweight)
  s2 = (I**2).sum()
  if center is not None:
    center = -scipy.array(center)
  mI, b = smooth_map(-x, -y, lum, center, z, r200, dx, width, profile,
                     rweight)
  d2 = (I - mI) ** 2
  d2 = d2.sum() / 2
  return d2 / s2

def centroid_shift(array, levels=None):
  """
  Measure the how does the centroid vary by changing the isopleth that is used
  to measure it, similar to wX in X-rays.

  Parameters
  ----------
    array     : array of floats
                (smoothed) galaxy luminosity or number density. Must be a
                square array (i.e., same number of rows and columns).
    levels    : list of floats
                intensity levels to use for the centroid shift

  """
  cm = ndimage.measurements.center_of_mass
  xo = len(array)/2.
  yo = len(array[0])/2.
  if levels is None:
    rms = -1
    masked = array
    while rms != scipy.std(masked):
      rms = scipy.std(masked)
      masked = masked[masked < 3*rms]
    levels = scipy.arange(2, 5.1, 0.5) * rms
  for l in levels:
    masked = array
    masked[masked < l] = 0
    #c = cm(masked
  return

def smooth_map(x, y, lum=None, center=None, z=None, r200=1, dx=0.02, width=4,
               profile='gauss', rweight='wen13'):
  """
  Calculate a smoothed map of the light distribution or galaxy number
  density, using the radial weighting of Wen & Han (2013)

  Parameters
  ----------
    x         : array of floats
                galaxy positions in the x direction, in Mpc (relative to
                cluster center unless center is given).
    y         : array of floats
                galaxy positions in the y direction, in Mpc (relative to
                cluster center unless center is given).
    lum       : array of floats, same length as x and y (optional)
                galaxy luminosity or other weighting value. If not specified
                the map returned corresponds to the galaxy density map.
    center    : array-like of 2 floats (optional)
                if defined, must be an array-like with the coordinates of
                the cluster center (RA and Dec). x and y must then be RA and
                Dec, respectively, and distances will be calculated here.
    z         : float (must be given if center is given)
                redshift of the cluster.
    r200      : float (default 1)
                radius of the cluster in Mpc, used mainly for the radial
                weighting of Wen & Han. Alternatively, x and y can already be
                in units of r200 or the desired radius.
    dx        : float (default 0.02)
                width of the bin, in Mpc. The same size is used for x and y.
    width     : float (default 4)
                total width of the map, in Mpc.
    profile   : str (default 'gaussian')
                2d profile to use in the smoothing. Currently only a gaussian
                with the radial weighting of Wen & Han (2013).
    rweight   : {'wen13', float}
                radial weighting for the smoothed profile. Currently, can only
                be that of Wen & Han or a constant weight for all galaxies,
                in which case the width of the gaussian (in Mpc) should be
                given.

  Returns
  -------
    smap      : 2-dimensional array of floats
                the smoothed map.

  Notes
  -----
    - Any selection of galaxies must be done outside of this function

  """
  # first, check data formats
  N = len(x)
  if len(y) != N:
    raise ValueError('lengths of x and y must be the same')
  if lum is not None:
    if len(lum) != N:
      raise ValueError('length of lum must be the same as len(x) if defined')
    lum = 1
  if center is not None:
    if len(center) != 2:
      msg = 'argument *center* must be an array-like with 2 values'
      raise ValueError(msg)
    if z is None:
      msg = 'argument center is given, therefore a redshift must be given'
      raise ValueError(msg)
  if profile != 'gauss':
    msg = 'WARNING: only a gaussian profile is implemented;'
    msg += ' falling back to it'
    print msg
    profile = 'gaussian'

  x200 = r200 / dx
  # check whether center is given
  if center is not None:
    dz = cosmology.dA(z) * scipy.pi/180
    # want to do it like this because the sign matters!
    x = -1 * (x - center[0]) * scipy.cos(scipy.pi/180 * center[1]) * dz
    y = (y - center[1]) * dz
  # create grid
  edge = width/2.
  bins = scipy.arange(-edge-dx/2., edge+dx, dx)
  Nbins = len(bins)
  xo = yo = scipy.array([(bins[i]+bins[i-1])/2 for i in xrange(1, Nbins)])
  # each galaxy's distance to each cell (dimension 3xN)
  dcells = scipy.array([[scipy.hypot(x-xi, y-yj) for xi in xo] for yj in yo])
  dist = scipy.hypot(x, y) / x200
  # choice of profile and calculation of map
  if profile == 'gauss':
    def p(t, s):
      # why is there a s**2 in the denominator, instead of s?
      return dx**2 * scipy.exp(-t**2/(2*s**2)) / (s*scipy.sqrt(2*scipy.pi))
    if rweight == 'wen13':
      w = (0.03 + 0.15*dist) * x200
    else:
      w = rweight
    grid = [scipy.array([sum(lum * p(dij, w)) for dij in dcells[jj]]) \
            for jj in xrange(len(yo))]
  # done!
  return scipy.array(grid), bins

def map_residual(smap, z=None, r200=1, dx=0.02, width=4, profile='king'):
  return