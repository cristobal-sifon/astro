#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import matplotlib
import os
import pylab
#import readfile
import numpy
import sys
from astLib import astCoords
from itertools import izip
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter

from matplotlib import rcParams
rcParams['axes.labelsize'] = 16

import operator
import time

c = 299792.458

def Delta(data, z0=0, cluster='', Nnear=10, Nsim=1000,
          stat=numpy.median, plot_filename=False,
          output=False, verbose=True, full_output=False):
    """
    Perform the DS test (Dressler & Schechtman 1998) and estimate Delta and its
    significance.

    Parameters
    ----------
    data : str or tuple of arrays
      Can be one of two things: (1) the name of the catalog file where the
      first 4 columns must be: ID, RA, Dec, z; (2) a tuple with ID, RA, Dec,
      z.
    z0 : float (default=0)
      Redshift of the cluster. If equal to zero, will be the biweight location
      estimator of the given sample
    cluster : str (default='')
      Name of the cluster, just for printing issues. Optional
    Nnear : int (default=10)
      Number of neighbors to consider for calculating Delta
    Nsim : int (default=1000)
      Number of times to bootstrap the sample to get the significance of Delta
    plot_filename : str (optional)
      Make a bubble plot? A string with the output filename should be given.
      Can be any format taken by matplotlib
    output : str (optional)
      Generate an output file with individual delta's? A string with the
      output filename should be given. The file would contain the following
      columns: ID, RA, Dec, delta
    verbose : bool (default True)
      whether to print results on screen
    full_output : bool (default False)

    Returns
    -------
    Delta : float
      The sum of individual delta's.
    sig : float
      The significance (i.e., "p-value") of Delta, estimated by bootstrap
      resampling.
    delta : array of floats (optional)
      If full_output==True. Return the delta's for individual galaxies.

    """
    if type(data) == str:
        #data = readfile.table(data, cols=(0,1,2,3))
        data = numpy.loadtxt(data, usecols=(0,1,2,3))
    elif len(data) != 4:
        msg = 'Wrong data type for variable data. Must be either\n'
        msg += '   (1) the name of the catalog file where the first 4'
        msg += ' columns are ID, RA, Dec, z; or'
        msg += '   (2) a tuple/list/array with ID, RA, Dec, z'
        print(msg)
        return
    array = numpy.array

    obj = array(data[0])
    ra = array(data[1])
    dec = array(data[2])
    z = array(data[3])

    if z0 == 0:
        z0 = Cbi(z)
    v = c * (z - z0) / (1 + z0)
    delta, Delta = getDelta(v, ra, dec, stat, Nnear)
    sig = significance(Delta, v, ra, dec, stat, Nnear, Nsim)
    if output:
        print_results(output, obj, ra, dec, delta)
    if plot_filename:
        plot(plot_filename, delta, ra, dec, v, Nnear, cluster)
    if verbose:
        if cluster:
            print('%s (%d members)' %(cluster, len(z)))
        print('Using %d neighbors and %d samples' \
              %(Nnear, Nsim))
        print('Delta   : %7.3f' %Delta)
        print('p-value : %7.3f\n' %sig)
    if full_output:
        return Delta, sig, delta
    return Delta, sig

def Cbi(x, c=6.):
    """
    Biweight Location estimator
    """
    mad = numpy.median(numpy.absolute(x - numpy.median(x)))
    m = numpy.median(x)
    u = (x - m) / (c * mad)
    good = (abs(u) < 1)
    num = sum((x[good] - m) * (1 - u[good]**2) ** 2)
    den = sum((1 - u[good]**2) ** 2)
    return m + num / den

def getDelta(v, ra, dec, stat, Nnear=10):
    vo = stat(v)
    stdv = numpy.std(v)
    results = [Near(v, ra, dec, x, y, stat, Nnear) for x, y in izip(ra, dec)]
    vnear, snear = numpy.transpose(results)
    #delta = numpy.sqrt(Nnear)/stdv * numpy.hypot(vnear-vo, snear-stdv)
    delta = operator.mul(operator.div(numpy.sqrt(Nnear), stdv),
                         numpy.hypot(vnear-vo, snear-stdv))
    return delta, sum(delta)

def Near(v, ra, dec, x, y, stat, Nnear):
    d = astCoords.calcAngSepDeg(ra, dec, x, y)
    dsort = numpy.argsort(d)
    vgroup = v[dsort[1:Nnear+1]]
    return stat(vgroup), numpy.std(vgroup)

def significance(Delta, v, ra, dec, stat, Nnear=10, Nsim=1000):
    permutation = numpy.random.permutation
    Q = numpy.array([getDelta(permutation(v), ra, dec, stat, Nnear)[1] \
                     for i in xrange(Nsim)])
    return len(Q[Q > Delta]) / float(Nsim)

def print_results(output, obj, ra, dec, delta):
    out = open(output, 'w')
    print('# ID    RA    Dec    delta', file=out)
    for i in xrange(len(obj)):
        print('%10s   %9.5f   %8.4f   %.3f' \
                 %(obj[i], ra[i], dec[i], delta[i]),
              file=out)
    out.close()
    return

def plot(filename, delta, ra, dec, v, Nnear, cluster='', color=True):
    # plot limits
    rawidth  = 1.2 * (max(ra) - min(ra))
    decwidth = 1.2 * (max(dec) - min(dec))
    avg_circle_size = rawidth / 50.
    n = len(v)
    j = numpy.arange(n, dtype=int)
    positive = j[v >= 0]
    negative = j[v < 0]
    fig, ax = pylab.subplots(figsize=(5.5,5))
    if color:
        colors = ('r', 'b')
    else:
        colors = ('k', 'k')
    size = numpy.exp(delta/2)
    size *= avg_circle_size / numpy.average(size)
    for indices, color in izip((positive, negative), colors):
        patches = [Circle((ra[i], dec[i]), size[i], fill=None) \
                   for i in indices]
        p = PatchCollection(patches, alpha=0.6, facecolors='w', lw=1.5,
                            edgecolors=color)
        ax.add_collection(p)
    ax.plot(ra[0], dec[0], ',', color='w')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('DEC (deg)')
    pylab.title(cluster + ' (%d neighbours)' %Nnear)
    halfra = (rawidth) / 2
    halfdec = (decwidth) / 2
    center = ((max(ra) + min(ra)) / 2, (max(dec) + min(dec)) / 2)
    #       RAMin       DECMin       RAMax       DECMax
    corners = (center[0] - halfra, center[1] - halfdec,
               center[0] + halfra, center[1] + halfdec)
    ax.set_xlim(corners[0], corners[2])
    ax.set_ylim(corners[1], corners[3])
    fig.tight_layout()
    pylab.savefig(filename, format=filename[-3:])
    pylab.close()
    return
