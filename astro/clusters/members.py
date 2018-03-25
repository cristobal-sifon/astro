#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import pylab
import numpy

"""
Galaxy cluster membership selection procedures

"""

# the speed of light in km/s
c = 299792.458

def sgapper(r, z, zo=0, cut=4000, rcut=5000, useabs=True, min_Nperbin=15,
            min_binsize=250, converge=True, maingap=500, maxgap=1000,
            sigma=False, verbose=True, debug=False, plot_output=False,
            full_output=False):
    """
    Shifting gapper (e.g., Katgert et al. 1996)

    Parameters
    ----------
      r         : float
                  distances to chosen center
      z         : array of floats
                  redshifts
      zo        : float (optional)
                  redshift of the cluster. If not given, will be determined
                  internally using the biweight estimator of location, Cbi.
      cut       : float (default 4000)
                  initial velocity cut. Set to zero to disable an initial cut
                  at fixed peculiar velocity.
      rcut      : float (default 5000)
                  maximum radius. All galaxies outside this radius will be
                  discarded prior to any analysis.
      useabs    : boolean (default True) -- NOT YET IMPLEMENTED
                  whether to use absolute values of the velocities to define
                  membership. Should only be false if there is a large number of
                  members (>~100).
      min_Nperbin : int (default 15)
                  minimum number of galaxies in each radial bin
      min_binsize : float (default 250)
                  minimum size of each bin, in kpc
      maingap   : float (default 500)
                  gap in velocity space that defines the "main body" of
                  galaxies for each bin
      maxgap    : float (default 1000)
                  gap in velocity space that defines the limit of membership
                  for each bin
      converge  : boolean (default True)
                  whether to run the shifting gapper iteratively until it
                  converges on the number of members, or to run it one single
                  time.
      sigma     : boolean (default False)
                  whether maxgap is in units of the velocity dispersion, in
                  which case maingap is ignored.
      verbose   : boolean (default True)
                  verbose.
      debug     : boolean (default False)
                  debug level verbose, including plots
      plot_output : {False, str} (default False)
                  whether to plot the rv diagram. If not False, should be the
                  name of the output, including the extension.
      full_output : boolean (default False)
                  if True, return a tuple with the member indices, the bin
                  sizes (of the last iteration) in kpc, the number of
                  iterations until convergence (nit) and the indices of
                  galaxies that passed the initial velocity cut. If
                  converge is set to False, return nit=1.

    Returns
    -------
      members   : array of ints
                  the indices of the galaxies selected as members

    Optional output
    ---------------
      binsize   : array of floats
                  the sizes of all bins for the last iteration, in kpc.
      nit       : int
                  number of iterations before the algorithm converged.

    """

    ## Auxiliary functions
    ## ------------------------------------------------------------- ##
    def binning(r, min_Nperbin=15, min_binsize=250, debug=False):
        """
        r contains the ordered distances

        """
        N = len(r)
        maxbins = int(float(N) / min_Nperbin) + 1
        # counter
        n = 0
        bins = []
        rlim = []
        binsizes = []
        for i in xrange(N):
            if n == N:
                break
            bins.append([])
            while len(bins[i]) < min_Nperbin and n < N:
                bins[i].append(n)
                n += 1
            if len(bins) == 1:
                while r[bins[i][-1]] < min_binsize:
                    bins[i].append(n)
                    n += 1
            else:
                while r[bins[i][-1]] - r[bins[i-1][-1]] < min_binsize \
                    and n < N:
                    bins[i].append(n)
                    n += 1
            # so that the last bin doesn't contain too few galaxies
            if N - n <= 0.8 * min_Nperbin:
                for j in xrange(n, N):
                    bins[i].append(j)
                n = N
            if i == 0:
                binsize = r[bins[i][-1]]
            else:
                binsize = r[bins[i][-1]] - r[bins[i-1][-1]]
            binsizes.append(binsize)
            if debug:
                print('binsize:', binsize, 'kpc')
                for i in bins:
                    pylab.axvline(r[i][-1], ls=':', color='b')
        if debug:
            print('Nbins:', len(bins))
        return bins, binsizes

    from itertools import count, izip
    def membership(r, v, bins, maingap=500, maxgap=1000,
                   useabs=True, sigma=False, debug=False):
      """
      TEST: if sigma is True, then maxgap should be in units of the velocity
      dispersion and maingap is ignored.

      will eventually implement useabs=False, and return the body limits
      """
      Nbins = len(bins)
      members = []
      limits = numpy.zeros(Nbins)
      #pylab.plot(r, abs(v), 'ko')
      for i in xrange(Nbins):
        bi = bins[i]
        vi = abs(v[bi])
        #pylab.axvline(max(r[bi]), ls=':', color='grey')
        Ni = len(vi)
        if sigma:
          # maybe should be iterative somehow
          s = numpy.std(vi)
          members = numpy.append(members, bi[vi < maxgap*s])
        else:
          bodylimit = None
          vorder = numpy.argsort(vi)
          #for j, rj, vj in izip(count(), r[bi][vorder], vi[vorder]):
              #pylab.annotate(j, xy=(rj, vj), color='r',
                             #ha='center', va='center')
          # the one with the lowest velocity is always a member
          members.append(bi[vorder[0]])
          for j in range(1, Ni):
            # the bodylimit is to be found only once per bin
            if not bodylimit:
              if vi[vorder[j]] - vi[vorder[j-1]] < maingap:
                #pylab.plot(r[bi][vorder[j]], vi[vorder[j]], 'gx', lw=2, ms=10)
                members.append(bi[vorder[j]])
              # if maingap is exceeded, define the body gap
              else:
                bodylimit = vi[vorder[j-1]]
                #pylab.plot([min(r[bi]), max(r[bi])],
                           #[bodylimit, bodylimit],
                           #ls='--', color='b')
                #pylab.plot([min(r[bi]), max(r[bi])],
                           #[bodylimit+maxgap, bodylimit+maxgap],
                           #ls='-', color='b')
            if bodylimit:
              if vi[vorder[j]] - bodylimit < maxgap:
                members.append(bi[vorder[j]])
                #pylab.plot(r[bi][vorder[j]], vi[vorder[j]], 'gx', lw=2, ms=10)
              else:
                break
          if bodylimit is None:
            limits[i] = max(vi)
          else:
            limits[i] = bodylimit
      if sigma:
        members = map(int, members)
        return numpy.array(members)
      #pylab.show()
      return numpy.array(members), limits

    def plot(x, y, memb=[], binsizes=[], limits=[],
             maxgap=1000, ylim=5000, output=False, verbose=True):
        if type(output) == str:
            pylab.axes([0.12, 0.1, 0.84, 0.86])
        if len(memb) > 0:
            nonmemb = numpy.array([i for i in xrange(len(x))
                                   if i not in memb])
            if len(nonmemb) > 0:
                pylab.plot(x[nonmemb], y[nonmemb], 'wo')
            pylab.plot(x[memb], y[memb], 'ko')
            xlim = 1.2*max(x[memb])
        else:
            pylab.plot(x, y, 'wo')
            xlim = 1.2*max(x)
        if len(binsizes) != 0:
            bins = [binsizes[i]+sum(binsizes[:i])
                    for i in xrange(len(binsizes))]
            for b in bins:
                pylab.axvline(b, ls=':', color='b')
            if len(limits) == len(binsizes):
                xb = (0, bins[0])
                yb = numpy.array([limits[0], limits[0]])
                pylab.plot(xb, yb, 'r--')
                pylab.plot(xb, -yb, 'r--')
                pylab.plot(xb, maxgap + yb, 'r-')
                pylab.plot(xb, -(maxgap + yb), 'r-')
                for i in xrange(1, len(limits)):
                    xb = (bins[i-1], bins[i])
                    yb = numpy.array([limits[i], limits[i]])
                    pylab.plot(xb, yb, 'r--')
                    pylab.plot(xb, -yb, 'r--')
                    pylab.plot(xb, maxgap + yb, 'r-')
                    pylab.plot(xb, -(maxgap + yb), 'r-')
        pylab.axhline(0, ls='--', color='k')
        pylab.xlim(0, xlim)
        pylab.ylim(-ylim, ylim)
        if xlim < 100:
            pylab.xlabel('distance (Mpc)')
        else:
            pylab.xlabel('distance (kpc)')
        pylab.ylabel('velocity (km/s)')
        if type(output) == str:
            pylab.savefig(output, format=output[-3:])
            if verbose:
                print('Saved to', output)
        else:
            pylab.show()
        pylab.close()
        return

    def run_membership(r, v, j,
                       min_Nperbin=15, min_binsize=250,
                       maingap=500, maxgap=1000, sigma=False,
                       verbose=True, debug=False):
      rorder = numpy.argsort(r[j])
      bins, binsizes = binning(r[j][rorder],
                               min_Nperbin=min_Nperbin,
                               min_binsize=min_binsize,
                               debug=debug)
      if sigma:
        members = membership(r[j][rorder], v[j][rorder],
                             bins, maingap=maingap,
                             maxgap=maxgap, sigma=sigma, useabs=useabs,
                             debug=False)
        return j[rorder[members]], binsizes
      members, limits = membership(r[j][rorder], v[j][rorder],
                                   bins, maingap=maingap,
                                   maxgap=maxgap, sigma=sigma, useabs=useabs,
                                   debug=debug)
      return j[rorder[members]], binsizes, limits

    def update(z, index):
      zo = Cbi(z[index])
      v = c * (z - zo) / (1 + zo)
      if verbose:
        print('  zo = %.4f' %zo)
      return zo, v

    def zoguess(z, zo=None):
      c = 299792.46
      if zo is None:
        bins = numpy.arange(min(z)-1e-4/c, max(z)+1e-4/c, 500/c)
      else:
        bins = numpy.arange(zo-1e4/c, zo+1e4/c, 500/c)
      hist, edges = numpy.histogram(z, bins=bins)
      centers = numpy.array([(edges[i]+edges[i-1])/2 \
                            for i in xrange(1, len(edges))])
      j = numpy.argmax(hist)
      if j == len(hist) - 1:
        zo = numpy.median(z[z > centers[j-1]])
      elif j == 0:
        zo = numpy.median(z[z < centers[1]])
      else:
        zo = numpy.median(z[(z > centers[j-1]) & (z < centers[j+1])])
      return zo
    ## ------------------------------------------------------------- ##

    no = len(r)
    if verbose:
      print(no, 'galaxies initially')
    j = numpy.arange(no)
    # initial redshift and peculiar velocities
    zo = zoguess(z, zo)
    v = c * (z - zo) / (1 + zo)
    if debug:
      print('zo = %.4f' %zo)
      bins = numpy.arange(0, 1.001, 0.01)
      pylab.hist(z, bins=bins, histtype='step')
      pylab.axvline(zo, ls='--')
      pylab.annotate('%d galaxies' %len(z), xy=(0.65,0.85),
                     xycoords='axes fraction', fontsize=16)
      pylab.xlabel('z')
      pylab.show()
      pylab.close()
      plot(r, v, ylim=10000)

    # initial cut
    if cut > 0:
      j = j[(abs(v) < cut) & (r < rcut)]
      n = len(j)
      if verbose:
        msg = '%d galaxies after initial cuts of %d km/s and' \
               %(n, cut)
        if numpy.median(r) < 10:
            msg += ' %.2f Mpc' %rcut
        else:
            msg += ' %d kpc' %rcut
        print(msg)
    else:
      if verbose:
        print('No initial cut based on velocity')
    if full_output:
      incut = j
    if debug:
      print(sorted(j), n)
    # calibrate redshift and peculiar velocities
    zo = numpy.median(z[j])
    v = c * (z - zo) / (1 + zo)

    results = run_membership(r, v, j,
                             min_Nperbin=min_Nperbin,
                             min_binsize=min_binsize,
                             maingap=maingap,
                             maxgap=maxgap,
                             sigma=sigma,
                             verbose=verbose,
                             debug=debug)
    if sigma:
      members, binsizes_orig = results
      limits = []
    else:
      members, binsizes_orig, limits = results

    if verbose:
      print('First pass: %d members' %len(members))
    if debug:
      print('members:', sorted(members))
      plot(r, v, members, binsizes=binsizes_orig, limits=limits)
    zo, v = update(z, members)

    if converge:
      nit = 1
      # a sufficiently large number to start
      N = 10000
      while N > len(members):
        N = len(members)
        results = run_membership(r, v, members,
                                 min_Nperbin=min_Nperbin,
                                 min_binsize=min_binsize,
                                 maingap=maingap,
                                 maxgap=maxgap,
                                 sigma=sigma,
                                 verbose=verbose,
                                 debug=debug)
        if sigma:
          members, binsizes = results
          limits = []
        else:
          members, binsizes, limits = results

        zo, v = update(z, members)
        nit += 1
        if verbose:
          print('Iteration %d: %d members' %(nit, len(members)))
        if debug:
          print('members:', sorted(members))
          plot(r, v, members, binsizes=binsizes, limits=limits)
      if verbose:
        print('Converged after %d iterations' %nit)

    if plot_output:
      plot(r, v, members, binsizes=binsizes_orig, limits=limits,
           output=plot_output, verbose=verbose)

    if verbose:
      print('Final number: %d members' %N)
    if full_output:
      return members, binsizes_orig, nit, incut
    return members

def Cbi(x, c=6.):
    """
    Biweight Location estimator
    """
    mad = MAD(x)
    m = numpy.median(x)
    u = (x - m) / (c * mad)
    good = (abs(u) < 1)
    num = sum((x[good] - m) * (1 - u[good]**2) ** 2)
    den = sum((1 - u[good]**2) ** 2)
    return m + num / den

def MAD(x):
    n = numpy.absolute(x - numpy.median(x))
    return numpy.median(n)
