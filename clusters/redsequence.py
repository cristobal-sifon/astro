#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Red Sequence fitting procedure

Fits the RS iteratively, first using ECGMM (Hao et al. 2009) to define the
initial sample to fit a RS to, and then doing a weighted least squares fit to a
straight line. Both these processes are iterated until the number of galaxies
considered converges.

Has the option to plot the CMD along with the results of ECGMM. All objects in
the catalog will be used; therefore, all "cleaning" (e.g., removing stars,
choosing aperture) should be done prior to calling this script.

Notes
-----
  ECGMM will fail if there are too few objects, and the red sequence fitting
  might fail in some cases, depending on how strong is the red sequence
  feature. For these cases, set argument do_ecgmm=False.

  It is recommended to set method='mle'. This performs better than 'wls', and
  'bayes' hasn't been tested extensively and may be quite slower (indeed it
  hasn't been properly debugged).

  The errors on the red sequence parameters look a bit overestimated. This
  hasn't been extensively tested yet.

References
----------
  Hao, J., Koester, B. P., McKay, T. A., et al., 2009, ApJ, 702, 745
  Rozo, E., Rykoff, E. S., Koester, B. P., et al., 2009, ApJ, 703, 601
  Rykoff, E. S., Koester, B. P., Rozo, E., et al., 2011, ApJ, 746, 178

"""
from __future__ import absolute_import, division, print_function

import emcee
import numpy as np
import pylab
import sys
import warnings
from itertools import count
from matplotlib import ticker
from scipy import integrate, optimize, stats
from scipy.special import erf
from time import sleep

if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
    xrange = range

# my code
from astro import cosmology
from plottools import plotutils
plotutils.update_rcParams()


## -------------------- ##
## -------------------- ##
##                      ##
##    Main functions    ##
##                      ##
## -------------------- ##
## -------------------- ##


def lambda_richness(z, R, mag, color, color_err, rs_zeropoint, rs_slope,
                    mstar, mlim, bg, mag_err=0, rs_sigma=0.05, rs_pivot=0,
                    alpha=-1.2, Ro=0.9, Rs=0.15, Rcore=0.1, sigma_R=0.05,
                    beta=0.2, h=0.7, maxiter=10000, tolerance=0.1,
                    full_output=True):
    """
    See Section 3 in Rykoff et al. (2012)

    Parameters
    ----------
        z       : float
                  cluster redshift
        R       : array of floats
                  galaxy distances to the cluster center
        mag     : array of floats
                  galaxy magnitudes
        color   : array of floats
                  galaxy colors
        color_err : array of floats
                  uncertainties on galaxy color
        rs_zeropoint : float
                  zero point of the red sequence
        rs_slope : float
                  slope of the red sequence
        mstar   : float
                  characteristic magnitude of the Schechter function
                  describing the luminosity function
        mlim    : float
                  limiting magnitude considered to calculate lambda
        bg      : array of floats
                  value of background for each galaxy's color and magnitude.
                  The grid (color vs. mag) from which these values were
                  calculated should be normalized such that its integral is
                  one.

    Optional parameters
    -------------------
        mag_err  : float
                  measurement uncertainties on magnitudes. If provided, they
                  are used to smooth the luminosity filter following Appendix
                  B of Rozo et al. (2015, MNRAS, 453, 38).
        rs_sigma : float
                  intrinsic scatter of the red sequence
        rs_pivot : float
                  pivot at which the red sequence is calculated (see equation
                  below)
        alpha   : float
                  faint-end slope of the Schechter luminosity function
        Ro      : float
                  normalization of the power law between lambda and cluster
                  size (Eq. 5 of Rykoff et al.)
        Rs      : float
                  scale radius of the NFW profile
        Rcore   : float
                  core radius, below which the density profile is kept
                  constant
        sigma_R : float
                  width of the smoothing kernel for the radial filter,
                  following Appendix B of Rozo et al. (2015, MNRAS, 453, 38).
                  Set to zero for a sharp edge.
        beta    : float
                  exponent of the power law between lambda and cluster size.
                  Note that setting beta=0 means using a constant cluster
                  size.
        h       : float
                  unitless Hubble constant, Ho/(100 km/s/Mpc)
        maxiter : float
                  maximum number of iterations for Newton's root finder
                  used to estimate lambda
        tolerance : float
                  tolerance for convergence of lambda and cluster size,
                  in units of delta(lambda) / lambda_err
        full_output : bool
                  see below

    Returns
    -------


    """
    Ro /= h
    Rs /= h
    Rcore /= h
    sigma_R /= h
    Cfilter = filter_color(mag, color, color_err, rs_zeropoint, rs_slope,
                           sigma=rs_sigma, magpivot=rs_pivot)
    Lfilter = filter_luminosity(mag, mstar, mlim, alpha=alpha)
    richness = _lambda(50., z, R, mag, mlim, Cfilter, Lfilter, bg,
                       mag_err=mag_err, Ro=Ro, Rs=Rs, Rcore=Rcore,
                       sigma_R=sigma_R, beta=beta, maxiter=maxiter)
    richness, px, Rfilter, Rc = richness
    i = 0
    if beta != 0:
        richness1 = 10000
        #median = np.median
        #std = np.std
        #nstart = 100
        #lambda_values = list(np.linspace(0, 100000, nstart))
        #richness_err = [0 for j in lambda_values]
        #while abs(richness - richness1) / richness > tolerance:
        lambda_values = []
        richness_err = []
        err = 1
        while abs(richness - richness1) / err > tolerance:
            richness1 = richness
            richness = _lambda(richness, z, R, mag, mlim, Cfilter, Lfilter,
                               bg, mag_err=mag_err, sigma_R=sigma_R, Ro=Ro,
                               Rs=Rs, Rcore=Rcore, beta=beta, maxiter=maxiter)
            richness, px, Rfilter, Rc = richness
            lambda_values.append(richness)
            err = ((px * (1-px)).sum())**0.5
            richness_err.append(err)
            i += 1
            # it seems for ACTPol clusters that by then things have converged
            # long ago
            if i == 1000:
                break
    richness_err = ((px * (1 - px)).sum())**0.5
    if full_output:
        return richness, richness_err, Rc, i, \
               (Cfilter, Lfilter, Rfilter, px)
    return richness, richness_err


def plot(rsg, pivot, mag, color, color_err=[], alpha=False, mu=False,
         sigma=False, rs=False, bcg=False, output='', comments='',
         fix_scatter=False, rsplot=True, mag_label='magnitude',
         color_label='color', ylim=False, verbose=True):
    """
    Plot the red sequence, along with the results of ECGMM when available

    Parameters
    ----------
        rsg      : array of ints or bool
                   indices of galaxies that belong to the red sequence,
                   as returned by rsgalaxies()
        pivot    : float
                   pivot magnitude for the red sequence
        mag      : array of floats
                   galaxy magnitudes
        color    : array of floats
                   galaxy color

    Optional arguments
    ------------------
        color_err : array of floats
                   uncertainty in galaxy color
        alpha    : list of floats
                   membership fraction for each population in color, as
                   identified by ECGMM
        mu       : list of floats
                   location(s) of the ECGMM population(s)
        sigma    : list of floats
                   width(s) of the ECGMM population(s)
        rs       : list of floats, length 2 or 3
                   red sequence parameters (zeropoint,slope[,scatter]). The
                   scatter is optional
        bcg      : tuple of length 2
                   magnitude and color of the BCG
        output   : str
                   name of the file where the plot will be saved. If not
                   provided, the plot will be shown on screen.
        comments : str
                   any comments to appear in the top-left corner of the plot,
                   such as cluster name and redshift
        fix_scatter : float
                   intrinsic scatter of the red sequence (used for
                   compatibility with the redsequence() wrapper)
        rsplot   : bool
                   whether to plot the red sequence line
        mag_label : str
                   xlabel of the red sequence plot
        color_label : str
                   ylabel of the red sequence plot
        ylim     : tuple of floats, length 2
                   y-axis limits of the plot. If not provided, are determined
                   automatically by matplotlib
        verbose  : bool
                   verbose

    """
    red = (1,0,0)
    blue = (0,0,1)
    yellow = (0.7,0.7,0)

    if len(color_err) == 0:
        color_err = np.zeros(mag.size)
    # plot limits
    if bcg is False:
        xmin = min(mag)
    else:
        xmin = min(min(mag), bcg[0])
    if xmin < 0:
        xmin = int(xmin) - 1.2
    else:
        xmin = int(xmin) - 0.2
    if fix_scatter:
        scatter = fix_scatter
    elif rs is not False:
        scatter = rs[2]

    # define figure
    fig = pylab.figure(figsize=(9,6))

    # CMD
    cmr = pylab.axes([0.12, 0.12, 0.65, 0.84])
    #cmr.set_xlabel('{0}-band magnitude'.format(mag_label))
    #cmr.set_ylabel('{0} color'.format(color_label))
    cmr.set_xlabel(mag_label)
    cmr.set_ylabel(color_label)
    # plot bcg if given in the proper format
    if bcg is not False:
        try:
            if len(bcg) == 2:
                cmr.plot(bcg[0], bcg[1], 'ko', mfc='none', mew=2, ms=9)
            else:
                raise ValueError('parameter bcg must have length 2')
        except ValueError:
            msg = 'parameter bcg must be an array-like of length 2'
            raise ValueError(msg)
    cmr.plot(mag, color, '.', color=blue)
    if len(rsg) > 0:
        cmr.errorbar(mag[rsg], color[rsg], yerr=color_err[rsg], fmt='.',
                     color=red, ecolor=red, elinewidth=1, mec=red, capsize=1)
    if comments:
        cmr.annotate(comments, xy=(0.05, 0.95), xycoords='axes fraction',
                     va='top', ha='left')
    # Initial color selection
    if ylim is False:
        bins = np.arange(min(color), max(color)+0.5, 0.5)
        hist = np.histogram(color, bins)[0]
        jo = np.argmax(hist)
        co = (bins[jo]+bins[jo+1]) / 2
        ylim = (co-1.5, co+1.5)

    # Red Sequence
    xticklabels = np.arange(int(xmin + pivot)-1, max(mag) + pivot, 1,
                               dtype=int)
    xticks = xticklabels - pivot
    xticklabels = ['${0}$'.format(xtl) for xtl in xticklabels]
    if rs is not False:
        if rs[1] >= 0:
            cmr.annotate(
                '%s = %.3f + %.3f (%s$-$%.2f) ($\sigma=%.2f$)' \
                    %(color_label.replace('$', ''), rs[0], rs[1],
                mag_label.replace('$', ''), pivot, scatter),
                xy=(0.05, 0.03), xycoords='axes fraction', fontsize=13)
        else:
            cmr.annotate(
                '%s = %.3f$ - $%.3f (%s$-$%.2f) ($\sigma=%.2f$)' \
                    %(color_label.replace('$', ''), rs[0], -rs[1],
                mag_label.replace('$', ''), pivot, scatter),
                xy=(0.05, 0.03), xycoords='axes fraction', fontsize=13)
        if rsplot:
            t = np.linspace(xticks[0], max(mag), 100)
            cmr.plot(t, rs[0] + rs[1]*t, 'k-', lw=2, zorder=-5)
            cmr.plot(t, rs[0] + rs[1]*t + scatter, 'k--', lw=2, zorder=-5)
            cmr.plot(t, rs[0] + rs[1]*t - scatter, 'k--', lw=2, zorder=-5)
    #cmr.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%s$'))
    cmr.set_xticks(xticks)
    cmr.set_xticklabels(xticklabels)
    cmr.set_xlim(xmin, max(mag))
    cmr.set_ylim(*ylim)

    # Color histogram
    ch = pylab.axes([0.77, 0.12, 0.2, 0.84])
    bins = np.arange(ylim[0], ylim[1] + 0.05, 0.05)
    n, bins, patches = ch.hist(color, bins=bins, histtype='stepfilled',
                               orientation='horizontal', fc=yellow)
    # the area of the histogram
    A_hist = n.sum() * (bins[1] - bins[0])
    # assuming that alpha, mu and sigma all have the same format
    tbin = 0.01
    t = np.arange(ylim[0], ylim[1] + tbin, tbin)
    if hasattr(mu, '__iter__'):
        # Red Sequence(s)
        if len(mu) == 1:
            f_rs = stats.norm.pdf(t, mu[0], sigma[0])
            f_rs = f_rs / (np.sqrt(2*np.pi) * sigma[0]**2)
            A_rs = f_rs.sum() * tbin
            pylab.plot(f_rs * (A_hist/A_rs), t, '-', color=red, lw=2)
        else:
            f_rs = np.zeros((len(mu)-1,t.size))
            A_rs = np.zeros((len(mu)-1,t.size))
            for i in xrange(1, len(mu)):
                f_rs[i-1] = stats.norm.pdf(t, mu[i], sigma[i])
                f_rs[i-1] /= np.sqrt(2*np.pi) * sigma[i]**2
                A_rs[i-1] = f_rs[i-1].sum() * tbin
            # Blue Cloud
            f_bc = stats.norm.pdf(t, mu[0], sigma[0]) / \
                   ((2*np.pi)**0.5 * sigma[0]**2)
            A_bc = f_bc.sum() * tbin
            # normalized so that the ratio of "areas" is equal to the ratio of
            # alpha's:
            if A_bc > 0:
                g_bc = f_bc * (alpha[0] * A_rs) / (sum(alpha[1:]) * A_bc)
                g_bc = np.squeeze(g_bc)
                A_bc = g_bc.sum() * tbin
            #pylab.plot(g_bc, t, '-', color=blue, lw=2)
            # the areas of the histograms and curves shall be the same
            ratio_areas = A_hist / (np.sum(A_rs, axis=0) + A_bc)
            for fi in f_rs:
                pylab.plot(fi*ratio_areas, t, '-', color=red, lw=2)
            if A_bc > 0:
                pylab.plot(g_bc*ratio_areas, t, '-', color=blue, lw=2)

    elif mu is not False: # will be a scalar
        f_rs = stats.norm.pdf(t, mu, sigma)
        f_rs = f_rs / (np.sqrt(2*np.pi) * sigma ** 2)
        A_rs = f_rs.sum() * tbin
        pylab.plot(g_rs * (A_hist/A_rs), t, '-', color=red, lw=2)

    # ticks and limits
    ch.set_ylim(*ylim)
    if max(xticks) == max(n):
        ch.set_xlim(0, max(n) + 2)
    else:
        ch.set_xlim(0, max(ch.get_xticks()))
    ch.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ch.yaxis.set_major_formatter(ticker.NullFormatter())

    # save?
    if output:
        try:
            pylab.savefig(output, format=output[-3:])
        except IOError:
            sleep(3)
            pylab.savefig(output, format=output[-3:])
        if verbose:
            print('  Saved to', output)
    else:
        pylab.show()
    pylab.close()
    return


def redsequence(mag, color, color_err=[], pivot=0, bcg=False, method='mle',
                fix_slope=False, fix_norm=False, fix_scatter=False,
                width=2, npoints=100, converge=True, make_plot=True,
                plot_output='', plot_comments='', mag_label='m',
                color_label='c', plot_ylim=False, do_ecgmm=True,
                mincolor=None, maxcolor=None, minmag=None, flavour='BIC',
                bootstrap=100, alpha=[0.2,0.8], mu=None, sigma=[1.0,0.05],
                verbose=True, debug=False):
    """
    Fit the red sequence. This is the main task; it is the only task in
    this file that should be called by itself for most purposes. All
    other functions are called upon by this one. All objects in the
    catalog will be used; therefore, all "cleaning"  (e.g., removing
    stars, choosing aperture) should be done prior to calling this
    function.

    Parameters
    ----------
      mag       : numpy array
                  Apparent magnitudes
      color     : numpy array
                  Colors
      color_err : numpy array
                  Uncertainties in colors
      pivot     : float (default 0)
                  magnitude pivot for the RS fit, such that
                  [color=A+B*(mag-pivot)]. The pivot MUST NOT have been
                  subtracted from the magnitudes
      bcg       : tuple of length 2 (optional)
                  the magnitude and color of the BCG
      method    : {'mle', 'wls', 'bayesian'} (default 'mle')
                  Method to use in calculating the best-fit RS. The bayesian
                  likelihood is calculated without priors on the RS, and is
                  currently much slower than the other two, which are
                  iterative. NOTE: 'bayesian' is not properly implemented.
      fix_slope : False or float
                  Fix the slope in the fit? If not False then it should be the
                  number to which the slope is fixed.
      fix_norm  : False or float
                  Fix the normalization in the fit? If not False then it
                  should be the number to which the slope is fixed.
      fix_scatter : False or float
                  Fix the scatter in the fit? If not False then it
                  should be the number to which the scatter is fixed.
      width     : float
                  number of times the scatter within which galaxies are still
                  considered part of the red sequence (i.e., a galaxy is
                  part of the red sequence if
                  abs(color_gal - color_RS) < width * scatter,
                  where scatter is the quadrature sum of the color uncertainty
                  and the intrinsic scatter and is calculated for every
                  galaxy.
      npoints   : int (default 100)
                  number of points sampled in the zero point and slope spaces
                  for the bayesian analysis. If method='wls', then this is the
                  number of bootstrap realisations for the errors on the
                  best-fit.
      converge  : boolean (default True)
                  iterate the RS algorithm until convergence? If false, will
                  return the results of the first fitting
      make_plot : boolean (default False)
                  plot the CMD with the resulting RS along with the color
                  histogram and results from ECGMM
      plot_output : str (optional)
                  output filename for the CMD plot, with the extension.
      plot_comments : str (optional)
                  a comment (e.g., name of the cluster), that will be placed
                  in the lower left corner of the plot.
      mag_label : str
                  magnitude band, used for the x-axis and for the RS formula
                  which is printed in the plot. The string will be printed in
                  math mode ($'s should not be included).
      color_label : str
                  color label (e.g., g-r), used for the y-axis and for the RS
                  formula which is printed in the plot. The string will be
                  printed in math mode ($'s should not be included).
      plot_ylim : list of length 2 (optional)
                  y-axis (i.e., color) limits for the plot. If False, then
                  limits are automatically determined from the color
                  distribution.
      do_ecgmm  : boolean (default True)
                  Use ECGMM to separate red from blue galaxies? If False, will
                  show the color magnitude and ask the user to provide color
                  cuts (lower and upper) and possibly a bright magnitude cut,
                  from which to fit the red sequence. It is advised that this
                  option be set (to False) for clusters that cannot be fit a
                  red sequence automatically. The user may also set the
                  parameter(s) `mincolor`, and optionally `maxcolor` and
                  `minmag`; if so the interactive query will be disabled.
      flavour   : {'AIC', 'BIC'}
                  which information criterion to use in ECGMM
      bootstrap : False or int
                  whether to bootstrap the ECGMM and, if so, how many
                  bootstrap samples to use
      alpha     : list of 2 floats
                  a guess on the fraction of objects in each sample
      mu        : list of 2 floats (optional)
                  a guess on the average color of each sample. If not
                  specified, will take the mode of the sample (which is
                  typically the red sequence) and the mode minus 1 magnitude
      sigma     : list of 2 floats
                  a guess on the width of each gaussian distribution

    Returns
    -------
      What is returned depends on the method chosen to calculate the RS. If
      method=='mle' or method=='wls', then the returned values are

        rsg     : numpy array
                  indices of all galaxies that belong to the RS after all
                  iterations
        Ars     : tuple of length 2
                  red sequence zero point and its uncertainty
        Brs     : tuple of length 2
                  red sequence slope and its uncertainty
        Srs     : tuple of length 2
                  red sequence intrinsic scatter (i.e., accounting for
                  measurement errors) and its uncertainty

      If the method is bayesian then the returned
      values are

        rsg     : numpy array
                  indices of all galaxies that belong to the RS after all
                  iterations
        L       : numpy array
                  Likelihood matrix of the zero point and slope of the RS

    """
    if not isinstance(fix_norm, bool) and isinstance(fix_norm, int):
        fix_norm = float(fix_norm)
    if not isinstance(fix_slope, bool) and  isinstance(fix_slope, int):
        fix_slope = float(fix_slope)
    if not isinstance(fix_scatter, bool) and isinstance(fix_scatter, int):
        fix_scatter = float(fix_scatter)

    if fix_scatter is not False:
        #if type(fix_scatter) not in (float, np.float64):
        if not isinstance(fix_scatter, float):
            msg = 'Error: parameter fix_scatter must either be False or'
            msg += 'a float between 0 and 1. Exiting.'
            raise ValueError(msg)
    array_equal = np.array_equal
    mag -= pivot
    if bcg is not False:
        bcg = list(bcg)
        bcg[0] -= pivot
    if verbose:
        print()
    # initialize errors if empty
    if color_err == []:
        color_err = 1e-5 * np.ones(len(mag))
    galaxies = np.arange(len(mag), dtype=int)
    # initialize mu?
    if mu is None and do_ecgmm:
        import ecgmmPy
        bins = np.arange(min(color), max(color)+0.01, 0.05)
        h, e = np.histogram(color, bins)
        co = e[np.argmax(h)]
        mu = [co-1, co]
        if verbose:
            print('  mu = ({0:.3f},{1:.3f})'.format(mu[0], mu[1]))

    # run ECGMM?
    if do_ecgmm:
        alpha, mu, sigma = _ecgmm(color, color_err, alpha, mu, sigma,
                                  flavour, bootstrap)
        rsg = rsgalaxies(mag, color, color_err, mu[1], sigma[1],
                         indices=galaxies, width=3)
        # this happens when ECGMM is not able to fit two gaussians, probably
        # because there are too few objects?
        if len(rsg) <= 5:
            alpha, mu, sigma = _ecgmm(color, color_err,
                                      alpha=[1], mu=[1.2], sigma=[0.05])
            rsg = rsgalaxies(mag, color, color_err, mu, sigma,
                             indices=galaxies, width=2)
        po = (max(mu), (fix_slope if isinstance(fix_slope, float) else -0.01))
        if verbose:
            print('mu =', mu)
            print('alpha =', alpha)
            print('sigma =', sigma)
    # select objects for red sequence manually
    else:
        alpha = False
        mu = False
        sigma = False
        if mincolor is None:
            comm = 'Choose a color range that roughly delimits\n' \
                   'red sequence galaxies. You will have\n' \
                   'to input it in the terminal. You can also give a third\n' \
                   'argument, which will be a minimum magnitude to use'
            plot(galaxies, pivot, mag, color, color_err, bcg=bcg, comments=comm,
                 mag_label=mag_label, color_label=color_label, ylim=plot_ylim,
                 fix_scatter=fix_scatter, verbose=verbose)
            msg = 'Write down lower and upper colors (space-separated): '
            yo = raw_input(msg)
            yo = [float(y) for y in yo.split()]
        else:
            yo = [mincolor]
            if maxcolor is not None:
                yo.append(maxcolor)
                if minmag is not None:
                    yo.append(minmag)
        if len(yo) == 1:
            rsg = galaxies[color > yo]
        elif len(yo) == 2:
            rsg = galaxies[(color > yo[0]) & (color < yo[1])]
        else:
            rsg = galaxies[(color > yo[0]) & (color < yo[1]) & \
                           (mag+pivot > yo[2])]
        if len(yo) == 1:
            mu = [yo[0] + 0.2]
        if len(yo) >= 2:
            mu = [(yo[0] + yo[1]) / 2]
        # just a guess: the mode of the color histogram
        else:
            n, x = np.histogram(color, np.arange(mu[0]-1,mu[0]+1,0.1))
            x = (x[:-1]+x[1:]) / 2
            mu = [x[np.argmax(n)]]
        if isinstance(fix_slope, float):
            po = [mu[0]]
        else:
            po = [mu[0], -0.01]

    #if method == 'bayesian':
        #t_zp = np.linspace(rs[0] - 0.5, rs[0] + 0.5, npoints)
        #t_sl = np.linspace(rs[1] - 0.5, rs[1] + 0.5, npoints)
        #L = fit_rs(rsg, mag, color, color_err, method='bayesian',
                    #t_zp=t_zp, t_sl=t_sl)
        #return rsg, L
    rs = fit_rs(
        rsg, mag, color, color_err, po=po, fix_slope=fix_slope,
        fix_norm=fix_norm, fix_scatter=fix_scatter, method=method)
    slope = (rs[1] if len(rs) >= 2 else fix_slope)
    sigma_int = (rs[2] if len(rs) >= 3 else fix_scatter)
    rsg = rsgalaxies(
        mag, color, color_err, rs[0], slope, indices=galaxies,
        sigma_int=sigma_int, width=width, fit='tilted')
    if fix_slope is False or fix_norm is False:
        if converge:
            rs1 = np.arange(1)
            nit = 0
            while not array_equal(rs1, rsg):
                rs1 = rsg
                rs = fit_rs(rsg, mag, color, color_err, fix_norm=fix_norm,
                            po=rs, fix_slope=fix_slope,
                            fix_scatter=fix_scatter, method=method)
                slope = (rs[1] if len(rs) >= 2 else fix_slope)
                sigma_int = (rs[2] if len(rs) >= 3 else fix_scatter)
                rsg = rsgalaxies(
                    mag, color, color_err, rs[0], slope, indices=rsg,
                    sigma_int=sigma_int, width=width, fit='tilted')
                nit += 1
            if verbose:
                print('  {0} iteration(s), final sample: {1} galaxies'.format(
                            nit, len(rsg)))
                print()
        else:
            slope = (rs[1] if len(rs) >= 2 else fix_slope)
            sigma_int = (rs[2] if len(rs) >= 3 else fix_scatter)
            rsg = rsgalaxies(
                mag, color, color_err, rs[0], rs[1], indices=rsg,
                sigma_int=sigma_int, fit='tilted', width=width)
            if verbose:
                print('  Only one iteration set: {0} galaxies'.format(
                        len(rsg)))
        # nice printing
        if debug:
            slope = (rs[1] if len(rs) >= 2 else fix_slope)
            if slope >=0:
                print('CMR : {0} = {1:.3f} + {2:.3f}({3} - {4:.2f})'.format(
                            color_label, rs[0], slope, mag_label, pivot))
            else:
                print('CMR : {0} = {1:.3f} - {2:.3f}({3} - {4:.2f})'.format(
                            color_label, rs[0], -slope, mag_label, pivot))
        # bootstrap errors
        a = []
        b = []
        s = []
        n = len(mag)
        while len(a) < npoints:
            j = np.random.random_integers(0, n-1, n)
            rsboot = fit_rs(
                j, mag, color, color_err, method=method, fix_slope=fix_slope,
                fix_scatter=fix_scatter, verbose=False)
            try:
                a.append(rsboot[0])
                if fix_slope is False:
                    b.append(rsboot[1])
                if fix_scatter is False:
                    s.append(rsboot[2])
            except TypeError:
                pass
        a = (rs[0], np.std(a))
        b = ((rs[1], np.std(b)) if len(rs) >= 2 else (fix_slope, 0.))
        s = ((rs[2], np.std(s)) if len(rs) >= 3 else (fix_scatter, 0.))
    # if both "fixes" are not False
    else:
        rsg = rsgalaxies(mag, color, color_err, fix_norm, fix_slope,
                         indices=rsg, fit='tilted', width=width)
        a = (fix_norm, 0)
        b = (fix_slope, 0)
        s = (rs[2], 0)
    if make_plot:
        plot(rsg, pivot, mag, color, color_err, alpha,
             mu, s, (a[0],b[0]), bcg=bcg, output=plot_output,
             #mu, sigma, rs, bcg=bcg, output=plot_output,
             comments=plot_comments, fix_scatter=fix_scatter,
             rsplot=True, mag_label=mag_label,
             color_label=color_label, ylim=plot_ylim,
             verbose=verbose)
    if verbose:
        print()
    return rsg, a, b, s


## ------------------------ ##
## ------------------------ ##
##                          ##
##    Filters for lambda    ##
##                          ##
## ------------------------ ##
## ------------------------ ##


def filter_color(mag, color, color_err, zp, slope, sigma=0.05, magpivot=0):
    """
    Color filter, Sec 3.3

    Parameters
    ----------
        mag       : numpy array
                    apparent magnitude of each galaxy
        color     : numpy array
                    color of each galaxy
        color_err : numpy array
                    error in the color of each galaxy
        zp        : float
                    zero point of the Color Magnitude Relation (CMR)
        slope     : float
                    slope of the CMR (including sign)
        sigma     : float (default 0.05)
                    intrinsic dispersion of the CMR
        magpivot  : float (default 0.0)
                    pivot of the scaling relation

    Returns
    -------
        cfilter   : numpy array
                    The value of the color filter for each galaxy

    """
    cmr = zp + slope * (mag - magpivot)
    sigma_total = (sigma**2 + color_err**2)**0.5
    Cfilter = np.exp(-(color - cmr)**2 / (2*sigma_total**2)) / \
              ((2*np.pi)**0.5 * sigma_total)
    return Cfilter


def filter_luminosity(mag, mstar, maglim, alpha=-1.2):
    """
    Luminosity filter, Sec 3.2.

    The luminosity filter is given by

        phi(m) = 10**(-0.4*(m-mstar)*(alpha+1) * exp(-10**(-0.4*(m-mstar)))

    Parameters
    ----------
        mag       : numpy array
                    apparent magnitude of each galaxy
        mstar     : float
                    characteristic Schechter apparent magnitude at the
                    redshift of the cluster
        maglim    : float
                    apparent magnitude cut at the faint end
        alpha     : float (default -1.2)
                    slope of the faint end of the luminosity function

    Returns
    -------
        lfilter   : numpy array
                    The value of the luminosity filter for each galaxy

    """
    # normalization
    f = lambda x: 10 ** (-0.4 * (x - mstar) * (alpha + 1)) * \
                  np.exp(-10 ** (-0.4 * (x - mstar)))
    A = 1 / integrate.quad(f, 0, maglim)[0]
    #Lfilter = 10 ** (-0.4 * (mag - mstar) * (alpha + 1))
    #Lfilter *= A * np.exp(-10 ** (-0.4 * (mag - mstar)))
    Lfilter = A * f(mag)
    Lfilter[mag > maglim] = 0
    return Lfilter


def filter_radius(R, Rc, Rs, Rcore):
    """
    Radial filter, Sec. 3.1.

    Parameters
    ----------
        R         : numpy array
                    distance from each galaxy to the center of the cluster
                    (e.g., the BCG), in Mpc
        Rc        : float
                    size of the cluster, in Mpc
        Rs        : float
                    scale radius of the cluster, in Mpc
        Rcore     : float
                    core radius of the cluster, in Mpc

    Returns
    -------
        filter    : numpy array
                    The value of the radial filter for each galaxy

    """
    def _f(x):
        """
        Auxiliary, Eq. 8 of Rykoff et al. (2012)
        """
        x[x == 1] = x[x == 1] + 1e-10
        y = np.zeros(x.shape)
        y[x > 1] = np.arctan(((x[x > 1] - 1) / (1 + x[x > 1]))**0.5)
        y[x < 1] = np.arctanh(((1 - x[x < 1]) / (1 + x[x < 1]))**0.5)
        return 1 - 2 * y / np.absolute(1 - x**2)**0.5
        #y = 1 - np.sign(x-1) * 2 * y / np.absolute(1 - x**2)**0.5
        #return y
    def _fscalar(x):
        if x == 1:
            x += 1e-10
        if x > 1:
            y = np.arctan(((x - 1) / (1 + x))**0.5)
            #return 1 - 2 * y / (x**2 - 1)**0.5
        elif x < 1:
            y = np.arctanh(((1 - x) / (1 + x))**0.5)
            #return 1 - 2 * y / (1 - x**2)**0.5
        return 1 - 2 * y / np.absolute(1 - x**2)**0.5

    x = R / Rs
    within_core = (R < Rcore)
    xcore = Rcore / Rs
    Rfilter = np.zeros(R.shape)
    # inside the core the filter is constant
    Rfilter[within_core] = _fscalar(xcore) / (xcore**2 - 1)
    j = (R <= Rc) & (R > Rcore)
    Rfilter[j] = _f(x[j]) / (x[j]**2 - 1)
    j = np.argsort(R)
    A = 1 / integrate.trapz(Rfilter[j], R[j])
    return A * Rfilter


def _lambda(lambda_value, z, R, mag, mlim, Cfilter, Lfilter, bx, mag_err=0,
            sigma_R=0.05, Ro=0.9, Rs=0.15, Rcore=0.1, beta=0.2, maxiter=1000):
    """
    bx is background per sq. deg.
    """
    Rc = Ro * (lambda_value/100.)**beta
    j = (R < Rc)
    # the radius in radians
    d = np.pi/180 * cosmology.dProj(z, Rc, input_unit='Mpc', unit='deg')
    # the prefactor (180/pi)**2 is to convert to sq. deg.
    area = (180/np.pi)**2 * 2*np.pi * (1 - np.cos(d/2.))
    Rfilter = filter_radius(R, Rc=Rc, Rs=Rs, Rcore=Rcore)
    ux = 2*np.pi*R*Rfilter * Cfilter * Lfilter
    ux[ux == 0] = 1e-100
    bg = 2 * np.pi * R * bx * area
    px = np.zeros(ux.shape)
    # smooth edges (Rozo et al. 2015, Appendix B)
    thetaL = 0.5 * (1 + erf((mlim - mag[j])/mag_err[j]))
    thetaR = 0.5 * (1 + erf((Rc - R[j])/sigma_R))
    #thetaL = thetaR = 1
    eq = lambda n: n - (n*ux[j] * thetaL * thetaR / (n*ux[j] + bg[j])).sum()
    try:
        richness = optimize.newton(eq, lambda_value, maxiter=maxiter)
    except RuntimeError:
        # doing this means that the loop
        print('Hit RuntimeError with maxiter={0}'.format(maxiter))
        return lambda_value, px, Rfilter, Rc
    #print('z={0:.2f} Rc={1:.2f} area={2:.4f} No={3:.1f} N={4:.1f}'.format(
                #z, Rc, area, lambda_value, richness))
    px[j] = richness * ux[j] / (richness * ux[j] + bg[j])
    return richness, px, Rfilter, Rc


## ------------------------------- ##
## ------------------------------- ##
##                                 ##
##    Auxiliary functions below    ##
##                                 ##
## ------------------------------- ##
## ------------------------------- ##


def _cmd_ylim(mu):
    cmax = np.ceil(mu)
    if np.ceil(mu) - mu < mu - np.floor(mu):
        cmax += 1
    cmin = cmax - 3
    return cmin, cmax


def _ecgmm(c, e, alpha=[0.2,0.8], mu=[0.5,1.2], sigma=[0.5,0.05],
           flavour='BIC', bootstrap=False):
    """
    Error-Corrected Gaussian Mixture Model (ECGMM) by Hao et al. (2009)

    Everything is coded in their script so here just execute it.

    Parameters
    ----------
        c       : float array
                  galaxy colours
        e       : float array
                  galaxy colour uncertainties
        alpha   : float array-like
                  initial guess for the mixing of the Gaussian models
        mu      : float array-like
                  initial guess for the locations of the Gaussians
        sigma   : float array-like
                  initial guess for the widths of the Gaussians
        flavour : {'AIC', 'BIC'}
                  which information criterion to use
        bootstrap : False or int
                  whether to bootstrap the ECGMM and, if so, how many
                  bootstrap samples to use

    """
    if bootstrap is not False:
        gmm = ecgmmPy.bsecgmm(c, e, alpha, mu, sigma, nboot=100,
                              InfoCriteria=flavour)
    elif flavour.upper() == 'AIC':
        gmm = ecgmmPy.aic_ecgmm(c, e, alpha, mu, sigma)
    elif flavour.upper() == 'BIC':
        gmm = ecgmmPy.bic_ecgmm(c, e, alpha, mu, sigma)
    return alpha, mu, sigma


def _fit_mle(x1, x2, x2err=[], x1err=[], fix_slope=False, fix_scatter=False,
             po=(1,0,0.1), verbose=False, full_output=False):
    """
    Maximum Likelihood Estimation of best-fit parameters

    For now I am forcing the slope to be lower than 0.02

    Parameters
    ----------
      x1, x2    : float arrays
                  the independent and dependent variables.
      x2err, x1err : float array (optional)
                  measurement uncertainties on galaxy colors and magnitudes
      fix_slope : False or float
                  if not False, should be the value to which it is fixed.
      fix_scatter : bool or float
                  whether to include intrinsic scatter in the MLE. If float,
                  then it is the fixed value of the intrinsic scatter
      po        : tuple of floats
                  initial guess for free parameters. If s_int is True, then
                  po must have 3 elements; otherwise it should have two
                  (for the zero point and the slope)

    Returns
    -------
      a         : float
                  Maximum Likelihood Estimate of the zero point.
      b         : float
                  Maximum Likelihood Estimate of the slope
      s         : float (optional, if s_int=True)
                  Maximum Likelihood Estimate of the intrinsic scatter

    """
    n = len(x1)
    if len(x2) != n:
        raise ValueError('x1 and x2 must have same length')
    if len(x1err) == 0:
        x1err = np.zeros(n)
    if len(x2err) == 0:
        x2err = np.zeros(n)
    if isinstance(fix_scatter, float):
        po = po[:2]
    if isinstance(fix_slope, float):
        po = po[:1]
        if fix_scatter is False:
            msg = '`fix_slope` is set, `fix_scatter` cannot be `False`.' \
                  ' Setting to 0.05.'
            warnings.warn(msg, Warning)

    # a straight line with a fixed or free slope
    #if fix_slope is False:
    f = lambda a, b: a + b*x1
    #else:
        #f = lambda a: a + fix_slope*x1

    # fix intrinsic scatter to chosen value
    if isinstance(fix_scatter, float):
        # also fix slope
        if isinstance(fix_slope, float):
            def _loglike(p):
                w = ((fix_slope*x1err)**2 + x2err**2 + fix_scatter**2)**0.5
                return 2*np.log(w).sum() \
                    + (((x2-f(p[0],fix_slope)) / w)**2).sum() \
                    + np.log(n*(2*np.pi)**2) / 2
        else:
            def _loglike(p):
                w = lambda b: ((b*x1err)**2 + x2err**2 + fix_scatter**2)**0.5
                return 2*np.log(w(p[1])).sum() \
                    + (((x2-f(*p)) / w(p[1]))**2).sum() \
                    + np.log(n*(2*np.pi)**2) / 2
    # if intrinsic scatter is free then everything else is
    # free as well.
    else:
        def _loglike(p):
            w = lambda b, s: ((b*x1err)**2 + x2err + s**2)**0.5
            return 2*np.log(w(p[1:])).sum() \
                + (((x2-f(*p[:2])) / w(p[1:]))**2).sum() \
                + np.log(n*(2*np.pi)**2) / 2

    out = optimize.fmin(_loglike, po, disp=verbose, full_output=full_output)
    return out


def fit_rs(rsg, mag, c, e, max_e=0.25, fix_slope=False, fix_norm=False,
           fix_scatter=False, t_zp=None, t_sl=None, method='mle', plim=5,
           po=(1.5,-0.02,0.05), verbose=True):
    """
    Fit the RS as a straight line to all galaxies selected (rsg).

    plim is the plot limits, in units of sigma (see below)
    p0 are the initial guesses for zero-point and slope

    NOTES
    -----
      Only objects with color uncertainties less than max_e are included in
      the fit
    """
    n = len(rsg)
    if fix_norm is not False and fix_slope is not False:
        # (total) scatter -- taken from Pratt et al.
        scatter = (c - (fix_norm + fix_slope * mag)).sum()**2 / (n-1)
        # intrinsic scatter
        scatter = scatter - ((e / c) ** 2).sum() / n
        return fix_norm, fix_slope, np.sqrt(scatter)
    mag = mag[rsg]
    c = c[rsg]
    e = e[rsg]
    if method == 'wls':
        if fix_slope is False and fix_norm is False:
            cmr = lambda p, x: p[0] + p[1] * x
        elif fix_norm is not False:
            # to make life easier, no need to change the format of po
            if hasattr(po, '__iter__'):
                po = po[-1]
            cmr = lambda p, x: fixed[1] + p * x
        elif fix_slope is not False:
            # to make life easier, no need to change the format of po
            if hasattr(po, '__iter__'):
                po = po[0]
            cmr = lambda p, x: p + fixed[1] * x
        else:
            msg = 'fixed can only take the vaues "norm" and "slope"'
            msg += ' as the first argument; see help page.'
            raise ValueError(msg)
        cmr_err = lambda p, x, y, dy: (y - cmr(p, x)) / dy
        # just in case
        e[e == 0] = 1e-5
        good = np.arange(len(mag))[e < max_e]
        out = optimize.leastsq(cmr_err, po, args=(mag[good],c[good],e[good]),
                               full_output=1)
        rs = out[0]
        # run one last time to allow a small variation
        #if fix_slope is not False:
            #if fixed[0] == 'norm':
                #rs = [fixed[1], rs]
            #elif fixed[0] == 'slope':
                #rs = [rs, fixed[1]]
            #else:
                #msg = 'fixed can only take the vaues "norm" and "slope"'
                #msg += ' as the first argument; see help page.'
                #raise ValueError(msg)
        if fix_scatter is False:
            # (total) scatter -- taken from Pratt et al.
            scatter = ((c - (rs[0] + rs[1]*mag))**2).sum() / (n - 1)
            # intrinsic scatter
            scatter = (scatter - ((e/c)**2).sum() / n)**0.5
        else:
            scatter = fix_scatter
        return rs[0], rs[1], scatter

    elif method == 'mle':
        mle = _fit_mle(
            mag, c, x2err=e, fix_slope=fix_slope, fix_scatter=fix_scatter,
            po=po)
        return mle

    elif method == 'bayesian':
        if verbose:
            print('  Calculating Likelihoods...')
        #L = np.zeros((t_zp.size,t_sl.size))
        #for i in xrange(t_zp.size):
            #for j in xrange(t_sl.size):
                #L[i][j] = np.prod(
                    #_likelihood(mag, c, t_zp[i], t_sl[j], e/c))
        ## marginalized distributions:
        #if verbose:
            #print('  Marginalizing...')
        #zp = np.sum(L, axis=1)
        #zp = zp / zp.sum() # normalized
        #zp_peak = t_zp[np.argmax(zp)]
        #zp_err = np.std(zp)
        #sl = np.sum(L, axis=0)
        #sl = sl / sl.sum()
        #sl_peak = t_sl[np.argmax(sl)]
        #sl_err = np.std(sl)
        #if verbose:
            #print('sl = {0:6.3f} +/- {1:.3f}'.format(sl_peak, sl_err))
            #print('zp = {0:6.3f} +/- {1:.3f}'.format(zp_peak, zp_err))
            #print('cov:')
            #print(np.cov(sl, zp))
        #return L
        sampler = emcee.EnsembleSampler

    return


def _likelihood(mag, c, zp, sl, e=1):
    """
    e is the fractional error on the color
    """
    line = zp + sl * mag
    n = np.sqrt(2*np.pi) * e * line
    p = np.exp(-(c - line) ** 2 / (2 * (e*line) ** 2))
    return p / n


def _lnprob(theta, mag, color, color_err=0.1, priors=None):
    #zp, sl = theta
    #if priors is not None:
    return


def rsgalaxies(mag, color, color_err, mu, sigma, indices=None, width=2,
               sigma_int=0.05, fit='flat'):
    """
    Uses the fit to the RS from fit_rs and selects all galaxies within
    width*sigma_tot.

    If fit == 'tilted', then mu and sigma are the zero point and slope of the
    CMR, respectively.

    Returns the indices of galaxies within the red sequence

    """
    if indices is None:
        indices = np.arange(mag.size)
    w = (sigma_int**2 + color_err[indices]**2)**0.5
    if fit == 'flat':
        rsg = indices[np.absolute(color[indices] - mu) < width*w]
    elif fit == 'tilted':
        # the location of the RS at the magnitude of each galaxy:
        cmr = mu + sigma*mag[indices]
        rsg = indices[np.absolute(color[indices] - cmr) < width*w]
    return rsg

