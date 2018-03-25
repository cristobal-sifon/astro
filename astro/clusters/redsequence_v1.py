#!/usr/bin/env python
"""
Red Sequence fitting procedure

Fits the RS iteratively, first using ECGMM (Hao et al. 2009) to define the
initial sample to fit a RS to, and then doing a weighted least squares fit to a
straight line. Both these processes are iterated until the number of galaxies
considered converges.

Has the option to plot the CMD along with the results of ECGMM. All objects in
the catalog will be used; therefore, all "cleaning" (e.g., removing stars, 
choosing aperture) should be done prior to calling this script.

Note
----
  It is strongly adviced that a cut in magnitude be performed, where only
  objects fainter than the BCG, including it, are included. Here it is assumed
  that the BCG is the brightest object in the catalog, and the BCG is not
  included in the fits to the RS. If the catalog does contain objects brighter
  than the BCG, this may cause a non-optimal fitting.

** ECGMM WILL FAIL IF THERE ARE TOO FEW (<~15) OBJECTS **
WHAT CAN BE DONE:
          create an option that if ECGMM fails, the CMD is plotted and the user
          can input the cut between red and blue, and then use that for RS
          estimation (the user should select an upper and lower border for the
          RS that mimics the 2sigma selection).

References
----------
  Hao, J., Koester, B. P., McKay, T. A., et al., 2009, ApJ, 702, 745
  Rozo, E., Rykoff, E. S., Koester, B. P., et al., 2009, ApJ, 703, 601
  Rykoff, E. S., Koester, B. P., Rozo, E., et al., 2011, ApJ, 746, 178

"""

import ecgmmPy
import numpy
import pylab
import scipy
from itertools import count, izip
from matplotlib.ticker import NullFormatter
from scipy import stats, optimize

def fit(mag, color, color_err=[], pivot=0, bcg=False, method='wls',
        bayesian=False, fix_slope=False, fix_norm=False,
        npoints=100, converge=True, plot=True, plot_output='',
        plot_comments='', mag_label='m', color_label='c', plot_ylim=(-1, 2),
        do_ecgmm=True, alpha=[0.2, 0.8], mu=[0.5, 1.2], sigma=[0.5, 0.05],
        verbose=True, debug=False):
  """
  Fit the red sequence. This is the main task; it is the only task in this file
  that should be called by itself. All other functions are called upon by this
  one. All objects in the catalog will be used; therefore, all "cleaning" 
  (e.g., removing stars, choosing aperture) should be done prior to calling this
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
    method    : {'wls', 'bayesian'} (default 'wls')
                Method to use in calculating the best-fit RS. The bayesian 
                likelihood is calculated without priors on
                the RS, and is currently much slower than the WLS. The latter
                is iterative
    bayesian  : boolean (default False)
                If the RS is estimated through the WLS, the bayesian likelihood
                can be calculated only on the final sample. This usually gives 
                the same result as the WLS, but is a good way of estimating 
                errors on the zero point and slope of the RS.
    fix_slope : False or float
                Fix the slope in the fit? If not False then it should be the
                number to which the slope is fixed.
    fix_norm  : False or float
                Fix the normalization in the fit? If not False then it should
                be the number to which the slope is fixed.
    npoints   : int (default 100)
                number of points sampled in the zero point and slope spaces for
                the bayesian analysis. If method='wls', then this is the number
                of bootstrap realisations for the errors on the best-fit.
    converge  : boolean (default True)
                iterate the RS algorithm until convergence? If false, will
                return the results of the first fitting
    plot      : boolean (default False)
                plot the CMD with the resulting RS along with the color
                histogram and results from ECGMM
    plot_output : str (optional)
                output filename for the CMD plot, with the extension.
    plot_comments : str (optional)
                a comment (e.g., name of the cluster), that will be placed in
                the lower left corner of the plot.
    mag_label : str
                magnitude band, used for the x-axis and for the RS formula
                which is printed in the plot. The string will be printed in
                math mode ($'s should not be included).
    color_label : str
                color label (e.g., g-r), used for the y-axis and for the RS
                formula which is printed in the plot. The string will be
                printed in math mode ($'s should not be included).
    plot_ylim : list of length 2 (default (-1, 2))
                y-axis (i.e., color) limits for the plot
    do_ecgmm  : boolean (default True)
                Use ECGMM to separate red from blue galaxies? If False, will
                fit a straight line to the given sample, where alpha and mu
                are the initial guesses for zero-point (at mag=pivot) and
                slope

  Returns
  -------
    What is returned depends on the method chosen to calculate the RS. If the
    method is WLS, then the returned values are

      rsg     : numpy array
                indices of all galaxies that belong to the RS after all
                iterations
      rs      : tuple
                zero point and slope of the best-fit RS

    If the method is bayesian, or if bayesian=True, then the returned values
    are

      rsg     : numpy array
                indices of all galaxies that belong to the RS after all
                iterations
      L       : numpy array
                Likelihood matrix of the zero point and slope of the RS

  """
  mag -= pivot
  if bcg is not False:
    bcg = list(bcg)
    bcg[0] -= pivot

  if verbose:
    print()
  if color_err == []:
    color_err = 1e-5 * scipy.ones(len(mag))
  galaxies = scipy.arange(len(mag), dtype=int)

  if do_ecgmm:
    alpha, mu, sigma = ecgmm(color, color_err, alpha, mu, sigma)
    rsg = rsgalaxies(galaxies, mag, color, color_err, mu[1], sigma[1])

    # this happens when ECGMM is not able to fit two gaussians, probably
    # because there are too few objects
    if len(rsg) <= 5:
      alpha, mu, sigma = ecgmm(color, color_err,
                              alpha=[1], mu=[1.2], sigma=[0.05])
      rsg = rsgalaxies(galaxies, mag, color, color_err, mu, sigma)
  # NOT IMPLEMENTED
  #else:

  if method == 'wls':

    if plot:
      RSplot(bcg, rsg, pivot, mag, color, color_err, alpha, mu, sigma, rs,
             output=plot_output, comments=plot_comments, rsplot=True,
             mag_label=mag_label, color_label=color_label, ylim=plot_ylim,
             verbose=verbose)

    if bayesian:
      t_zp = scipy.linspace(rs[0] - 0.5, rs[0] + 0.5, npoints)
      t_sl = scipy.linspace(rs[1] - 0.5, rs[1] + 0.5, npoints)
      L = fit_rs(rsg, mag, color, color_err, method='bayesian',
                 t_zp=t_zp, t_sl=t_sl)
      return rsg, L

    if verbose:
      print()
    return rsg, a, b, s

  # NOT COMPLETE
  elif method == 'bayesian':
    a, b, cov, L = bayesfit(mag, color, color_err, zero=zero, slope=slope,
                            zero_range=zero_range, slope_range=slope_range,
                            full_output=full_output)
    #rsg = rsgalaxies

def rsgalaxies(gals, mag, c, e, mu, sigma, width=2, sigma_int=0.05,
               fit='flat'):
  """
  Uses the fit to the RS from fit_rs and selects all galaxies within width * 
  sigma_tot.

  If fit == 'tilted', then mu and sigma are the zero point and slope of the CMR, respectively, and
  the intrinsic scatter in the CMR is set by default to 0.05, as in Rykoff et al. (2012). This value should be
  checked for high-z clusters, however.
  """
  if fit == 'flat':
    w = scipy.sqrt((mu * e[gals]) ** 2 + (c[gals] * sigma) ** 2)
    rsg = gals[abs(c[gals] - mu) < width * w]
  elif fit == 'tilted':
    w = scipy.sqrt(sigma_int ** 2 + e[gals] ** 2)
    # the location of the RS at the magnitude of each galaxy:
    cmr = mu + sigma * mag[gals]
    rsg = gals[abs(c[gals] - cmr) < width * w]

  return rsg

def wlsfit():
  """
  Fit the RS with a WLS
  """
  rs = fit_rs(rsg, mag, color, color_err,
              fix_slope=fix_slope, fix_norm=fix_norm, method=method)
  if fix_slope is False or fix_norm is False:
    if converge:
      rs1 = []
      nit = 0
      while len(rs1) != len(rsg):
        rs1 = rsg
        rsg = rsgalaxies(rs1, mag, color, color_err, rs[0], rs[1],
                        fit='tilted')
        rs = fit_rs(rsg, mag, color, color_err, fix_norm=fix_norm,
                    fix_slope=fix_slope, method='wls')
        nit += 1
      if verbose:
        print('%d iteration(s), final sample: %d galaxies' %(nit, len(rsg)))
    else:
      if verbose:
        print('Only one iteration set: %d galaxies' %len(rsg))
    # nice printing
    if debug:
      if rs[1] >=0:
        print('CMR : %s = %.3f + %.3f(%s - %.2f)' \
              %(color_label, rs[0], rs[1], mag_label, pivot))
      else:
        print('CMR : %s = %.3f - %.3f(%s - %.2f)' \
              %(color_label, rs[0], -rs[1], mag_label, pivot))

    a = scipy.zeros(npoints)
    b = scipy.zeros(npoints)
    s = scipy.zeros(npoints)
    n = len(mag)
    for i in xrange(npoints):
      j = scipy.random.random_integers(0, n - 1, n)
      rsboot = fit_rs(j, mag, color, color_err, method='wls')
      a[i] = rsboot[0]
      b[i] = rsboot[1]
      s[i] = rsboot[2]
    a = (rs[0], scipy.std(a))
    b = (rs[1], scipy.std(b))
    s = (rs[2], scipy.std(s))

  # if both "fixes" are not False
  else:
    rsg = rsgalaxies(rsg, mag, color, color_err, fix_norm, fix_slope,
                      fit='tilted')
    a = (fix_norm, 0)
    b = (fix_slope, 0)
    s = (rs[2], 0)
  return

def bayesfit(mag, c, e, zero=None, slope=None, scatter=None,
             zero_range=(0,2), slope_range=(-0.1,0.05),
             scatter_range=(0,0.2), full_output=False):
  """
  Fit the RS with a bayesian approach

  Parameters
  ----------
    mag       : numpy array
                Apparent magnitudes
    c         : numpy array
                Colors
    e         : numpy array
                Uncertainties in colors
    zero      : float tuple of length 2 (optional)
                the peak and standard deviation of the gaussian prior on the
                zero point (at magnitude=pivot). If not given, then a flat
                prior is assumed
    slope     : float tuple of length 2 (optional)
                the peak and standard deviation of the gaussian prior on the
                slope of the red sequence. If not given, then a flat prior is
                assumed
    scatter   : float tuple of length 2 (optional)
                peak and standard deviation of the gaussian prior on the
                scatter of the red sequence. If not given, a flat prior is
                assumed
    zero_range : tuple of length 2 (default (0,2))
                possible range for the zero point. Also assumed as the flat
                prior range if zero is not given
    slope_range : tuple of length 2 (default (-0.1,0.05))
                possible range for the slope. Also assumed as the flat prior
                range if slope is not given
    scatter_range : 
                
    full_output : boolean (default False)
                If True, also return the full posterior likelihood matrix

  Returns
  -------
    a         : tuple of 2 floats
                central value and uncertainty on the zero point of the red
                sequence at the pivot magnitude
    b         : tuple of 2 floats
                central value and uncertainty on the slope of the red sequence
    cov       : 2 x 2 array
                covariance matrix of a and b
    L         : 2 x N array (optional)
                the full (a,b) likelihood array

  """
  if verbose:
    print('  Calculating Likelihoods...')
  t_zp = scipy.linspace(zero_range[0], zero_range[1], 100)
  t_sl = scipy.linspace(slope_range[0], slope_range[1], 100)
  L = scipy.zeros((len(t_zp), len(t_sl)))
  if zero is None:
    p_zp = t_zp
  else:
    p_zp = stats.norm.pdf(t_zp, zero[0], zero[1])
  if slope is None:
    p_sl = t_sl
  else:
    p_sl = stats.norm.pdf(t_sl, slope[0], slope[1])
  for i in xrange(len(t_zp)):
    for j in xrange(len(t_sl)):
      L[i][j] = scipy.prod(likelihood(mag, c, p_zp[i], p_sl[j], e/c,
                                      log=False))

  # marginalized distributions:
  if verbose:
    print('  Marginalizing...')
  zp = scipy.sum(L, axis=1)
  zp = zp / scipy.sum(zp) # normalized
  zp_peak = t_zp[scipy.argmax(zp)]
  zp_err = scipy.std(zp)
  sl = scipy.sum(L, axis=0)
  sl = sl / scipy.sum(sl)
  sl_peak = t_sl[scipy.argmax(sl)]
  sl_err = scipy.std(sl)
  A = (zp_peak, zp_err)
  B = (sl_peak, sl_err)
  cov = scipy.cov(sl, zp)
  if verbose:
    print('zp = %6.3f +/- %.3f' %A)
    print('sl = %6.3f +/- %.3f' %B)
    print('cov:', cov)

  if full_output:
    return A, B, cov, L
  return A, B, cov

def fit_rs(rsg, mag, c, e, max_e=0.25, fix_slope=False, fix_norm=False,
           t_zp=None, t_sl=None, method='wls', plim=5, p0=(2, -0.02)):
  """
  Fit the RS as a straight line to all galaxies selected (rsg), excluding the BCG, which is assumed to be the brightest
  object in the sample.

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
    scatter = sum((c - (fix_norm + fix_slope * mag)) ** 2) / (n - 1)
    # intrinsic scatter
    scatter = scatter - sum((e / c) ** 2) / n
    return fix_norm, fix_slope, scipy.sqrt(scatter)

  mag = mag[rsg]
  c = c[rsg]
  e = e[rsg]
  #bcg = scipy.argmin(mag)

  if method == 'wls':
    if fix_slope is False and fix_norm is False:
      cmr = lambda p, x: p[0] + p[1] * x
    elif fix_norm is not False:
      # to make life easier, no need to change the format of p0
      if type(p0) in (tuple, list, numpy.ndarray):
        p0 = p0[-1]
      cmr = lambda p, x: fixed[1] + p * x
    elif fix_slope is not False:
      # to make life easier, no need to change the format of p0
      if type(p0) in (tuple, list, numpy.ndarray):
        p0 = p0[0]
      cmr = lambda p, x: p + fixed[1] * x
    else:
      msg = 'fixed can only take the vaues "norm" and "slope" as the first'
      msg += ' argument; see help page.'
      raise ValueError(msg)
    cmr_err = lambda p, x, y, dy: (y - cmr(p, x)) / dy
    # just in case
    e[e == 0] = 1e-5
    good = scipy.arange(len(mag))[e < max_e]
    out = optimize.leastsq(cmr_err, p0, args=(mag[good],c[good],e[good]),
                           full_output=1)
    rs = out[0]
    # run one last time to allow a small variation
    #if fix_slope is not False:
      #if fixed[0] == 'norm':
        #rs = [fixed[1], rs]
      #elif fixed[0] == 'slope':
        #rs = [rs, fixed[1]]
      #else:
        #msg = 'fixed can only take the vaues "norm" and "slope" as the first'
        #msg += ' argument; see help page.'
        #raise ValueError(msg)
    # (total) scatter -- taken from Pratt et al.
    scatter = sum((c - (rs[0] + rs[1] * mag)) ** 2) / (n - 1)
    # intrinsic scatter
    scatter = scatter - sum((e / c) ** 2) / n

    return rs[0], rs[1], scipy.sqrt(scatter)

  elif method == 'bayesian':
    if verbose:
      print('  Calculating Likelihoods...')
    L = scipy.zeros((len(t_zp), len(t_sl)))
    for i in range(len(t_zp)):
      for j in range(len(t_sl)):
        L[i][j] = scipy.prod(likelihood(r, c, t_zp[i], t_sl[j], e/c,
                                        log=False))

    # marginalized distributions:
    if verbose:
      print('  Marginalizing...')
    zp = scipy.sum(L, axis=1)
    zp = zp / scipy.sum(zp) # normalized
    zp_peak = t_zp[scipy.argmax(zp)]
    zp_err = scipy.std(zp)
    sl = scipy.sum(L, axis=0)
    sl = sl / scipy.sum(sl)
    sl_peak = t_sl[scipy.argmax(sl)]
    sl_err = scipy.std(sl)
    if verbose:
      print('sl = %6.3f +/- %.3f' %(sl_peak, sl_err))
      print('zp = %6.3f +/- %.3f' %(zp_peak, zp_err))
      print('cov:')
      print(scipy.cov(sl, zp))

    return L
  return


def likelihood(r, c, zp, sl, e=1, log=True):
  """
  e is the fractional error on the color
  """
  line = zp + sl * r
  if log:
    n = -scipy.log(scipy.sqrt(2 * scipy.pi) * e * line)
    p = -(c - line) ** 2 / (2 * (e * line) ** 2)
    return n + p
  else:
    n = scipy.sqrt(2 * scipy.pi) * e * line
    p = scipy.exp(-(c - line) ** 2 / (2 * (e * line) ** 2))
    return p / n

def get_levels(L):
  """
  not finished
  """
  Lx, Ly = L.shape
  s = []
  while len(s) < 0.68 * Lx * Ly:
    s.append(L.max())
  return

def RSplot(bcg, rsg, pivot, mag, c, e, alpha, mu, sigma, rs, output='',
           comments='', rsplot=True, mag_label='m', color_label='c', 
           ylim=(-1, 2), verbose=True):
  """
  Plot the RS, along with the results of ECGMM

  Parameters
  ----------
    rsg
  
  rs is a 2-element vector containing the slope and zero-point of the
  red sequence
  """
  # plot limits
  if bcg is False:
    xmin = min(mag)
  else:
    xmin = min(min(mag), bcg[0])
  if xmin < 0:
    xmin = int(xmin) - 1
  else:
    xmin = int(xmin)

  # define figure
  pylab.figure(figsize=(12,8))

  # CMD
  cmr = pylab.axes([0.1, 0.1, 0.6, 0.85])
  cmr.set_xlabel('$%s$' %mag_label, fontsize=18)
  cmr.set_ylabel('$%s$' %color_label, fontsize=18)
  # plot bcg if given in the proper format
  if bcg is not False:
    try:
      if len(bcg) == 2:
        pylab.plot(bcg[0], bcg[1], 'ko', mec='k', mfc='none', mew=2, ms=9)
      else:
        raise ValueError('parameter bcg must have length 2')
    except ValueError:
      raise ValueError('parameter bcg must be an array-like of length 2')
  pylab.plot(mag, c, 'b.')
  if len(rsg) > 0:
    pylab.errorbar(mag[rsg], c[rsg], yerr=e[rsg], fmt='r.', ecolor='r')

  if comments:
    cmr.annotate(comments, xy=(0.1, 0.1), xycoords='axes fraction',
                 va='bottom', fontsize=18)

  # Red Sequence
  if len(rs) == 3:
    if rs[1] >= 0:
      pylab.annotate('$%s=%.3f+%.3f(%s-%.2f)$ $(\sigma=%.2f)$' \
                     %(color_label, rs[0], rs[1], mag_label, pivot, rs[2]),
                     xy=(0.05, 0.03), xycoords='axes fraction',
                     fontsize=20)
    else:
      pylab.annotate('$%s=%.3f-%.3f(%s-%.2f)$ $(\sigma=%.2f)$' \
                     %(color_label, rs[0], -rs[1], mag_label, pivot, rs[2]),
                     xy=(0.05, 0.03), xycoords='axes fraction',
                     fontsize=20)
  xticklabels = scipy.arange(int(xmin + pivot), max(mag) + pivot, 0.5)
  xticks = xticklabels - pivot
  if rsplot:
    t = scipy.linspace(xticks[0], max(mag), 100)
    pylab.plot(t, rs[0] + rs[1] * t, 'k-')
    pylab.plot(t, rs[0] + rs[1] * t + rs[2], 'k--')
    pylab.plot(t, rs[0] + rs[1] * t - rs[2], 'k--')
  cmr.set_xticks(xticks)
  cmr.set_xticklabels(xticklabels)
  cmr.set_xlim(xmin, max(mag))
  cmr.set_ylim(ylim[0], ylim[1])

  # Color histogram
  ch = pylab.axes([0.7, 0.1, 0.25, 0.85])
  bins = scipy.arange(ylim[0], ylim[1] + 0.05, 0.05)
  n, bins, patches = ch.hist(c, bins=bins,
                             orientation='horizontal', fc='g')

  t = scipy.arange(ylim[0], ylim[1] + 0.01, 0.01)
  if type(mu) in (list, tuple, numpy.ndarray):
    # Red Sequence(s)
    for i in range(1, len(mu)):
      f_rs = stats.norm.pdf(t, mu[i], sigma[i])
      f_rs = f_rs / (scipy.sqrt(2 * scipy.pi) * sigma[i] ** 2)
      g_rs = f_rs / max(f_rs) * max(n) # normalized to the peak
      A_rs = sum(g_rs) # A version of the area
      pylab.plot(g_rs, t, 'r-')
    if len(mu) > 1:
      # Blue Cloud
      f_bc = stats.norm.pdf(t, mu[0], sigma[0]) / \
             (scipy.sqrt(2 * scipy.pi) * sigma[0] ** 2)
      A_bc = sum(f_bc)
      # normalized so that the ratio of "areas" is equal to the ratio of
      # alpha's:
      g_bc = f_bc * (alpha[0] * A_rs) / (sum(alpha[1:]) * A_bc)
      pylab.plot(g_bc, t, 'b-')
  else: # will be a scalar
    f_rs = stats.norm.pdf(t, mu, sigma)
    f_rs = f_rs / (scipy.sqrt(2 * scipy.pi) * sigma ** 2)
    g_rs = f_rs / f_rs * max(n) # normalized to the peak
    pylab.plot(g_rs, t, 'r-')

  # ticks and limits
  ch.set_ylim(ylim[0], ylim[1])
  xticks = ch.get_xticks()
  if max(xticks) < 30:
    step = 5
  else:
    step = 10
  ch.set_xticks(scipy.arange(step, max(xticks) + step, step))
  if max(xticks) == max(n):
    ch.set_xlim(0, max(n) + 2)
  else:
    ch.set_xlim(0, max(ch.get_xticks()))
  ch.yaxis.set_major_formatter(NullFormatter())

  # save?
  if output:
    pylab.savefig(output, format = output[-3:])
    if verbose:
      print('Saved to', output)
  else:
    pylab.show()
  pylab.close()
  return

def cmd_ylim(mu):
  if scipy.ceil(mu) - mu < mu - scipy.floor(mu):
    cmax = scipy.ceil(mu) + 1
  else:
    cmax = scipy.ceil(mu)
  cmin = cmax - 3
  return cmin, cmax

def ecgmm(c, e, alpha = [0.2, 0.8], mu = [0.5, 1.2], sigma = [0.5, 0.05]):
  """
  Error-Corrected Gaussian Mixture Model (ECGMM) by Hao et al. (2009)
  
  Everything is coded in their script so here just execute it.
  """
  bic = ecgmmPy.bic_ecgmm(c, e, alpha, mu, sigma)
  return alpha, mu, sigma
