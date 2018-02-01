#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
#matplotlib.use('pdf')
import argparse
import glob
import lnr
import numpy
import os
import pylab
import sys
from astropy.io import fits
from itertools import count, izip
from matplotlib import cm, colors as mplcolors, rcParams, ticker
from numpy import arange, array, linspace, log10, logspace
from scipy import optimize, stats
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy

# Until matplotlib is upgraded to >=1.5
import colormaps

from kids_ggl_pipeline.halomodel import hm_utils, nfw#, utils
from kids_ggl_pipeline.sampling import sampling_utils

## local
sys.path.append(os.getcwd())
try:
    from literature import ihod, leauthaud12
except ImportError:
    leauthaud12 = None # (?)
import models
#import nfw
import utils
#from utils import read_avgs

# my code
import plottools
import readfile
import stattools
from astro.clusters import conversions, profiles

from astro import cosmology
cosmology.h = 1
cosmology.Omega_M = 0.315
cosmology.Omega_L = 0.685
h = cosmology.h
Om = cosmology.Omega_M
Ol = cosmology.Omega_L

plottools.update_rcParams()

#for key in rcParams:
    #if 'math' in key or 'tex' in key:
        #print '{0:<40s}  {1}'.format(key, rcParams[key])
#exit()

red = (1,0,0)
green = (0.2,0.8,0)
blue = (0,0,1)
yellow = (1,1,0.2)
magenta = (1,0.4,0.6)
cyan = (0,0.9,0.9)
orange = (1,0.7,0)
purple = (0.8,0,0.4)
brown = (0.6,0.3,0)

"""
TODO:
    -

"""


def main(save_output=True, ext='pdf', cmap='inferno'):
    args = read_args()
    # for easy consistency for now
    burn = args.burn

    ## calculate chi2 for the cross component vs. 0
    #chi2_xnull(args)

    colors = get_colors(n=20, cmap=cmap)
    if cmap == 'viridis':
        colors_corner = [colors[0], colors[16], yellow]
        esdcolors = [colors[0], colors[16], yellow]
        fitcolors = colors[15]
    elif cmap == 'inferno':
        colors_corner = [colors[0], colors[13], colors[17]]
        esdcolors = colors_corner
        fitcolors = colors[12]

    paramtypes = ('function', 'read', 'uniform', 'loguniform',
                  'normal', 'lognormal', 'fixed')

    hdr = utils.read_header(args.chainfile.replace('.fits', '.hdr'))
    params, prior_types, val1, val2, val3, val4, \
        datafiles, cols, covfile, covcols, exclude_bins, \
        model, nwalkers, nsteps, nburn = hdr
    jfree = (prior_types == 'uniform') | (prior_types == 'loguniform') | \
            (prior_types == 'normal') | (prior_types == 'lognormal')
    if not hasattr(jfree, '__iter__'):
        jfree = numpy.array([jfree])

    tree = args.chainfile[:-5].split('/')
    root = tree[-1][:-5]
    if args.output_path is None:
        #args.output_path = os.path.join('mcmcplots', *tree[1:])
        # assuming that the output data are always located in a folder
        # called `output*`
        args.output_path = os.path.join(
            tree[0].replace('output', 'mcmcplots'), *tree[1:])
        #args.output_path = os.path.join('mcmcplots', root)
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    print 'Reading file', args.chainfile, '...'
    data = fits.getdata(args.chainfile)
    names = data.names

    # get all esds
    # single bin
    if 'esd' in names:
        esd_keys = ['esd']
    elif 'esd_total' in names:
        esd_keys = ['esd_total']
    # multiple bins
    else:
        if 'esd1' in names:
            esd_name = 'esd'
        #elif 'esd_total1' in names:
        else:
            esd_name = 'esd_total'
        i = 1
        esd_keys = ['{0}1'.format(esd_name)]
        while True:
            i += 1
            key = '{0}{1}'.format(esd_name, i)
            if key in names:
                esd_keys.append(key)
            else:
                break

    good = (data.field('chi2') > 0) & (data.field('chi2') < 9999)
    # for some reason these happen every now and then
    #for i in xrange(len(esd_keys)):
        #good = (good) & (data.field('csat{0}'.format(i+1)) < 100)
    chain, keys = numpy.transpose([(data.field(key)[good], key)
                                   for key in names])

    esds = numpy.array([data.field(key)[good] for key in esd_keys])
    Nobsbins, Nsamples, Nrbins = esds.shape
    dof = Nobsbins*Nrbins - jfree[jfree].size - 1

    #if 'linpriors' not in args.chainfile:
        #for i, key in enumerate(keys):
            #if 'Msat' in key or 'Mgroup' in key:
                #chain[i] = 10**chain[i]
    lnlike = data.field('lnlike')[good]

    print len(lnlike), 'samples'
    # doing delta_chi2
    if 'chi2_total' in keys:
        chi2_key = 'chi2_total'
    else:
        chi2_key = 'chi2'
    j = list(keys).index(chi2_key)
    best = numpy.argmax(lnlike)
    bestn = numpy.argsort(lnlike)[-10:]
    print 'min(chi2) = %.2f at step %d' %(chain[j][best], best)
    print 'max(lnlike) = %.2f' %lnlike.max()
    print 'Best %d steps:' %len(bestn), bestn
    burn = min(burn, int(0.9*len(chain[0])))
    best = numpy.argmax(lnlike[burn:])
    minchi2 = chain[j][burn+best]
    print 'min(chi2_burned) = {0:.2f} at step {1}'.format(minchi2, burn+best)
    pte = stats.chisqprob(minchi2, dof)
    if pte > 0.01:
        pte_fmt = '{:.2f}'
    elif pte > 1e-3:
        pte_fmt = '{:.3f}'
    else:
        pte_fmt = '{:.2e}'
    print 'dof = {0} --> PTE = {1}'.format(dof, pte_fmt.format(pte))
    print 'max(lnlike_burned) = %.2f' %((lnlike[burn:]).max())
    Nparams = len(keys)

    # trying...
    #n, edges = numpy.histogram(data.field('lnprob')[burn:], bins=100)
    #binwidth = edges[1] - edges[0]
    #centers = 0.5 * (edges[1:] + edges[:-1])
    #evidence = n.sum() * binwidth
    #print 'Evidence =', evidence

    print 'Plotting...'
    show = not save_output

    chi2 = data.field(chi2_key)[good]

    R, signal = sampling_utils.load_datapoints(datafiles, cols, exclude_bins)
    Nobsbins, Nrbins = signal.shape
    if (isinstance(datafiles, basestring) and len(esd_keys) > 1) or \
            Nobsbins != len(esd_keys):
        msg = 'ERROR: number of data files ({0})'.format(Nobsbins)
        msg += ' does not match number of ESDs ({0})'.format(len(esd_keys))
        raise ValueError(msg)

    cov = sampling_utils.load_covariance(covfile, covcols, Nobsbins, Nrbins,
                                         exclude_bins)
    cov, icov, likenorm, signal_err, cov2d = cov
    # I just always have corr one column after cov
    covcols[0] += 1
    corr = sampling_utils.load_covariance(covfile, covcols, Nobsbins, Nrbins,
                                          exclude_bins)[0]

    #out = plot_esd_verbose(args.chainfile, esds, esd_keys, best,
                           #burn=burn, show=show,
                           #save_output=save_output,
                           #output_path=output_path, ext=ext)
    for bw in (True, False):
        out = plot_esd(args, args.chainfile, chain, keys, esds, esd_keys,
                       best, R, signal, signal_err, Nobsbins, Nrbins,
                       colors=esdcolors, burn=burn, show=show,
                       bw=bw, save_output=save_output,
                       output_path=args.output_path, ext=ext)
    model_bestfit, model_percentiles = out

    corr = plot_covariance(args.chainfile, corr, Nobsbins, Nrbins,
                           corr=True, save_output=save_output,
                           output_path=args.output_path, ext=ext)
    cov = plot_covariance(args.chainfile, cov, Nobsbins, Nrbins,
                          save_output=save_output,
                          output_path=args.output_path, ext=ext)

    if Nobsbins > 1 or args.literature != '':
        if args.observable is None:
            observable = read_observable(args.chainfile)[0]
        else:
            observable = args.observable
        print 'observable =', observable
        literature = {'logmstar':
                            [('velander14', 'zu15', 'mandelbaum16',
                              'vanuitert16'),
                             ('eagle', 'rodriguez13', 'li16'),
                             ('velander14', 'zu15', 'mandelbaum16',
                              'vanuitert16', 'eagle'),
                             ('leauthaud12', 'velander14', 'mandelbaum16',
                              'vanuitert16'),
                             ('leauthaud12', 'velander14', 'mandelbaum16',
                              'vanuitert16', 'rodriguez13'),
                             ('eagle', 'rodriguez13')],
                      'distBCG':
                            [('sifon15', 'vdBosch16', 'pvdBosch16', 'li16'),
                             ('sifon15', 'vdBosch16'),
                             ('sifon15', 'vdBosch16', 'li16')]}
        #if args.udg:
            #literature['logmstar'] = [('sifon17', 
        if observable not in literature:
            literature[observable] = [[]]
        def do_plot_massobs(
                mass_trunc, mass_rbg, ratio, xlog, ylog, bw, mass_host='mcmc'):
            for j in literature[observable]:
                if (observable == 'distBCG' and ratio) or \
                        (observable == 'logmstar' and not ratio):
                    lit = ','.join(j)
                    dolit = True
                else:
                    lit = ''
                    dolit = False
                plot_massobs(args, chain, hdr, keys, Nobsbins, Nrbins,
                             colors=fitcolors, save_output=save_output,
                             literature=lit, norm=True, bw=bw, xlog=True,
                             ylog=True, ratio=ratio, mass_host=mass_host,
                             burn=burn, output_path=args.output_path,
                             mass_trunc=mass_trunc, mass_rbg=mass_rbg)
            if observable != 'logmstar' and not ratio:
                plot_massobs(args, chain, hdr, keys, Nobsbins, Nrbins,
                             show_mstar=True, ratio=False, burn=burn, bw=bw,
                             norm=True, output_path=args.output_path,
                             save_output=save_output, mass_trunc=mass_trunc,
                             mass_rbg=mass_rbg, colors=fitcolors)
            return

        for bw in (False, True):
            for ratio in (False, True):
                if 'Msat_rbg1' in keys:
                    do_plot_massobs(False, True, ratio, True, True, bw)
                do_plot_massobs(False, False, ratio, True, True, bw)
                #if not (ratio or bw):
                    #return
            if 'tnfw' in model:
                do_plot_massobs(True, False, ratio, True, True, bw)
            if 'distBCG' in args.chainfile:
                do_plot_massobs(False, True, ratio, True, True, bw, mass_host=6e14)
            # just do colors
            break
        #return

    for i in xrange(2):
        model = args.chainfile.split('/')[-1].split('-')[i]
        plotkeys = get_plotkeys(model, keys)
        if plotkeys is not None:
            break
    plot_keys, plot_names = plotkeys
    for pkeys, output in izip(plot_keys, plot_names):
        if keys[numpy.in1d(keys, pkeys)].size != len(pkeys):
            continue
        print 'pkeys =', pkeys
        for bw, out in izip((True, False),
                            ('{0}_bw'.format(output), output)):
            plot_samples(args, chain, keys, (best,minchi2,dof,pte),
                         Nobsbins*Nrbins, plot_keys=pkeys, burn=burn, bw=bw,
                         bcolors=colors_corner, save_output=save_output,
                         output_path=args.output_path, output=out, ext=ext)

    plot_keys = [key for key in keys
                 if chain[keys == key][0].std() and 'esd' not in key]
    colors = [colors[8], colors[12], colors[16]]
    plot_samples(args, chain, keys, (best,minchi2,dof,pte),
                 Nobsbins*Nrbins, plot_keys=plot_keys, bcolors=colors,
                 burn=burn, save_output=save_output,
                 output_path=args.output_path, output='corner_all', ext=ext)

    #plot_satsignal(args.chainfile, chain, keys, Ro, signal, signal_err,
                   #burn=burn)
    # I want to do this only for single-bin chains
    #if 'bin' in args.args.chainfile:
        #chi2grid(hdr, bestfit[2], bestfit[3])
    return


def chi2grid(hdr, Mgroup=None, fc_group=None, Msat=None, fc_sat=1,
             datafile_index=-1, allpoints=False, save_plot=True):
    # read data
    if save_plot:
        output = hdr.replace('outputs/', 'plots/')
        output = output.replace('.hdr', '_chi2grid.png')
    #if Mgroup is not None:
        #output = output.replace('.png', '-fixgroup.png')
    #elif Msat is not None:
        #output = output.replace('.png', '-fixsat.png')
    #if os.path.isfile(output):
        #return
    params, prior_types, sat_profile_name, group_profile_name, \
        val1, val2, val3, val4, \
        datafile, cols, covfile, covcol, \
        model, nwalkers, nsteps, nburn = hdr
    function = getattr(models, model)
    sat_profile = getattr(nfw, sat_profile_name)
    group_profile = getattr(nfw, group_profile_name)
    if datafile_index > -1:
        R, Ro, esd, esd_err, used = read_datafile(len(datafile),
                                                  datafile[datafile_index],
                                                  cols)
        R = [R for i in xrange(len(datafile))]
        val1[params == 'Mgroup%d' %(datafile_index+1)] = Mgroup
        jM = (params == 'Msat%d' %(datafile_index+1))
        # this only for simulate_data.py
    else:
        R, Ro, esd, esd_err, used = read_datafile(len(datafile),
                                                  datafile, cols)
        val1[params == 'Mgroup'] = Mgroup
        jM = (params == 'Msat')
    val2[params == 'fc_group'] = fc_group
    ja = (params == 'fc_sat')
    #setup
    k = 7
    Rrange = logspace(log10(0.99*Ro.min()),
                            log10(1.01*Ro.max()), 2**k)
    Rrange = numpy.append(0, Rrange)
    if datafile_index > -1:
        Rrange = [Rrange for i in xrange(len(datafile))]
    angles = linspace(0, 2*numpy.pi, 540)

    Msat = numpy.arange(10.8, 12.81, 0.1)
    fc_sat = numpy.arange(0.1, 2.01, 0.1)
    theta = numpy.append(val1, [Rrange, angles])
    chi2 = numpy.zeros((len(Msat),len(fc_sat)))
    # here loop over all Msat's and fc_sat's and compute the chi2,
    # then plot results in a grid.
    imin = numpy.zeros(len(Msat), dtype=int)
    # this is very slow but what the hell
    for i, M in enumerate(Msat):
        theta[jM] = M
        for j, a in enumerate(fc_sat):
            theta[ja] = a
            expected = function(theta, R, sat_profile, group_profile)[0]
            if datafile_index > -1:
                expected = expected[datafile_index]
            chi2[i][j] = utils.chi2(expected[used], esd[used], esd_err[used])
    ijmin = numpy.unravel_index(chi2.argmin(), chi2.shape)
    title = r'$\min(\chi^2)=%.2f\,{\rm at}\,(%.1f,%.1f)$' \
            %(chi2[ijmin], fc_sat[ijmin[1]], Msat[ijmin[0]])
    extent = (fc_sat[0],fc_sat[-1],Msat[0],Msat[-1])
    if save_plot:
        ax = pylab.axes()
        pylab.imshow(log10(chi2), origin='lower',
                     cmap=cm.gist_stern, interpolation='nearest',
                     extent=extent)
        pylab.colorbar(label=r'$\log\,\chi^2$')
        pylab.xlabel(r'$A_{cM}$')
        pylab.ylabel(r'$\log\, M_{\rm sub}/{\rm M}_\odot$')
        pylab.title(title)
        pylab.savefig(output, format=output[-3:])
        pylab.close()
        print 'Saved to', output
    return chi2, extent, fc_sat[ijmin[1]], Msat[ijmin[0]]


def chi2_xnull(args, output_path):
    hdrfile = args.chainfile.replace('.fits', '.hdr')
    with open(hdrfile) as file:
        for line in file:
            line = line.split()
            if line[0] == 'datafile':
                datafile = line[1]
            if line[0] == 'covfile':
                covfile = line[1]
            if line[0] == 'cols':
                datacols = [float(i) for i in line[1].split(',')]
                datacols[1] += 1
            if line[0] == 'covcols':
                covcols = [float(i) for i in line[1].split(',')]
    data = readfile.table(datafile, cols=datacols)
    cov = readfile


def plot_covariance(chainfile, cov, Nobsbins, Nrbins, corr=False,
                    save_output=True, output_path='mcmcplots', ext='pdf'):
    try:
        cmap = cm.viridis_r
    except AttributeError:
        cmap = colormaps.viridis_r
    hdrfile = chainfile.replace('.fits', '.hdr')
    hdr = open(hdrfile)
    for line in hdr:
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'covfile':
            covfile = line[1]
        elif line[0] in ('covcol', 'covcols'):
            covcols = [int(i) for i in line[1].split(',')]
            break
    hdr.close()
    if corr:
        output = os.path.join(output_path, 'correlation.{0}'.format(ext))
        vmin = int(10*cov.min()) / 10.
        vmax = int(10*cov[cov < 1].max()) / 10. + 0.1
        log = False
        cbarticks = numpy.arange(-0.1, 0.71, 0.1)
    else:
        output = os.path.join(output_path, 'covariance.{0}'.format(ext))
        vmin, vmax = numpy.percentile(cov[~numpy.isnan(cov)], [0.5, 99.5])
        log = False

    # now plot full covariances
    labelsize = 22 + 2*Nobsbins
    #labelsize = 16 + 2*(Nobsbins-1)
    tickfmt = ticker.FormatStrFormatter
    if log:
        cov = log10(cov)
    fig, axes = pylab.subplots(figsize=(4*Nobsbins,4*Nobsbins),
                               nrows=Nobsbins, ncols=Nobsbins)
    if Nobsbins == 1:
        axes = [[axes]]
    for m, row in enumerate(axes):
        for n, ax in enumerate(row[::-1]):
            title = r'$({0},{1})$'.format(Nobsbins-m, n+1)
            # note that the extent is set by hand!
            img = ax.imshow(cov[m][n], extent=(0.02,2,0.02,2),
                            origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                            interpolation='nearest')
            ax.set_xscale('log')
            ax.set_yscale('log')
            if m == len(axes) - 1:
                ax.xaxis.set_major_formatter(tickfmt('$%s$'))
                ax.xaxis.set_tick_params(labelsize=labelsize, pad=10)
                ax.set_xlabel(r'$R\,({\rm Mpc})$', fontsize=labelsize)
            else:
                ax.set_xticklabels([])
            if n == len(row) - 1:
                ax.yaxis.set_major_formatter(tickfmt('$%s$'))
                ax.yaxis.set_tick_params(labelsize=labelsize)
                ax.set_ylabel(r'$R\,({\rm Mpc})$', fontsize=labelsize)
            else:
                ax.set_yticklabels([])
            #ax.set_title(title, fontsize=20)
    # this only for 3 bins
    #fig.subplots_adjust(left=0.10, bottom=0.10, right=0.96, top=0.96)
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.8)
    if save_output:
        #output = os.path.join(output_path, output.split('/')[-1])
        #output = output.replace('.fits', '_{0}.{1}'.format(suffix, ext))
        pylab.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    #fig = pylab.figure(figsize=(10,2.5))
    #ax = pylab.axes([0.10, 0.01, 0.88, 0.88], frameon=False,
                    #xticks=[], yticks=[], axisbg='none')
    subplot_kw = {'frameon': False, 'xticks': [], 'yticks': [],
                  'axisbg' : 'none'}
    fig, ax = pylab.subplots(figsize=(10,1.6), subplot_kw=subplot_kw)
    if log:
        label = r'$\log\,C_{mnij}$'
    else:
        #label = r"$\boldsymbol{C'}_{mnij}$"
        label = r'${\rm Corr}_{mnij}$'
    cbar = fig.colorbar(img, ax=ax, fraction=.8, aspect=18,
                        orientation='horizontal')
    cbar.set_label(label=label, fontsize=30)
    #for tl in cbar.ax.get_xticklabels():
        #tl.set_fontsize(18)
    cbar.ax.xaxis.set_tick_params(labelsize=18)
    fig.tight_layout(pad=0.4)
    if save_output:
        output = output.replace('.{0}'.format(ext), '_cbar.{0}'.format(ext))
        pylab.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    if log:
        return 10**cov
    return cov


def plot_esd_verbose(args, hdr, esds, esd_keys, best, burn=10000,
                     percentiles=(2.5,16,84,97.5), show=False,
                     save_output=False, ext='png'):
    # plot esd with extra information
    out = utils.read_header(chainfile.replace('.fits', '.hdr'))
    params, prior_types, \
        val1, val2, val3, val4, \
        datafiles, cols, covfile, covcol, exclude_bins, \
        model, nwalkers, nsteps, nburn = out
    n = len(datafiles)
    if (isinstance(datafiles, basestring) and len(esd_keys) == 1) or \
            n != len(esd_keys):
        msg = 'ERROR: number of data files does not match number of ESDs'
        print msg
        return
    signal = [[] for i in xrange(n)]
    cross = [[] for i in xrange(n)]
    signal_err = [[] for i in xrange(n)]
    residuals = []
    for i, datafile in enumerate(datafiles):
        data = read_datafile(n, datafile, cols, covfile, covcol, i)
        R, Ro, signal[i], cross[i], signal_err[i], used = data
    pylab.figure(figsize=(5*n,4.5))
    xo = [0.12, 0.09, 0.06][n-1]
    axw = [0.85, 0.45, 0.305][n-1]
    yo = 0.16
    axh = 0.82
    inset_ylim = [(0, 50), (-10, 30), (-10, 30)]
    for i, s, serr, esd in izip(count(),
                                          signal, signal_err, esds):
        ax = pylab.axes([xo+i*axw, yo, axw, axh], xscale='log')
        median_signal = numpy.median(esd[burn:], axis=0)
        #chi2 = (((median_signal - s[used]) / serr[used])**2).sum()
        #chi2_total += chi2
        #print 'KS:', stats.ks_2samp(median_signal, s[used])
        residuals.append(s[used] - median_signal)
        per_signal = [numpy.percentile(esd[burn:], p, axis=0)
                      for p in percentiles]
        ax.errorbar(Ro[used], s[used], yerr=serr[used],
                    fmt='o', color=red, mec=red, mfc='none',
                    mew=1, elinewidth=1, zorder=10)
        ax.errorbar(Ro[~used], s[~used], yerr=serr[~used],
                    fmt='o', color='0.7', mec='0.7', mfc='none',
                    mew=1, elinewidth=1, zorder=10)
        ax.plot(Ro[used], esd[best], '-', color='0.5')
        # setting to 10 because I consider the concentration of satellites
        # to be fixed. Otherwise should be 9.
        ax.plot(Ro[used], median_signal, '-', color=blue, lw=2)#,
                #label=r'$\chi^2/{\rm d.o.f.}=%.2f/10$' %(chi2))
        ax.plot(Ro[used], per_signal[0], ':', color=blue)
        ax.plot(Ro[used], per_signal[3], ':', color=blue)
        ax.plot(Ro[used], per_signal[1], '-', color=blue)
        ax.plot(Ro[used], per_signal[2], '-', color=blue)
        #l = ax.legend(loc='upper right')
        #l.get_frame().set_alpha(0)
        if i == 0:
            ylabel = r'$\Delta\Sigma\,(h\,\mathrm{M_\odot pc^{-2}})$'
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
            ax.get_xticklabels()[1].set_visible(False)
        ax.set_xlabel(r'$R\,({\rm Mpc})$')
        ax.set_ylim(-20, 150)
        # inset
        inset = pylab.axes([xo+(i+0.5)*axw, yo+0.40*axh, 0.47*axw, 0.45*axh],
                           xscale='log')
        inset.errorbar(Ro[used], s[used], yerr=serr[used],
                    fmt='+', color=red, mec=red, mfc='none',
                    mew=1, elinewidth=1, zorder=10)
        inset.plot(Ro[used], esd[best], '-', color='0.5')
        inset.plot(Ro[used], median_signal, '-', color=blue, lw=2)
        inset.plot(Ro[used], per_signal[0], ':', color=blue)
        inset.plot(Ro[used], per_signal[3], ':', color=blue)
        inset.plot(Ro[used], per_signal[1], '--', color=blue)
        inset.plot(Ro[used], per_signal[2], '--', color=blue)
        inset.set_xlim(0.15, 2)
        inset.set_ylim(*inset_ylim[i])
        xticks = [0.2, 0.5, 1, 2]
        inset.set_xticks(xticks)
        inset.set_xticklabels(xticks)
        inset.yaxis.set_major_locator(ticker.MultipleLocator(10))
        inset.set_xlabel('$R$')
        inset.set_ylabel('$\Delta\Sigma$')
    if save_output:
        #output = os.path.join(args.output_path, output.split('/')[-1])
        #output = output.replace('.fits', '_esd_verbose.' + ext)
        output = os.path.join(output_path, 'esd_verbose.{0}'.format(ext))
        pylab.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    # plot normalized residuals
    #t = linspace(-3, 3, 100)
    #pylab.hist(res, bins=7, histtype='step', color=(0,0,1))
    #pylab.plot(t, numpy.exp(-t**2/2), 'k-')
    #pylab.show()
    residuals = numpy.array(residuals)
    out = (Ro, signal, signal_err, used,
           median_signal, per_signal, residuals)
    return out


def plot_esd(args, chainfile, chain, keys, esds, esd_keys, best,
             R, signal, signal_err, Nobsbins, Nrbins,
             colors='ry', burn=10000, percentiles=(2.5,16,84,97.5),
             show=False, save_output=False, bw=False,
             output_path='mcmcplots', ext='png'):
    """
    plot the ESD in the format that will go in the paper

    """
    if bw:
        colors = ('k', '0.55', '0.8')
    if Nobsbins == 1:
        fig, axes = pylab.subplots(figsize=(6,4))
    else:
        fig, axes = pylab.subplots(figsize=(5*Nobsbins,5), ncols=Nobsbins)
    if Nobsbins == 1:
        axes = [axes]
    path = os.path.split(chainfile)
    print 'path =', path
    if args.observable is None:
        observable = read_observable(args.chainfile)
    else:
        observable = args.observable
    print 'observable =', observable
    if observable is None:
        obsbins = None
    else:
        obsbins = numpy.array([float(o) for o in observable[1]])
        print 'obsbins =', obsbins
        observable = observable[0]
    t = logspace(-2, 0.7, 100)
    model_percentiles = []
    ylabel = r'$\Delta\Sigma\,(h\,\mathrm{M_\odot pc^{-2}})$'
    if args.scale is not False:
        args.scale[1] = float(args.scale[1])
        if args.scale[0] == 'logLstar':
            scale_label = ('L', r'L_\star')
        elif args.scale[0] == 'logmstar':
            scale_label = (r'm_\star', r'{\rm M}_\odot')
        #pivot_exp_label = round(args.scale[1], 0)
        pivot_exp_label = int(args.scale[1])
        pivot_label = round(10**args.scale[1] / 10**pivot_exp_label, 0)
        label_scale = r'$\left(\frac{{{0}}}'.format(scale_label[0])
        label_scale += r'{{{0:.1f}\times10^{{{1:.0f}}}{2}}}\right)'.format(
                        pivot_label, pivot_exp_label, scale_label[1])
        if float(args.scale[2]) != 1:
            label_scale += '^{%s}' %args.scale[2]
        label_scale += '$'
        ylabel = r'{0}{1}'.format(label_scale, ylabel)
    for i, ax, Ro, s, serr, esd in izip(count(), axes, R, signal,
                                        signal_err, esds):
        bestfit = esd[burn:][best]
        per_signal = [numpy.percentile(esd[burn:], p, axis=0)
                      for p in percentiles]
        #per_signal = numpy.percentile(esd[burn:], percentiles, axis=0)
        model_percentiles.append(per_signal)
        ax.errorbar(
            Ro, s, yerr=serr, fmt='o', color='k', mec='k', mfc='k',
            capsize=2, ms=10, mew=2, elinewidth=2, zorder=10)
        ##ax.errorbar(Ro, x, yerr=serr,
                    ##fmt='o', color='0.7', mec='0.7', mfc='none',
                    ##ms=6, mew=1, elinewidth=1, zorder=9)
        ax.plot(Ro, bestfit, '-', color=colors[0], lw=3)
        ##ax.plot(Ro, per_signal[0], '-', color='k')
        ##ax.plot(Ro, per_signal[3], '-', color='k')
        ##ax.plot(Ro, per_signal[1], '-', color='k')
        ##ax.plot(Ro, per_signal[2], '-', color='k')
        ax.fill_between(Ro, per_signal[0], per_signal[3],
                        color=colors[1], lw=0)
        ax.fill_between(Ro, per_signal[0], per_signal[1],
                        color=colors[2], lw=0)
        ax.fill_between(Ro, per_signal[2], per_signal[3],
                        color=colors[2], lw=0)
        if Nobsbins > 1:
            label = get_label(observable, obsbins[i], obsbins[i+1])
            if i == len(signal) - 1:
                label = label.replace('<', r'\leq')
            ax.annotate(label, xy=(0.95,0.92), ha='right', va='top',
                        xycoords='axes fraction', fontsize=22)
        ax.axhline(0, ls='--', color='k', lw=1)
        ax.set_xscale('log')
        if Ro.max() < 1:
            ax.set_xlim(0.02, 1)
            w_pad = -0.5
        else:
            ax.set_xlim(0.02, 2)
            w_pad = 0.4
        if args.udg:
            ax.set_ylim(-40, 50)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        else:
            ax.set_ylim(-30, 180)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%s$'))
        if i == 0:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%d$'))
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
            #ax.get_xticklabels()[0].set_visible(False)
        ax.set_xlabel(r'$R\,({\rm Mpc})$')
    fig.tight_layout(pad=0.4, w_pad=w_pad)
    if save_output:
        output = os.path.join(output_path, 'esd.{0}'.format(ext))
        if bw:
            output = output.replace('.{0}'.format(ext), '_bw.{0}'.format(ext))
        fig.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    return bestfit, numpy.array(model_percentiles)


def plot_massobs(
        args, data, hdr, keys, Nobsbins, Nrbins, colors='ry', burn=10000,
        errorbars=(16,84), ratio=True, mass_host='mcmc', show_mstar=False,
        dx=0.02, save_output=False, output_path='mcmcplots', norm=True,
        radius_3d=False, mass_trunc=False, mass_rbg=False, literature='',
        xlog=True, ylog=True, bw=False, twopanels=False, out_fit=None):
    params, prior_types, val1, val2, val3, val4, \
        datafile, cols, covfile, covcols, exclude_bins, \
        model, nwalkers, nsteps, nburn = hdr
    if show_mstar:
        observable = 'logmstar'
    else:
        #observable = args.chainfile.split('/')[1].split('-')[0].split('_')[0]
        observable = read_observable(args.chainfile)[0]

    output = massobs_output(
        observable, output_path, literature, mass_trunc, mass_rbg,
        ratio, show_mstar, radius_3d, xlog, ylog, bw, mass_host=mass_host)

    if twopanels:
        fig, axes = pylab.subplots(figsize=(5,7), nrows=2)
    else:
        #if observable == 'distBCG':
            #fig, ax = pylab.subplots(figsize=(5.5,4))
        #else:
        fig, ax = pylab.subplots(figsize=(5.5,5))
        axes = [ax]

    Mstar = utils.read_avgs('logmstar', chainfile=args.chainfile)
    Mstar_lo = Mstar[3]
    Mstar_hi = Mstar[4]
    Mstar_err = Mstar[1]
    Mstar = Mstar[0]
    good = (data[keys == 'chi2'][0] > 0)
    if 'Mhost1' in keys:
        Mhost, (Mhost_lo, Mhost_hi) = \
            get_errorbars(data, good, burn, keys, 'Mhost', n=None)
    else:
        Mhost, (Mhost_lo, Mhost_hi) = \
            get_errorbars(data, good, burn, keys, 'Mgroup', n=None)

    if observable == 'distBCG':
        if 'rsat1' in keys and radius_3d:
            #obs, (obs_lo, obs_hi) = get_errorbars(data, good, burn,
                                                  #keys, 'rsat', Nobsbins)
            obs = utils.read_avgs(observable, chainfile=args.chainfile)
            obs_err = obs[1]
            obs_lo = obs[3]
            obs_hi = obs[4]
            obs = obs[0]
            xlabel = r'$r_{\rm sat}$'
        else:
            pfh = stattools.percentile_from_histogram
            Rsat = 10**val1[params == 'Rsat'][0]
            n_Rsat = [val1[params == 'n_Rsat%d' %(i+1)][0]
                      for i in xrange(Nobsbins)]
            obs = numpy.array([pfh(n, Rsat, 50) for n in n_Rsat])
            obs_lo = obs - numpy.array([pfh(n, Rsat, 16) for n in n_Rsat])
            obs_hi = numpy.array([pfh(n, Rsat, 84) for n in n_Rsat]) - obs
            # fix later
            obs_err = numpy.zeros(obs.size)
            xlabel = r'$R_{\rm sat}$'
        if norm:
            if isinstance(mass_host, float):
                r200 = conversions.rsph(mass_host, 0.1, ref='200a')
            else:
                r200 = conversions.rsph(Mhost, 0.1, ref='200a')
            obs /= r200
            obs_lo /= r200
            obs_hi /= r200
            obs_err /= r200
            xlabel += r'$/r_{{200},{\rm h}}$'
    elif observable in ('logLstar', 'logmstar'):
        obs = utils.read_avgs(observable, chainfile=args.chainfile)
        obs_lo = obs[3]
        obs_hi = obs[4]
        obs_err = obs[1]
        obs = obs[0]
        if observable == 'logLstar':
            xlabel = r'$L/L^\star$'
        elif observable == 'logmstar':
            xlabel = r'$m_\star\,({\rm M}_\odot)$'

    if 'Msat1_rt' in keys or 'Msat_rt1' in keys and mass_trunc:
        msub_name = 'Msat_rt'
        if 'Msat1_rt' in keys:
            Msat, (Msat_lo, Msat_hi) = get_errorbars(
                data, good, burn, keys, 'Msat', Nobsbins, suffix='_rt')
        else:
            Msat, (Msat_lo, Msat_hi) = get_errorbars(
                data, good, burn, keys, 'Msat_rt', Nobsbins)
        if 'rt1' in keys:
            rt, (rt_lo, rt_hi) = get_errorbars(
                data, good, burn, keys, 'rt', Nobsbins)
        else:
            rt, (rt_lo, rt_hi) = get_errorbars(
                data, good, burn, keys, 'logrt', Nobsbins)
    elif 'Msat_rbg1' in keys and mass_rbg:
        msub_name = 'Msat_rbg'
        Msat, (Msat_lo, Msat_hi) = get_errorbars(
            data, good, burn, keys, 'Msat_rbg', Nobsbins)

    else:
        msub_name = 'Msat'
        Msat, (Msat_lo, Msat_hi) = get_errorbars(
            data, good, burn, keys, 'Msat', Nobsbins)

    print '**\nMsat = {0}\n**'.format(Msat)
    print 'Mstar =', Mstar

    # the uncertainties in obs are the 68% range of the distribution,
    # not the unceratinty on the mean (which is surely negligible)
    print '** Fitting (ratio={0}) **'.format(ratio)
    if ratio:
        #y1 = calc_ratios(Mstar, Msat, ylo=Msat_lo, yhi=Msat_hi)
        y1 = calc_ratios(Msat, Mstar, xlo=Msat_lo, xhi=Msat_hi)
        y2 = calc_ratios(Msat, Mhost, Msat_lo, Msat_hi, Mhost_lo, Mhost_hi)
        print 'Msat / Mstar = {0}'.format(y1)
        print 'Msat / Mhost = {0}'.format(y2)
        mo = 1
        y2lo = numpy.array([min(lo, (1-1e-10)*y)
                            for lo, y in izip(y2[1], y2[0])])
        # fit a power law  only to the Msub/Mhost vs L relation
        y_to_fit = (y1[0], (y1[1]+y1[2])/2)
        if twopanels:
            ax_with_fit = axes[1]
    else:
        y1 = (Msat, Msat_lo, Msat_hi)
        y2 = (Mhost, Mhost_lo, Mhost_hi)
        mo = 1e15
        y2lo = y2[1]
        # fit a power law only to the Msub vs L relation
        y_to_fit = (y1[0], (y1[1]+y1[2])/2)
        if twopanels:
            ax_with_fit = axes[0]
    if not twopanels:
        ax_with_fit = ax

    y1lo = numpy.array([min(lo, (1-1e-10)*y)
                        for lo, y in izip(y1[1], y1[0])])
    y1lo = y1[1]
    ylabel = get_axlabel(msub_name)
    if ratio:
        ylabels = (ylabel + r'$/m_\star$', ylabel + r'$/M_{\rm cl}$')
    else:
        ylabels = (ylabel + r'$\,({\rm M}_\odot)$',
                   r'$M_{\rm cl}\,(10^{%d}{\rm M}_\odot)$' %log10(mo))

    if twopanels:
        return

    ylabel = ylabels[0]
    do_fit = True
    if observable == 'distBCG':
        if ratio:
            do_fit = False
            #logify_fit = False
        else:
            for ax in axes:
                ax.set_xlim(0.05, 2)
    plot_massobs_panel(ax, observable, obs, y1[0], (y1lo,y1[2]), obs_err,
                       xlabel=xlabel, ylabel=ylabel, zorder=10,
                       xlog=xlog, ylog=ylog)

    if observable == 'logmstar' and not ratio:
        if literature:
            for axi in axes:
                axi.set_xlim(1e9, 5e11)
            ylim = [1e10, 2e14]
        else:
            ylim = [1e10, 2e13]
        #if mass_rbg:
            #ylim[0] = 5e9
        for axi in axes:
            axi.set_ylim(*ylim)
    elif observable == 'distBCG':
        if msub_name == 'Msat_rbg' and ratio:
            ax.set_ylim(2, 500)

    #xpivot = 10**((log10(obs) * w).sum() / w.sum())
    # this one actually gives smaller errors
    xpivot = numpy.median(obs)
    if do_fit:
        w = 1 / (y_to_fit[1] / y_to_fit[0])**2
        if observable == 'logmstar':
            xpivot_log = int(log10(xpivot))
            xpivot = round(xpivot / 10**xpivot_log, 0) * 10**xpivot_log
        x_to_fit = (obs, obs_err, xpivot)
        if observable == 'logmstar' and not ratio and literature \
                and '/disks/shear7/' in os.getcwd():
            fitkind = 'kelly'
        else:
            fitkind = 'bces'
        fitkind = 'bces'
        try:
            fit = plot_fit(ax_with_fit, x_to_fit, y_to_fit, kind=fitkind,
                           colors=colors, logify=ylog)
        except ValueError:
            return
        #afit, bfit, sfit = [ufloat(i) if len(i) == 2 else ufloat(i, 0)
                            #for i in fit]
        afit, bfit, sfit = fit
        print 'xpivot =', xpivot, log10(xpivot)
        print 'afit =', afit
        print 'bfit =', bfit
        print 'sfit =', sfit
        #if out_fit is not None:
            #fitline = '{0:<10s}  
    else:
        #afit, bfit, sfit = unumpy.uarray(numpy.zeros((2,3)))
        afit, bfit, sfit = numpy.zeros((3,2))

    if observable == 'logmstar' and literature:
        curves, curve_labels = plot_literature(
            args, axes[0], obs, observable, mass=msub_name,
            literature=literature)
    elif observable == 'distBCG' and literature:
        if do_fit:
            if ylog:
                norm = 10**afit[0]
            else:
                norm = afit[0]
        else:
            norm = numpy.median(y1[0])
        for ax in axes:
            #ax.set_xscale('linear')
            #ax.set_xlim(0, 1)
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            #ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            if xlog:
                ax.set_xlim(0.05, 1)
            if not ylog:
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        curves, curve_labels = plot_literature(
            args, axes[0], obs, observable, xnorm=xpivot, norm=norm, bw=bw,
            mass=msub_name, literature=literature)

    # Just for comparison
    #if observable == 'logmstar' and literature:
    if 'logmstar' in args.chainfile and literature:
        ratio_output = os.path.split(output)
        ratio_output = os.path.join(
            ratio_output[0], 'ratio_{0}'.format(ratio_output[1]))
        rfig, rax = pylab.subplots(figsize=(6,3.5))
        for i, curve, clabel in izip(count(), curves, curve_labels):
            #print clabel
            xmin = (curve.x[1] if curve.x[1] < 10 else curve.x[0])
            xmax = curve.x[numpy.argmin(numpy.absolute(curve.x-11.3))]
            x = linspace(xmin, xmax, 10)
            # show the EAGLE prediction of msub/M200,h
            if 'EAGLE' in clabel:
                # in this case using the relation for centrals from EAGLE
                xeag = array([10.45, 10.75, 11.05, 11.35, 11.55]) + (0.6777-0.7)
                ycent = array([12.46, 12.92, 13.13, 13.39, 13.69]) + (0.6777-0.7)
                ycent = interp1d(xeag, ycent)
                xratio = (max(x.min(), xeag.min()), min(x.max(), xeag.max()))
                x = linspace(xratio[0], xratio[1], 10)
                print('xeag =', xeag)
                print('xratio =', xratio)
                print('x =', x)
                y = ycent(x)
                # subhalo masses
                yref = curve(x)
                ls = 'k--'
                clabel = 'EAGLE'
            else:
                y = curve(x)
                # log(SHSMR)
                yref = afit[0] + bfit[0]*(x - log10(x_to_fit[2]))
                ls = 'C{0}-'.format(i)
            rax.plot(10**x, 10**(yref-y), ls, lw=3, label=clabel)
        rax.axhline(1, ls='--', color='k', lw=1)
        rax.legend(fontsize=14, loc='upper center', ncol=2)
        rax.set_xlabel(r'$m_\star\,(\mathrm{M}_\odot)$')
        rax.set_ylabel(r'$m_\mathrm{bg}/M_\mathrm{200,central}$')
        rax.set_xscale('log')
        ylog = False
        if ylog:
            rax.set_ylim(0.2, 2)
            rax.set_yscale('log')
            rax.yaxis.set_major_formatter(ticker.LogFormatter(labelOnlyBase=True))
            #rax.yaxis.set_major_locator(ticker.FixedLocator((0.2,1,2)))
            #rax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%s$'))
        else:
            rax.set_ylim(0, 1.6)
            rax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            rax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        plottools.savefig('{0}.pdf'.format(ratio_output), fig=rfig)
        print

    fig.tight_layout(pad=0.5, h_pad=0.1)
    if save_output:
        #for ext in ('pdf', 'png'):
            #fig.savefig(output+'.'+ext, format=ext)
        fig.savefig('{0}.pdf'.format(output))
        #pylab.close()
        print 'Saved to', output
    else:
        output = ''
        fig.show()
    # save same figure but with slope printed out
    if ax.legend_ is not None:
        ax.legend_ = None
        pylab.draw()
    msg = r'${\rm slope}=%.2f$' %bfit[0]
    if len(bfit) == 2:
        msg += r'$\pm%.2f$' %bfit[1]
    elif len(bfit) == 3:
        msg += r'$_{-%.2f}^{+%.2f}$' %(bfit[1], bfit[2])
    ax.annotate(msg, xy=(0.97,0.03), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=22)
    fig.savefig('{0}_slope.pdf'.format(output))

    # copy the top panel only
    #fig.delaxes(axes[0])
    pylab.close(fig)
    #ax = axes[0]
    #fig = pylab.figure(figsize=(5.5,5))
    #fig.add_axes(ax)
    #output += '_top'
    #for ext in ('pdf', 'png'):
        #fig.savefig(output+'.'+ext, format=ext)
    #pylab.close()
    return afit, bfit, sfit


def plot_massobs_panel(ax, observable, x, y, yerr=[], xerr=[], symbol='ko',
                       xlog=True, ylog=True, xlabel='', ylabel='', zorder=-2):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=symbol, capsize=2,
                ms=11, mew=2, elinewidth=2, zorder=zorder, label='_none_')

    if xlog:
        ax.set_xscale('log')
    if ylog:
        try:
            ax.set_yscale('log')
        except ValueError:
            return
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(yerr) == 2 and hasattr(yerr[0], '__iter__'):
        format_axes(ax, observable, y, yerr[1], xlabels=bool(xlabel))
    else:
        format_axes(ax, observable, y, yerr, xlabels=bool(xlabel))
    if observable == 'logmstar':
        ax.set_xlim(1e9, 2e11)

    return


def plot_samples(args, data, keys, best, npts, plot_keys=[],
                 bcolors='ry', burn=10000, save_output=False, bw=False,
                 output_path='mcmcplots', output='corner', ext='png',
                 plot_all=False, **kwargs):
    """
    Should be a lot more general in scope

    """
    if bw:
        bcolors = ['k', '0.55', '0.8']

    best, minchi2, dof, pte = best
    if 'chi2_total' in keys:
        chi2_key = 'chi2_total'
    else:
        chi2_key = 'chi2'

    model = os.path.split(args.chainfile)[-1].split('-')[0]
    print 'model =', model
    data, keys = numpy.transpose([(data[keys == key][0], key)
                                  for key in keys if 'esd' not in key])
    #data = numpy.array(data, dtype=float)
    Nparams = len(keys)
    avgs = []
    xmax = max([len(value) for value in data])
    fig, axes = pylab.subplots(figsize=(8,Nparams),
                               nrows=Nparams, sharex=True)
    if len(data[0]) > 1e6:
        j = linspace(0, len(data[0])-1, len(data[0])/100, dtype=int)
    else:
        j = linspace(0, len(data[0])-1, len(data[0]), dtype=int)
    for i, ax, value, key in izip(count(), axes, data, keys):
        ax.plot(j, value[j], ',', color='0.7')
        to, avg = runningavg(value, thin=value.size/50)
        #ax.plot(to, cumavg(value, samples=len(to)),
                #'-', color=(0,0.8,0.8), lw=1)
        ax.plot(to, avg, '.', color=(0,0,1), ms=6)
        if burn > 0:
            ax.axvline(burn, ls='--', color=(0.6,0,0.6))
        avgs.append(avg[-1])
        ax.set_ylabel(key.replace('_', '\_'))
        ax.set_xlim(0, xmax)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.get_yticklabels()[0].set_visible(False)
        ax.get_yticklabels()[-1].set_visible(False)
    axes[-1].set_xlabel('samples')
    fig.subplots_adjust(bottom=0.02, top=0.98, right=0.98, hspace=0.1)
    if save_output:
        trace_output = os.path.join(output_path, 'trace.png')
        pylab.savefig(trace_output)
        pylab.close()
        print 'Saved to', trace_output
    else:
        pylab.show()
    # to use the results from above
    bgweight = numpy.arange(len(data[0]))
    data = numpy.array([x[burn:] for x in data])
    chi2 = data[keys == chi2_key][0]
    if save_output:
        output = output.replace('_trace', '_corner')
    if 'lnlike' in keys:
        best = numpy.argmax(data[keys == 'lnlike'][0])
    else:
        chi2sort = numpy.argsort(chi2)
        best = chi2sort[0]
    x = [numpy.median(chi2)]
    x.append(x[0] - numpy.percentile(chi2, 16))
    x.append(numpy.percentile(chi2, 84) - x[0])

    print 'keys =', keys
    print 'plot_keys =', plot_keys
    data1 = numpy.array([data[keys == key][0]
                         for key in plot_keys if key in keys])
    if data1.shape[0] == 1:
        return

    line = ''
    logfile = args.chainfile.replace('.fits', '.out')
    log = open(logfile, 'w')
    print >>log, '# param  best  median  e16  e84  p95  p99'
    print >>log, '#\n# chi2 = {0:.2f}'.format(minchi2)
    print >>log, '# dof = {0}\n# pte = {1:.3e}\n#'.format(dof, pte)

    truths = numpy.zeros(len(plot_keys))
    take_log = numpy.zeros(len(plot_keys), dtype=bool)
    for i, key, d in izip(count(), plot_keys, data1):
        m = numpy.median(d)
        p95, p99 = numpy.percentile(d, [95, 99])
        truths[i] = d[best]
        if m > 1e4:
            logm = log10(m)
            vals = (logm, logm - log10(numpy.percentile(d, 16)),
                    log10(numpy.percentile(d, 84)) - logm)
            p95 = log10(p95)
            p99 = log10(p99)
            truths[i] = log10(truths[i])
            x = logm
            take_log[i] = True
        else:
            vals = (m, m - numpy.percentile(d, 16),
                    numpy.percentile(d, 84) - m)
            x = m
        print >>log, '%-10s  %11.3e' %(key, truths[i]),
        print >>log, ' %11.3e  %10.3e  %10.3e' %vals,
        print >>log, ' {0:11.3e}  {1:11.3e}'.format(p95, p99)
        line += '$%.2f_{-%.2f}^{+%.2f}$' %vals
        if key == 'fc':
            line += '}'
        line += ' & '
    log.close()
    print 'Saved to', logfile

    if len(plot_keys) > 25:
        return

    # otherwise it overwrites the external value...
    labels = [i for i in plot_keys]
    for i in xrange(len(data1)):
        if take_log[i] or plot_keys[i][0] == 'M':
            if take_log[i]:
                data1[i] = log10(data1[i])
            labels[i] = 'log{0}'.format(labels[i])

    # already burned
    labels = [get_axlabel(key) for key in labels]
    lnlike = data[keys == 'lnlike'][0]
    limits = []
    for key, d in izip(plot_keys, data1):
        if 'Msat' in key:
            #if args.udg:
                #limits.append((8.3, 12.5))
            #else:
                limits.append((9,12.5))
        else:
            limits.append(numpy.percentile(d, [0.01,99.99]))
    truth1d_properties = dict(color=bcolors[0], zorder=10, ls='-')
    truth2d_properties = dict(color=bcolors[0], zorder=10, ls='+')
    likelihood_properties = dict(color=red, dashes=(8,6))
    corner = plottools.corner(data1, labels=labels, bins=25, bins1d=25,
                              #clevels=(0.68,0.95,0.99),
                              #output=output,
                              #ticks=ticks,
                              limits=limits,
                              truths=truths,
                              truth1d_properties=truth1d_properties,
                              truth2d_properties=truth2d_properties,
                              truths_in_1d=True,
                              medians1d=False, percentiles1d=False,
                              likelihood=lnlike, likesmooth=1,
                              likelihood_properties=likelihood_properties,
                              style1d='step',
                              #smooth=(0.5,0.5,0.5,0.1,0.2,0.2,0.2),
                              #smooth=(1,1,1,0.35,0.35,0.35,0.35),
                              background='filled',
                              linewidths=1, show_contour=True,
                              bcolor=bcolors[1:][::-1], verbose=True,
                              pad=0.5, h_pad=-0.15*len(data1),
                              w_pad=-0.22*len(data1), **kwargs)
    # to compute point statistics
    for i, key in enumerate(plot_keys):
        if key[:3] == 'log':
            data1[i] = 10 ** data1[i]
    fig, diagonals, offdiag = corner
    # illustrating how the flat prior in linear space looks in log space
    if model[-4:] != '_log':
        if 'Msat' in plot_keys:
            j = plot_keys.index('Msat')
            hist = numpy.histogram(log10(linspace(1e8, 1e13, 10000)),
                                   linspace(8, 13, 100))
            x = (hist[1][:-1] + hist[1][1:]) / 2
            y = 10. * hist[0] / hist[0].max()
            diagonals[j].plot(x, y, color=red, dashes=(6,4), lw=1)
            #exit()
    #fig.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    for ax in offdiag:
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        #if ax.get_xlabel():
            #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        #if ax.get_ylabel():
            #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    for ax, key, d in izip(diagonals, plot_keys, data1):
        m = numpy.median(d)
        v68 = [numpy.percentile(d, i) for i in (16, 84)]
        v95 = [numpy.percentile(d, i) for i in (2.5, 97.5)]
        #if 'M' in key:
        if key[:3] == 'log' or key[0] == 'M':
            m = log10(m)
            v68 = [log10(v) for v in v68]
            v95 = [log10(v) for v in v95]
        #ax.axvline(m, color='k', lw=2)
        #for v in v68:
            #line = ax.axvline(v, color='k', lw=2)
            #line.set_dashes([10,6])
        #for v in v95:
            #line = ax.axvline(v, color='k', lw=2)
            #line.set_dashes([3,4])
        ax.xaxis.set_tick_params(width=2)
        #if ax.get_xlabel():
            #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    if save_output:
        output = os.path.join(output_path, '{0}.pdf'.format(output))
        pylab.savefig(output, format=output[-3:])
        pylab.close()
        print 'Saved to', output

    if not plot_all:
        return

    if save_output:
        output = os.path.join(output_path, 'corner_all.pdf')
        if burn == 0:
            output = output.replace('corner_all', 'corner_all_noburn')

    j = [i for i, d in enumerate(data) if numpy.std(d)]
    truths = [d[best] for d in data[j]]
    keys1 = [k.replace('_', '\_') for k in keys[j]]
    try:
        plottools.corner(data[j], labels=keys1, bins=25, bins1d=50,
                         clevels=(0.68,0.95,0.99), #likelihood=lnlike,
                         truth_color=red, style1d='step',
                         truths=truths, background='filled', output=output,
                         top_labels=True, bcolor=bcolor, verbose=True)
        if save_output:
            print 'Saved to', output
    except ValueError:
        return
    return


def plot_satsignal(args, chain, keys, Ro, signal, signal_err,
                   burn=0, thin=100, h=1, Om=0.3, Ol=0.7):
    from itertools import count, izip
    from matplotlib import cm, colors as mplcolors
    from scipy import interpolate
    from time import time
    # local
    from nfw import esd, esd_offset, esd_sharp
    from utils import cM_duffy08, density_average, r200m

    cmap = pylab.get_cmap('spectral')
    cNorm  = mplcolors.Normalize(vmin=0, vmax=1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = [scalarMap.to_rgba(x) for x in (0.15, 0.35, 0.85)]
    #rng = numpy.arange(0, 1.01, 0.05)
    #colors = [scalarMap.to_rgba(x) for x in rng]
    #for i, color in izip(rng, colors):
        #pylab.plot(i, i, 'o', color=color, mec=color, ms=15)
    #pylab.xlim(-0.02, 1.02)
    #pylab.ylim(-0.02, 1.02)
    #pylab.show()
    symbols = ('o', 's', '^')

    esd_host = numpy.array([chain[keys == 'esd_host%d' %i][0][burn:]
                            for i in xrange(1, 4)])
    esd_host_err = numpy.std(esd_host, axis=1)
    esd_host = numpy.median(esd_host, axis=1)
    error_total = numpy.hypot(signal_err, esd_host_err)
    R = logspace(-2, 1, 100)
    z = numpy.array([0.17, 0.19, 0.21])

    fig, ax = pylab.subplots(figsize=(5.5,5))
    ax.set_xscale('log')
    re = []
    shift = 0.04
    for i, color, symbol, dx in izip(count(), colors, symbols, (-1,0,1)):
        re.append(fit_sis(Ro, signal[i]-esd_host[i],
                          error_total[i], z[i]))
        ax.errorbar((1+dx*shift)*Ro, signal[i]-esd_host[i],
                    yerr=signal_err[i],
                    fmt=symbol, color=color, ms=7, mew=2, mec=color,
                    label='Bin %d' %(i+1))
        sis = sis_gammat(R, re[-1][0]) * sigma_c(z[i], 0.65)
        ax.plot(R, sis, '-', color=color, zorder=-5)
    re = numpy.array(re)
    re_print = tuple(1e3 * numpy.reshape(re, -1))
    print r'  rE = $%.2f\pm%.2f$ & $%.2f\pm%.2f$ & $%.2f\pm%.2f$ \\ kpc/h' \
          %re_print
    diff = ufloat(re[2][0], re[2][1]) - \
           ufloat(re[0][0], re[0][1])
    print '    (3-1) =', diff,
    print '(%.2f sigma)' %(diff.nominal_value / diff.std_dev)
    ax.set_xlabel(r'$R\,({\rm Mpc})$')
    ax.set_ylabel(r'$\Delta\Sigma_{\rm sub}\,(h\,\mathrm{M_\odot pc^{-2}})$')
    #ax.set_ylabel(r'$\Delta\Sigma\,(h\,\mathrm{M_\odot pc^{-2}})$')
    ax.set_xlim(2e-2, 0.5)
    ax.set_ylim(-20, 100)
    ax.xaxis.set_major_locator(ticker.FixedLocator((0.02, 0.1, 0.5)))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(ticker.FixedLocator((0, 50, 100)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    legend = ax.legend(loc='upper right', numpoints=1)
    legend.get_frame().set_alpha(0)
    fig.tight_layout()
    output = args.chainfile.replace('outputs/', 'plots/')
    output = output.replace('.fits', '_satsignal.pdf')
    pylab.savefig(output, format=output[-3:])
    print 'Saved to', output
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 100)
    output = output.replace('.pdf', '_log.pdf')
    pylab.savefig(output, format=output[-3:])
    pylab.close()
    print 'Saved to', output
    return


def calc_ratios(x, y, xlo=None, xhi=None, ylo=None, yhi=None):
    n = x.shape
    if xlo is None:
        xlo = numpy.zeros(n)
    if xhi is None:
        xhi = numpy.zeros(n)
    if ylo is None:
        ylo = numpy.zeros(n)
    if yhi is None:
        yhi = numpy.zeros(n)
    ratio = x / y
    try:
        ratio_lo = [(ufloat(xi, xloi) / ufloat(yi, yhii)).s
                    for xi, xloi, yi, yhii in izip(x, xlo, y, yhi)]
        ratio_hi = [(ufloat(xi, xhii) / ufloat(yi, yloi)).s
                    for xi, xhii, yi, yloi in izip(x, xhi, y, ylo)]
    except AssertionError:
        ratio_lo = [(ufloat((xi, xloi)) / ufloat((yi, yhii))).std_dev()
                    for xi, xloi, yi, yhii in izip(x, xlo, y, yhi)]
        ratio_hi = [(ufloat((xi, xhii)) / ufloat((yi, yloi))).std_dev()
                    for xi, xhii, yi, yloi in izip(x, xhi, y, ylo)]
    #print ratio_lo
    #print ratio_hi
    #exit()
    return ratio, numpy.array(ratio_lo), numpy.array(ratio_hi)


def chi2(model, esd, esd_err):
    return (((esd - model) / esd_err) ** 2).sum()


def convert_groupmasses(chain, keys, hdr):
    izip = izip
    h = cosmology.h
    keys = list(keys)

    to = time()
    jfc = keys.index('fc_group')
    for i in xrange(1, 4):
        z = hdr[4][hdr[0] == 'zgal%d' %i][0]
        jm = keys.index('Mgroup%d' %i)
        m200c = 10**chain[jm]
        m200a = [profiles.nfw(M, z, c=fc, ref_in='200c', ref_out='200a')
                for M, fc in izip(m200c, chain[jfc])]
        m200a = numpy.array(m200a)
        chain[jm] = log10(m200a)
    ratio = m200a / m200c
    median = numpy.median(ratio)
    print '<m200a/m200c> = %.3f -%.3f +%.3f' \
          %(median, median-numpy.percentile(ratio, 16),
            numpy.percentile(ratio, 84)-median)
    rhoc = cosmology.density(z, ref='critical')
    rhoa = cosmology.density(z, ref='average')
    r200c = (m200c / (4*numpy.pi/3) / (200*rhoc)) ** (1./3.)
    r200a = (m200a / (4*numpy.pi/3) / (200*rhoa)) ** (1./3.)
    Ac = 5.71 / (m200c/(2e12/h))**0.084 / (1+z)**0.47
    Aa = 10.14 / (m200a/(2e12/h))**0.089 / (1+z)**1.01
    fca = (r200a / r200c) * (Ac / Aa)
    chain[jfc] *= fca
    print 'converted masses and concentrations in', (time()-to)/60, 'minutes'
    return chain


def cumavg(value, samples=100):
    avg = [numpy.median(value[:i])
           for i in linspace(0, len(value), samples)]
    return numpy.array(avg)


def format_axes(ax, observable, y, yerr=None, xlabels=True):
    if xlabels:
        xlim = ax.get_xlim()
        if ax.get_xscale() == 'log' and xlim[0] > 1e-4 and xlim[1] < 1e4:
            if xlim[0] >= 1:
                fmt = ticker.FormatStrFormatter('$%d$')
            else:
                fmt = ticker.FormatStrFormatter('$%s$')
            ax.xaxis.set_major_formatter(fmt)
    ylim = ax.get_ylim()
    if ax.get_yscale() == 'log' and ylim[0] > 1e-4 and ylim[1] < 1e4:
        if ylim[0] >= 1:
            fmt = ticker.FormatStrFormatter('$%d$')
        else:
            fmt = ticker.FormatStrFormatter('$%s$')
        ax.yaxis.set_major_formatter(fmt)
    # to be used only in log plots
    if yerr is None:
        yerr = numpy.zeros(len(x))
    if observable == 'logLstar':
        ax.set_xlim(0.02, 10)
    elif observable == 'logmstar':
        ax.set_xlim(1e9, 5e11)
    return


def massobs_output(observable, output_path, literature, mass_trunc, mass_rbg,
                   ratio, show_mstar, radius_3d, xlog, ylog, bw, mass_host='mcmc'):
    output = os.path.join(output_path, 'mass')
    if mass_trunc:
        output += '_trunc'
    elif mass_rbg:
        output += '_rbg'
    if ratio:
        output += '_ratio'
    if show_mstar:
        output += '_mstar'
    if observable == 'distBCG' and radius_3d:
        output += '_r3d'
    if mass_host != 'mcmc':
        output += '_mhost{0:.0e}'.format(mass_host).replace('+','')
    if xlog:
        output += '_xlog'
    if ylog:
        output += '_ylog'
    #if norm:
        #output = output.replace('.'+ext, '_norm.'+ext)
    if literature:
        names = []
        for i, source in enumerate(literature.split(',')):
            if source == 'eagle':
                names.append('EAGLE')
                continue
            if source[:3] == 'van':
                name = 'v{0}'.format(source[3].upper())
            elif source[:2] == 'vd':
                name = 'vd{0}'.format(source[2].upper())
            else:
                name = source[0].upper()
            names.append('{0}{1}'.format(name, source[-2:]))
        output += '-{0}'.format('_'.join(names))
    if bw:
        output = '{0}-bw'.format(output)
    return output


def plot_fit(ax, x1, x2, kind='kelly', colors='ry', logify=True):
    def model(x, a, b):
        return 10**a * x**b
    x1, x1err, x1p = x1
    x2, x2err = x2
    if ax.get_xscale() == 'log':
        t = linspace(0.5*x1.min(), 2*x1.max(), 100)
    else:
        size = (x1+x1err).max() - (x1-x1err).min()
        t = linspace((x1-x1err).min() - 0.1*size,
                           0.1*size + (x1+x1err).max(), 100)
    if kind == 'kelly':
        fit = lnr.kelly(x1/x1p, x2, x1err=x1err/x1p, x2err=x2err,
                        logify=logify)
        xo = []
        for x in fit:
            xo.append([numpy.median(x)])
            xo[-1].append(xo[-1][0] - numpy.percentile(x, 16))
            xo[-1].append(numpy.percentile(x, 84) - xo[-1][0])
        afit, bfit, sfit = xo
    elif kind == 'bces':
        # should not bootstrap because there are so few points - and
        # because they are averages not individual measurements
        fit = lnr.bces(x1/x1p, x2, x1err=x1err/x1p, x2err=x2err,
                       logify=logify, model='yx', full_output=True,
                       bootstrap=False)
        afit, bfit, covfit = fit
        print 'cov12 =', covfit[1][0]
        logx1, logx1err = lnr.to_log(x1/x1p, x1err/x1p)
        logx2, logx2err = lnr.to_log(x2, x2err)
        #sfit = ((x2 - model(x1/x1p, afit[0], bfit[0])) / x2err)**2
        #sfit = [numpy.sqrt(sfit.sum())]
        sfit = ((logx2 - (afit[0] + bfit[0]*logx1)))**2 - \
               numpy.median(logx2err**2)
        sfit = [numpy.sqrt(sfit.sum())]
    elif kind == 'mle':
        fit = lnr.mle(x1/x1p, x2, x1err=x1err/x1p, x2err=x2err,
                      logify=logify, po=(12.,0.8,0.1))
        afit, bfit, sfit = fit
    yfit = model(t/x1p, afit[0], bfit[0])
    lnr.plot(t, afit[0], bfit[0], a_err=afit[1:], b_err=bfit[1:], zorder=-5,
             #pivot=x1p, log=logify, ax=ax, color=colors, lw=4, alpha=0.5)
             pivot=x1p, log=logify, ax=ax, color=('0.3','0.7'), lw=4, alpha=1)
    #legend = ax_with_fit.legend(loc='upper left', fontsize=14)
    #legend.get_frame().set_alpha(0)
    return afit, bfit, sfit


def plot_literature(
        args, ax, xvalues, observable, path='literature/', norm=1, xnorm=1,
        mass='Msat_rbg', literature='leauthaud12', bw=False, show_legend=True,
        h=0.7):
    def log2lin(logm, dlogm_lo, dlogm_hi):
        m = 10**logm
        mlo = m - 10**(logm - dlogm_lo)
        mhi = 10**(logm + dlogm_hi) - m
        return m, mlo, mhi

    curves = []
    curve_labels = []

    ncols = 2
    if observable == 'distBCG':
        Mstar_kids = 10**array([10.45, 10.51, 10.66]) / 0.7**2
        Rsat_kids = array([0.12, 0.25, 0.43]) / 0.7
        if 'vdBosch16' in literature or 'pvdBosch16' in literature:
            if bw:
                colors = ['0.5', '0.7']
            else:
                colors = ['C1', '0.7']
            # MENeaCS prediction
            filename = os.path.join(path, 'vdbosch16_mass_rproj.txt')
            vdb = readfile.table(filename)
            wang = readfile.table(
                'literature/wang13_Minfall_nsub_unevolved.txt')
            mstars = utils.read_avgs(
                         'logmstar', chainfile=args.chainfile)[0]
            if 'pvdBosch16' in literature:
                mstars = [10**numpy.array([10.68, 10.72, 10.78]), mstars]
                xval = [numpy.array([0.3, 0.6, 1.0]), xvalues]
            else:
                mstars = [mstars]
                xval = [xvalues]
            for im, xv, mstar in izip(count(), xval, mstars):
                # do a linear extrapolation if necessary
                # this assumes all x-arrays are sorted
                print 'xvalues =', xv
                if xv[0] < vdb[0][0]:
                    slope = (vdb[1][1]-vdb[1][0]) / (vdb[0][1]-vdb[0][0])
                    amp = vdb[1][0] - slope*vdb[0][0]
                    vdb[0] = numpy.append(xv[0], vdb[0])
                    vdb[1] = numpy.append(amp + slope*xv[0], vdb[1])
                if xv[-1] > vdb[0][-1]:
                    slope = (vdb[1][-1]-vdb[1][-2]) / (vdb[0][-1]-vdb[0][-2])
                    amp = vdb[1][-1] - slope*vdb[0][-1]
                    vdb[0] = numpy.append(vdb[0], xv[-1])
                    vdb[1] = numpy.append(vdb[1], amp + slope*xv[-1])
                macc = numpy.zeros(mstar.size)
                y = numpy.zeros(mstar.size)
                for i, m in enumerate(mstar):
                    j = numpy.argmin(abs(wang[0]-m))
                    macc[i] = wang[1][j]
                    m_over_mstar = vdb[1] * macc[i] / m
                    ratio = interp1d(vdb[0], m_over_mstar)
                    y[i] = ratio(xv[i])
                print('y =', y)
                if im == 1:
                    # the 20% is made up. See how it compares to e.g.
                    # spread in Mstar for the paper
                    ax.fill_between(xv, 0.8*y, 1.2*y, color=colors[0],
                                    lw=0, zorder=-10)
                else:
                    ax.fill_between(
                        xv, 0.8*y, 1.2*y, facecolor='none', lw=2, zorder=-9,
                        edgecolor='C6', linestyle='--')#dashes=(8,6))
            ax.plot([], [], '-', color=colors[0], lw=8,
                    label='vdB+16 + W+13 prediction')
            readfile.save(
                'literature/vdb16_w13_m_mstar.txt', [xvalues, mstar, macc, y],
                fmt='%.2e', header='# r/r200h  mstar  macc  msub/mstar')
        # KiDS x GAMA (Sifon+15)
        if 'sifon15' in literature:
            # note that this is mbg (see background\ density.ipynb)
            if 'bg' in mass:
                y = [5.83, 8.95, 16.03]
                yerr = [[3.15, 4.96, 6.83], [4.38, 6.69, 8.71]]
            else:
                y = [24.5, 21.4, 33.1]
                yerr = [[11.2, 9.5, 19.1],[42.7, 37.2, 51.3]]
            ax.errorbar([0.17, 0.35, 0.42], y, yerr=yerr,
                        fmt='s', color='0.5', mec='0.5', ms=7, elinewidth=1.5,
                        label=r"Sif\'on+15")
                        #label='KiDSxGAMA (Chapter 5)')
        if 'li16' in literature:
            # assume Mhalo = 1e14 Msun
            rh = conversions.rsph(1e14, 0.3, ref='200c')
            x = array([0.2, 0.4, 0.7]) / rh
            xerr = array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.2]]) / rh
            y = [4.4, 17.2, 54.6]
            yerr = [[2.2, 6.8, 15.8], [6.6, 7.0, 15.6]]
            # not showing xerr
            ax.errorbar(
                x, y, yerr=yerr, fmt='C6s', mfc='none', ms=6, capsize=2,
                mew=1.5, elinewidth=1.5, zorder=-8, label='Li+16')
            ax.set_xlim(0.05, 1.5)
        ax.errorbar([2], [2], yerr=[1], fmt='ko', ms=9, mew=2, capsize=2,
                    label='This work')
        legend = ax.legend(loc='upper left', ncol=1, fontsize=15,
                           numpoints=1, frameon=False)
        rcParams['mathtext.rm'] = 'serif'

    elif observable == 'logmstar':
        if len(literature) > 0:
            ax.set_xlim(1e9, 1e12)

        lines = []
        labels = []
        # Leauthaud et al. (2012) - COSMOS GGL + clustering + SMF
        if 'leauthaud12' in literature:
            zbin = '1'
            zleau = {'1': 0.36, '2': 0.66, '3': 0.88}
            x = logspace(9, 11.7, 100) * (0.72/h)**2
            y = 10**leauthaud12.shmr(x, zbin) * (0.72/h)
            yerr = unumpy.std_devs(y)
            y = unumpy.nominal_values(y)
            #label = 'L+12 (z={0})'.format(zleau[zbin])
            label = 'Leauthaud+12'
            line, = ax.plot(x, y, 'k', dashes=(6,4), lw=2, label=label)
            curves.append(interp1d(log10(x), log10(y)))
            curve_labels.append('Leauthaud+12')
            lines.append(line)
            labels.append(label)
        # Velander et al. (2014) - CFHTLenS -- note this is M200c!
        if 'velander14' in literature:
            c = ('C4', 'C9')
            filename = os.path.join(path, 'velander2014_{0}.txt')
            for color, label in izip(c, ('red','blue')):
                fname = filename.format(label)
                v14 = readfile.table(fname)
                # stellar mass
                v14[1] *= 1e10 * (0.7/h)**2
                for i in xrange(2, 5):
                    v14[i] *= 1e11 * (0.7/h)
                # convert to m200a
                m200a = array(
                    [profiles.nfw(i, 0.32, ref_in='200c', ref_out='200a')
                     for i in v14[2]])
                factor = m200a / v14[2]
                v14[2:] *= factor
                #linelabel = 'V+14 {0} (z=0.32)'.format(label.capitalize())
                linelabel = 'Velander+14 {0}'.format(label.capitalize())
                line = ax.errorbar(
                    v14[1], v14[2], yerr=(v14[3],v14[4]), fmt='s',
                    ecolor=color, mec=color, mfc='none', mew=1.5,
                    elinewidth=1.5, capsize=2, zorder=20, label=linelabel)
                curves.append(interp1d(log10(v14[1]), log10(v14[2])))
                curve_labels.append('Velander+14 {0}'.format(label.capitalize()))
                lines.append(line)
                labels.append(linelabel)
                # don't show blue galaxies
                break
        if 'coupon15' in literature:
        #if True:
            c15 = readfile.table('literature/coupon15.txt')
            #label = 'C+15 (z=0.8)'
            label = 'Coupon+15'
            line, = ax.plot(c15[0], c15[1], '-', color='C3', label=label)
            lines.append(line)
            labels.append(label)
        if 'zu15' in literature:
            logx = numpy.linspace(10, 12, 100)
            logy = ihod.logMh(logx, h=h)
            line, = ax.plot(10**logx, 10**logy, color='C9', lw=3)
            #label = 'Z\&M15 (z=0.1)'
            label = 'Zu\&Mandelbaum15'
            lines.append(line)
            labels.append(label)
            curves.append(interp1d(logx, logy))
            curve_labels.append(label)
        # Mandelbaum et al. (2016, MNRAS, 457, 3200)
        if 'mandelbaum16' in literature:
            #c = (red, blue)
            c = ('C3', 'C0')
            filename = os.path.join(path, 'mandelbaum2015_%s.txt')
            for color, label, z in izip(c, ('red','blue'), ('0.13','0.11')):
                fname = filename %label
                m15 = readfile.table(fname)
                m15[0] = 10**m15[0] * (0.673/h)**2
                #y, ylo, yhi = log2lim(*m15[1:])
                m15[1:] = log2lin(*m15[1:])
                for i in xrange(1, len(m15)):
                    m15[i] *= (0.673/h)
                #linelabel = 'M+16 {0} (z={1})'.format(label.capitalize(), z)
                linelabel = 'Mandelbaum+16 {0}'.format(label.capitalize())
                line = ax.errorbar(
                    #m15[0], y*h, yerr=(ylo*h,yhi*h), fmt='o', ecolor=color,
                    m15[0], m15[1], yerr=(m15[2],m15[3]), fmt='o', ecolor=color,
                    mec=color, mfc='none', mew=1.5, elinewidth=1.5, capsize=2,
                    zorder=20, label=linelabel)
                curves.append(interp1d(log10(m15[0]), log10(m15[1])))
                curve_labels.append('Mandelbaum+16 {0}'.format(label.capitalize()))
                lines.append(line)
                labels.append(linelabel)
                # don't show blue galaxies
                break
        # Edo's KiDS paper
        if 'vanuitert16' in literature:
            color = 'C2'
            filename = os.path.join(path, 'vanuitert16.txt')
            data = readfile.table(filename)
            x = 10**data[0] * (0.7/h)**2
            y = 10**unumpy.uarray([data[1], data[2]/2]) * (0.7/h)
            yerr = unumpy.std_devs(y)
            y = unumpy.nominal_values(y)
            # the label for this is not showing
            ax.fill_between(x, y-yerr, y+yerr, color=color, zorder=-30)
            #label = 'vU+16 (z=0.2)'
            label = 'vanUitert+16'
            line, = ax.plot([], [], '-', color=color, lw=7, label=linelabel)
            curves.append(interp1d(log10(x), log10(y)))
            curve_labels.append('van Uitert+16')
            lines.append(line)
            labels.append(label)
        # Zu & Mandelbaum (2016, MNRAS, 457, 4360)
        if 'zu16' in literature:
            for color, label in izip(('C3','C0'), ('red', 'blue')):
                filename = os.path.join(
                    path, 'zu_mandelbaum_haloquench_%s.csv' %label)
                cent = readfile.table(filename)
                #linelabel = 'Z\&M16 {0}'.format(label.capitalize())
                linelabel = 'Zu\&Mandelbaum16'
                line, = ax.plot(cent[0]/h**2, cent[1]/h, '-', color=red,
                                zorder=1, label=linelabel)
                curves.append(interp1d(log10(cent[0]/h**2), log10(cent[1]/h)))
                curve_labels.append(
                    'Zu\&Mandelbaum16 {0}'.format(label.capitalize()))
                lines.append(line)
                labels.append(linelabel)
        if 'eagle' in literature:
            color = 'C0'
            filename = os.path.join(path, 'eagle_allstarssat.txt')
            data = readfile.table(filename)
            N = readfile.table(filename.replace('.txt', '_ngal.txt'))
            # N = Ngal * fsat
            N = N[0] * N[2]
            msub = 10**data[3] / h
            errlo = (msub - 10**data[4]/h) / N**0.5
            errhi = (10**data[5]/h - msub) / N**0.5
            # yes, only one h
            x = 10**data[0] / h
            ax.fill_between(x, msub-errlo, msub+errhi,
                            color=color, zorder=-20)
            ax.plot(x, msub-errlo, '-', color=color, lw=1, zorder=-4)
            ax.plot(x, msub+errhi, '-', color=color, lw=1, zorder=-4)
            #label = 'EAGLE (z=0.18)'
            label = 'Velliscig+16 (EAGLE)'
            line, = ax.plot([], [], '-', color=color, lw=7, label=label)
            curves.append(interp1d(log10(x), log10(msub)))
            curve_labels.append('EAGLE')
            lines.append(line)
            labels.append(label)
        if 'rodriguez13' in literature:
            filename = 'literature/rp13_Msub_C_nsub_unevolved.txt'
            rp = readfile.table(filename)
            #poor statistics will mess up the relation at high m
            imax = max([i for i in xrange(1, rp[0].size)
                        if rp[1][i] > rp[1][i-1]])
            # to account for proper indexing
            imax += 1
            print 'imax =', imax
            rp = [i[:imax] for i in rp]
            #label = 'RP+13 (z=0.15)'
            label = 'Rodriguez-Puebla+13'
            ## not sure if this is right, because I would need to
            ## adjust mstar as well, right?
            # adjusted for the different host halo masses,
            # for a slope 0.94 (should be changed if the slope changes)
            f = (5e14/1e13)**(1-0.94)
            # it's not very interesting to show if the conversion
            # is small
            if f > 1.5:
                ax.plot(rp[0], f*rp[1], '-', color='C5', lw=1, zorder=-4,
                        label='_none_')
                dashes = (8, 6, 2, 6)
            else:
                dashes = (10, 1e-5)
            line, = ax.plot(rp[0], rp[1], color='C5', dashes=dashes, lw=3,
                            zorder=-4, label=label)
            line, = ax.plot([], [], 'C5-', lw=3)
            #ax.fill_between(
                #rp[0], rp[1]-rp[2], rp[1]+rp[3], color='C5', lw=0, zorder=-5)
            curves.append(interp1d(log10(rp[0]), log10(rp[1])))
            curve_labels.append("Rodr\'iguez-Puebla+13")
            lines.append(line)
            labels.append(label)
        # Li et al. CS82
        if 'li16' in literature:
            x = array([10**10.2,10**11.4])
            y, ylo = lnr.to_linear([11.14,12.38], [0.73,0.16])
            yhi = lnr.to_linear([11.14,12.38], [0.66,0.16])[1]
            line = ax.errorbar(
                x/h, y/h, yerr=(ylo/h,yhi/h),
                #xerr=((10**0.2,10**0.3),(10**0.4,10**0.6)),
                fmt='C2^', mfc='none', ms=6, mew=1.5, lw=1.5, capsize=2)
            lines.append(line)
            labels.append('Li+16')
        # Niemiec et al. (not yet published)
        #if 'niemiec17' in literature:
            #y, ylo = lnr.to_linear([
        # My MENeaCS paper
        if 'sifon17' in literature:
            # update to use the latest results once I have them
            # this table only has bound masses
            filename = 'literature/sifon17.txt'
            s17 = readfile.table(filename)
            s17[0] = 10**s17[0]
            s17[1:] = log2lin(*s17[1:])
            label = r"MENeaCS satellites (Sifon+17)"
            line = ax.errorbar(
                s17[0], s17[1], yerr=(s17[2],s17[3]), fmt='ko', ms=7,
                capsize=2, elinewidth=2, mew=2, label=label)
            curves.append(interp1d(log10(s17[0]), log10(s17[1])))
            curve_labels.append("Sif\'on+17")
            lines.append(line)
            labels.append(label)
        # my UDG paper
        if 'udgs' in literature:
            filename = '../UDGs/output/fullnfw_log-csat_10_20-udgs-delmag' \
                       '-exclude_8_9_10.out'
            udg = readfile.table(filename)
            

        #label = 'MENeaCS (This work)'
        label = 'This work (MENeaCS)'
        line = ax.errorbar(
            [1], [1], yerr=[0.1], fmt='ko', ms=9, mew=2, capsize=2)
        lines.append(line)
        labels.append(label)

        #lines, labels = ax.get_legend_handles_labels()
        # construct legend(s)
        #curve_labels_short = [c.split()[0] for c in curve_labels]
        #centrals = ('L+12 (z=0.36)', 'L+12 (z=0.66)', 'L+12 (z=0.88)',
                    #'V+14 Red (z=0.32)', 'C+15',
                    #'M+16 Red (z=0.13)', 'M+16 Blue (z=0.11)',
                    #'Z\&M15 (z=0.1)', 'vU+16 (z=0.2)',
                    #'Z&M16 Red', 'Z&M Blue')
        #satellites = ('MENeaCS (this work, z=0.1)', 'RP+13 (z=0.15)',
                      #'EAGLE (z=0.18)', 'S+17 UDGs (z=0.07)')
        centrals = ('Leauthaud+12',
                    'Velander+14 Red', 'Velander+14 Blue', 'Coupon+15',
                    'Mandelbaum+16 Red', 'Mandelbaum+16 Blue',
                    'Zu\&Mandelbaum15', 'vanUitert+16',
                    'Zu&Mandelbaum16 Red', 'Zu&Mandelbaum16 Blue')
        satellites = ('Rodriguez-Puebla+13', 'Velliscig+16 (EAGLE)',
                      'Li+16', 'Niemiec+17', 'Sifon+17 UDGs', 'This work')
        lit = {'Centrals': array([labels.index(i) for i in centrals
                                  if i in labels]),
               'Satellites' : array([labels.index(i) for i in satellites
                                     if i in labels])}
        labels = array(labels)
        lines = array(lines)
        # there is always at least MENeaCS!
        if len(lit['Centrals']) > 0:
            samples = ['Centrals']
            if len(lit['Satellites']) > 0:
                samples.append('Satellites')
        else:
            samples = ['Satellites']
        print 'samples =', samples
        loc_text = [(0.05, 0.92), (0.36,0.06+0.06*len(lit[samples[-1]]))]
        print '**\nloc_text = {0}\n**'.format(loc_text)
        bbox = ((0.02,0.94), None)
        loc_legend = ('upper left', 'lower right')
        legends = []
        for i, sample in enumerate(samples):
            print 'sample:', sample
            ax.annotate(
                '{0}:'.format(sample), xy=loc_text[i], fontsize=16,
                 xycoords='axes fraction', ha='left', va='bottom')
            lkwargs = dict(loc=loc_legend[i], ncol=1, fontsize=14,
                           numpoints=1, frameon=False)
            if bbox[i] is not None:
                lkwargs['bbox_to_anchor'] = bbox[i]
            print 'kwargs =', lkwargs
            print 'lines =', lines[lit[sample]]
            print 'labels =', labels[lit[sample]]
            #try:
            if True:
                legends.append(
                    ax.legend(lines[lit[sample]], labels[lit[sample]],
                    **lkwargs))
                if i > 0:
                    pylab.gca().add_artist(legends[i-1])
            #except IndexError as err:
                #print err
            print

    return curves, curve_labels


def read_observable(chainfile):
    file = chainfile.split('/')[1]
    if file.split('.')[-1].lower() not in ('fits', 'fit'):
        msg = 'WARNING: file {0} does not have a FITS extension. Are' \
              ' you sure you want to continue? [y/N] '
        go_on = raw_input(msg)
        if go_on[0].lower() != 'y':
            sys.exit()
    observable = ''
    for i in file.split('-'):
        i = i.split('_')
        if i[0] in ('logLstar', 'logmstar', 'distBCG', 'redshift'):
            return i[0], i[1:]
    return None


def runningavg(value, thin=1000):
    t = numpy.arange(0, len(value), thin, dtype=int)
    avgs = [numpy.median(value[i*thin:(i+1)*thin]) for i in xrange(len(t)-1)]
    x = [(t[i]+t[i-1])/2 for i in xrange(1, len(t))]
    return numpy.array(x), numpy.array(avgs)


def values(data, burn=0, log=False):
    data = data[burn:]
    m = numpy.median(data)
    if log:
        m = log10(m)
        p16 = m - log10(numpy.percentile(data, 16))
        p84 = log10(numpy.percentile(data, 84)) - m
    else:
        p16 = numpy.percentile(data, 16)
        p84 = numpy.percentile(data, 84)
    return m, m-p16, p84-m


# fit an sis
def fit_sis(R, esd, esd_err, zl, zs=0.65, po=1.):
    from scipy import optimize
    sc = sigma_c(zl, zs)
    gammat = esd / sc
    gammat_err = esd_err / sc
    model = lambda r, re: sis_gammat(r, re)
    re, var = optimize.curve_fit(model, R, gammat, po,
                                 sigma=gammat_err, absolute_sigma=True)
    re_err = numpy.sqrt(var[0][0])
    #print 'rE =', re[0], '+/-', re_err
    return re[0], re_err
def sigma_c(zl, zs):
    from astropy import constants, units as u
    beta = cosmology.dA(zl, zs) / (cosmology.dA(zl) * cosmology.dA(zl, zs))
    beta /= u.Mpc
    sc = constants.c**2 / (4*numpy.pi*constants.G)
    return (sc.to(u.Msun/u.pc) * beta.to(1/u.pc)).value
def sis_gammat(r, re):
    if re <= 0:
        return numpy.inf
    return re / (2*r)


def get_axlabel(name):
    labels = {'c_c200': r'$c_{c200}$',
              'fcsat': r'$f_{\rm c}^{\rm sub}$',
              'fchost': r'$f_{\rm c}^{\rm host}$',
              'Msat': r'$m_{200}$',
              'Mhost': r'$M_{\rm cl}$',
              'Msat_rbg': r'$m_{\rm bg}$',
              'Msat_rt': r'$m_{\rm t}$'}
    keys = labels.keys()
    for key in keys:
        #if key[:3] == 'log':
        labels['log{0}'.format(key)] = r'$\log %s' %labels[key][1:]
        if 'sat' in key:
            labels[key.replace('sat', 'sub')] = labels[key]
    keys = labels.keys()
    try:
        int(name[-1])
        if name[:-1] in keys:
            i = keys.index(name[:-1])
            if '_' in labels[keys[i]]:
                label = labels[keys[i]].replace('}', ',%s}' %name[-1])
            else:
                label = labels[keys[i]][:-1] + '_%s$' %name[-1]
            return label
    except ValueError:
        pass
    if name in labels:
        return labels[name]
    return name.replace('_', '')


def get_colormap(cmap):
    # I'm assuming that it's either a string or a colormap instance
    if isinstance(cmap, basestring):
        try:
            cmap = getattr(cm, cmap)
        except AttributeError:
            cmap = getattr(colormaps, cmap)
    return cmap


def get_colors(array=None, vmin=0, vmax=1, n=0, cmap='viridis'):
    cmap = get_colormap(cmap)
    if array is None:
        array = linspace(vmin, vmax, n)
    cnorm = mplcolors.Normalize(vmin=0, vmax=1)
    scalarmap = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    return scalarmap.to_rgba(array)


def get_errorbars(data, good, burn, keys, key,
                  n=0, suffix='', levels=(16,84)):
    if n == 0:
        val = data[keys == key+suffix][0][good][burn:]
    elif n is None:
        n = 1
        while 'key{0}{1}'.format(key, n, suffix) in keys:
            n += 1
    if n > 0:
        val = [data[keys == '%s%d%s' %(key, i+1, suffix)][0][good][burn:]
               for i in xrange(n)]
        val = numpy.array(val)
    med = numpy.percentile(val, 50, axis=1)
    err = [numpy.absolute(numpy.percentile(val, p, axis=1) - med)
           for p in levels]
    if key[:3] == 'log':
        med = 10**med
        err = 10**numpy.array(err)
    return med, err


def get_label(observable, lo, hi, fmt='%.2f', last=False):
    labels = {'distBCG': r'R_{\rm sat}/{\rm Mpc}',
              'logmstar': r'\log m_\star/{\rm M}_\odot',
              'logLstar': r'\log L/L^*'}
    label = labels[observable]
    label = '$' + fmt %lo + r'\leq %s < ' %label + fmt %hi + '$'
    if last:
        label = label.replace('<', r'\leq')
    return label


def get_plotkeys(model, params):
    print '[in get_plotkeys] model =', model
    mass_names = ['Msat', 'Msat_rbg'] + \
                 ['Menclosed{0}'.format(i) for i in xrange(1, 4)]
    # this is for single bin only
    if model in ('fullnfw_cfree', 'fullnfw_cfree_log', 'fullnfw_log'):
        plot_keys = [['Msat', 'Mhost'], ['Msat_rbg', 'Mhost'],
                     ['csat', 'Msat', 'Mhost'],
                     ['csat', 'Msat_rbg', 'Mhost'],
                     ['csat', 'Msat', 'chost', 'Mhost'],
                     ['csat', 'Msat_rbg', 'chost', 'Mhost']]
        names = [[''], [''], [''], [''], [''], ['']]
        plot_names = ['corner_mass', 'corner_mass_rbg', 'corner',
                      'corner_rbg', 'corner_chost', 'corner_rbg_chost']
    elif model == 'fullnfw_csat':
        plot_keys = [[''], ['fchost']]
        names = [['csat', 'Msat_rbg'], ['csat', 'Msat_rbg', 'Mhost']]
        plot_names = ['corner_sat', 'corner']
    elif model == 'fullnfw_csat_cMhost':
        plot_keys = [[''], ['achost', 'bchost'], ['achost', 'bchost']]
        names = [['csat', 'Msat_rbg'], ['Mhost'],
                 ['Mhost', 'csat', 'Msat_rbg']]
        plot_names = ['corner_sat', 'corner_host', 'corner']
    elif model == 'fullnfw_cMsat':
        plot_keys = [['asat', 'bsat', 'fchost'], ['asat', 'bsat']]
        names = [['Msat', 'Mhost'], ['Msat']]
        plot_names = ['corner', 'corner_sat']
    elif model[:10] == 'fullnfw_fc':
        plot_keys = [['fcsat'], ['fcsat', 'fchost']]
        names = [['Msat'], ['Msat', 'Mhost']]
        plot_names = ['corner_sat', 'corner']
    elif model[:30] == 'fullnfw_moline16_cfixed_cMhost':
        plot_keys = [[''], ['achost', 'bchost']]
        names = [['Msat_rbg'], ['Mhost', 'Msat_rbg']]
        plot_names = ['corner_sat', 'corner']
    elif model[:23] == 'fullnfw_moline16_cfixed':
        plot_keys = [[''], ['fchost']]
        names = [['Msat_rbg'], ['Mhost', 'Msat_rbg']]
        plot_names = ['corner_sat', 'corner']
    #elif model == 'fullnfw_moline17_uniform_cMhost':
        #plot_keys = [['c_c200', 'achost', 'bchost'], ['c_c200']]
        #names = [['Msat_rbg'], ['Msat_rbg', ['Mhost'], ['Msat'], ['Msat', 'Mhost']
        #names = [[name]+[name,'Mhost'] for name in mass_names]
        #plot_keys = [['c_c200', 
    elif model[:17] == 'fullnfw_moline17c':
        plot_keys = [['c_c200', 'fchost'], ['c_c200']]
        if 'bc' in model:
            for i in xrange(len(plot_keys)):
                plot_keys[i].append('b_c200')
        names = [['Msat_rbg', 'Mhost'], ['Msat_rbg']]
        plot_names = ['corner', 'corner_sat']
    elif model[:7] == 'fullnfw':
        plot_keys = [[''], [''], ['']]
        names = [['Msat'], ['Msat_rbg'], ['Msat_rbg', 'Mhost']]
        plot_names = ['corner_sat', 'corner_sat_rbg', 'corner']
    elif model[:5] == 'sshmr':
        plot_keys = [['logm0', 'exponent', 'fch']]
        names = [['Mh']]
        plot_names = ['corner']
    elif model[:10] == 'tnfw_fixed':
        plot_keys = [['fcsat', 'fchost'], ['fcsat'], ['fcsat']]
        names = [['rt', 'Msat_rt', 'Mhost'], ['rt', 'Msat_rt'], ['Msat_rt']]
        plot_names = ['corner', 'corner_sat', 'corner_sat_mass']
    elif model == 'tnfw_fc':
        plot_keys = [['fcsat', 'Art', 'Brt', 'fchost'],
                     ['fcsat', 'Art', 'Brt'], ['fcsat', 'Art', 'Brt']]
        names = [['Msat_rt', 'Mhost'], ['Msat_rt'], ['Msat_rt', 'rt']]
        plot_names = ['corner', 'corner_sat_mass', 'corner_sat']
    elif model[:10] == 'tnfw_fixed':
        plot_keys = [['fcsat', 'fchost'], ['fcsat']]
        names = [['Msat_rt', 'Mhost'], ['Msat_rt', 'rt']]
        plot_names = ['corner_masses', 'corner_sat']
    elif model[:11] == 'tnfw_theory':
        plot_keys = [['fchost', 'fcsat', 'Art'], ['fcsat', 'Art'], ['fcsat']]
        names = [['Msat_rt', 'Mhost'], ['Msat_rt'], ['Msat_rt', 'rt']]
        plot_names = ['corner_masses', 'corner_sat', 'corner_sat_rt']
        if 'Brtfixed' not in model:
            for i in xrange(len(plot_keys)):
                if 'Art' in plot_keys[i]:
                    plot_keys[i].append('Brt')
    else:
        msg = 'ERROR in get_plotkeys:'
        msg += ' plot keys for model {0} not specified'.format(model)
        print msg
        return
    # so that they always have two dimensions
    if isinstance(plot_keys[0], basestring):
        plot_keys = [plot_keys]
    if isinstance(names[0], basestring):
        names = [names]
    if isinstance(plot_names, basestring):
        plot_names = [plot_names]
    if 'Msat_rbg1' in params:
        for j in xrange(len(names)):
            if 'Msat' in names[j]:
                plot_keys.append([i for i in plot_keys[j]])
                names.append(['Msat_rbg' if name == 'Msat' else name
                              for name in names[j]])
                plot_names.append('{0}_bg'.format(plot_names[j]))
    for j, name_set in enumerate(names):
        if not plot_keys[j][0]:
            plot_keys[j] = []
        for name in name_set:
            i = 1
            while '{0}{1}'.format(name, i) in params:
                plot_keys[j].append('{0}{1}'.format(name, i))
                i += 1
            if i == 1 and '_' in name:
                while '{1}{0}_{2}'.format(i, *name.split('_')) in params:
                    plot_keys[j].append('{1}{0}_{2}'.format(
                                            i, *name.split('_')))
                    i += 1
    return plot_keys, plot_names


def read_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('chainfile')
    add('--burn', dest='burn', default=2000000, type=int,
        help='Size of burn-in sample')
    add('--corner', dest='corner', nargs='*', default='',
        help='Create additional corner plot(s) with provided parameters')
    add('--literature', dest='literature', nargs='?', default='',
        help='Comma-separated list of which lit. results to show')
    add('--observable', dest='observable', default=None)
    add('--output-path', dest='output_path', default=None)
    add('--scale', dest='scale', default=False,
        help='syntax: <param>,<pivot>,<exp>[,exp[10]]')
    add('--udg', dest='udg', action='store_true')
    args = parser.parse_args()
    if args.scale is not False:
        args.scale = args.scale.split(',')
        if len(args.scale) == 3:
            args.scale.append('')
    #if args.udg:
        #args.burn = 100000
    # set args.scale automatically
    hdr = open(args.chainfile.replace('.fits', '.hdr'))
    for line in hdr:
        line = line.split()
        if line[0] != 'datafile':
            continue
        if 'scale' not in line[1]:
            break
        for folder in line[1].split('/'):
            for value in folder.split('-'):
                if value[:5] != 'scale':
                    continue
                args.scale = value.split('_')[1:]
                break
    if 'udg' in args.chainfile:
        args.udg = True
    return args


if __name__ == '__main__':
    main()


