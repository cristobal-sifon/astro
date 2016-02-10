#!/usr/bin/env python
import glob
import itertools
import numpy
import os
import plottools
import pylab
import readfile
import sys
import stattools
from astro.clusters import conversions
from astropy.io import fits
from matplotlib import cm, colors as mplcolors, ticker
from scipy import optimize, stats
from uncertainties import ufloat

# local
import models
import nfw
import utils

from astro import cosmology
cosmology.h = 1
cosmology.Omega_M = 0.315
cosmology.Omega_L = 0.685
h = cosmology.h
Om = cosmology.Omega_M
Ol = cosmology.Omega_L

from matplotlib import rcParams
for tick in ('xtick', 'ytick'):
    rcParams['{0}.major.size'.format(tick)] = 8
    rcParams['{0}.minor.size'.format(tick)] = 4
    rcParams['{0}.major.width'.format(tick)] = 2
    rcParams['{0}.minor.width'.format(tick)] = 2
    rcParams['{0}.labelsize'.format(tick)] = 20
rcParams['axes.linewidth'] = 2
rcParams['axes.labelsize'] = 22
rcParams['legend.fontsize'] = 18
rcParams['lines.linewidth'] = 2
rcParams['mathtext.fontset'] = 'stix'
rcParams['pdf.use14corefonts'] = True
rcParams['text.usetex'] = True
rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']

red = (1,0,0)
green = (0.2,0.6,0)
blue = (0,0,1)
yellow = (1,1,0.2)
magenta = (1,0.4,0.6)
cyan = (0.2,0.6,1)
orange = (1,0.7,0)
purple = (0.8,0,0.4)
#blue = '#2D7EDF'
#red = '#D34A1E'
#green = '#36911C'
#purple = '#650D52'
#yellow = '#F6B91F'
#cNorm = mplcolors.Normalize(vmin=0, vmax=1)
#cmap = pylab.get_cmap('jet')
#scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
#cyan, red, blue = scalarMap.to_rgba((0.34, 0.88, 0.10))

#bcolor = [(1,0,0), (1,0.5,0), (1,0.7,0.4)]
#bcolor = [blue, (0,0.6,1), (0.5,0.7,1)]
#bcolor = [blue, (0.5,0.6,1), (0.6,0.9,1)]
#bcolor = [blue, (0.2,0.6,1), (0.6,0.9,1)]
#bcolor = ['0.4', '0.7']
#bcolor = [(0.1,0.5,1), (0.5,0.8,1), (0.8,0.9,1)]
bcolor = [orange, yellow, cyan]


def main(save_output=True, burn=100000, ext='pdf',
         corner_params=None):
    paramtypes = ('function', 'read', 'uniform', 'loguniform',
                  'normal', 'lognormal', 'fixed')
    chainfile = sys.argv[1]
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        output_path = 'mcmcplots'
    hdr = chainfile.replace('.fits', '.hdr')
    print 'Reading file', chainfile, '...'
    data = fits.getdata(chainfile)
    names = data.names
    good = (data.field('chi2') > 0) & (data.field('chi2') < 9999)
    chain, keys = numpy.transpose([(data.field(key)[good], key)
                                   for key in names])
    if 'linpriors' not in chainfile:
        for i, key in enumerate(keys):
            if 'Msat' in key or 'Mgroup' in key:
                chain[i] = 10**chain[i]
    lnlike = data.field('lnlike')[good]

    print len(lnlike), 'samples'
    # doing delta_chi2
    if 'chi2_total' in keys:
        chi2_key = 'chi2_total'
    else:
        chi2_key = 'chi2'
    j = list(keys).index(chi2_key)
    best = numpy.argmax(lnlike)
    bestn = numpy.argsort(lnlike)[-30:]
    print 'min(chi2) = %.2f at step %d' %(chain[j][best], best)
    print 'max(lnlike) = %.2f' %lnlike.max()
    print 'Best %d steps:' %len(bestn), bestn
    if burn > len(chain[0]) - 10000:
        burn = max(0, len(chain[0]) - 10000)
    best = numpy.argmax(lnlike[burn:])
    print 'min(chi2_burned) = %.2f at step %d' \
            %(chain[j][burn+best], burn+best)
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

    esds, esd_keys = numpy.transpose([(data.field('esd%d' %i)[good],
                                       'esd%d' %i)
                                      for i in xrange(1, 4)])
    chi2 = data.field(chi2_key)[good]

    out = plot_esd_verbose(chainfile, esds, esd_keys, best,
                           burn=burn, show=show,
                           save_output=save_output,
                           output_path=output_path, ext=ext)
    out = plot_esd(chainfile, chain, keys, esds, esd_keys, best,
                   burn=burn, show=show, save_output=save_output,
                   output_path=output_path, ext=ext)
    Ro, signal, signal_err, used, \
        median_signal, percent_signal, residuals = out


    corr = plot_covariance(chainfile, (len(esds),len(esds[0][0])),
                           corr=True, save_output=save_output, ext=ext)
    cov = plot_covariance(chainfile, (len(esds),len(esds[0][0])),
                          save_output=save_output,
                          output_path=output_path, ext=ext)
    Nobsbins, Nrbins = numpy.array(signal).shape
    # this copied from run.py to get chi2
    cov = numpy.transpose(cov, axes=(0,2,1,3))
    cov = cov.reshape((Nobsbins*Nrbins,Nobsbins*Nrbins))
    icov = numpy.linalg.inv(cov)
    icov = icov.reshape((Nobsbins,Nrbins,Nobsbins,Nrbins))
    icov = numpy.transpose(icov, axes=(2,0,3,1))
    chi2bf = numpy.array([numpy.outer(residuals[m], residuals[n]) * icov[m][n]
                          for m in xrange(Nobsbins)
                          for n in xrange(Nobsbins)]).sum()
    print 'chi2_bestfit = %.2f' %chi2bf


    plot_samples(chainfile, chain, keys, best, chi2bf,
                 len(Ro[used])*len(signal), burn=burn,
                 corner_params=corner_params, save_output=save_output,
                 output_path=output_path, ext=ext)
    #plot_satsignal(chainfile, chain, keys, Ro, signal, signal_err,
                   #burn=burn)
    #plot_massradius(chainfile, chain, keys, burn=burn,
                    #save_output=save_output, output_path=output_path, ext=ext)
    #plot_massradius(chainfile, chain, keys, burn=burn,
                    #save_output=save_output,
                    #output_path=output_path, ext=ext, norm=True)
    #plot_massradius(chainfile, chain, keys, burn=burn,
                    #save_output=save_output,
                    #output_path=output_path, ext='png', norm=True,
                    #Lietal=True)
    # I want to do this only for single-bin chains
    #if 'bin' in chainfile:
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
    out = utils.read_header(hdr)
    params, prior_types, sat_profile_name, group_profile_name, \
        val1, val2, val3, val4, \
        datafile, cols, covfile, covcol, \
        model, nwalkers, nsteps, nburn = out
    function = getattr(models, model)
    sat_profile = getattr(nfw, sat_profile_name)
    group_profile = getattr(nfw, group_profile_name)
    if datafile_index > -1:
        R, Ro, esd, esd_err, used = read_datafile(datafile[datafile_index],
                                                  cols)
        R = [R for i in xrange(len(datafile))]
        val1[params == 'Mgroup%d' %(datafile_index+1)] = Mgroup
        jM = (params == 'Msat%d' %(datafile_index+1))
        # this only for simulate_data.py
    else:
        R, Ro, esd, esd_err, used = read_datafile(datafile, cols)
        val1[params == 'Mgroup'] = Mgroup
        jM = (params == 'Msat')
    val2[params == 'fc_group'] = fc_group
    ja = (params == 'fc_sat')
    #setup
    k = 7
    Rrange = numpy.logspace(numpy.log10(0.99*Ro.min()),
                            numpy.log10(1.01*Ro.max()), 2**k)
    Rrange = numpy.append(0, Rrange)
    if datafile_index > -1:
        Rrange = [Rrange for i in xrange(len(datafile))]
    angles = numpy.linspace(0, 2*numpy.pi, 540)

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
        pylab.imshow(numpy.log10(chi2), origin='lower',
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

def plot_covariance(chainfile, shape, corr=False,
                    save_output=True, output_path='mcmcplots', ext='pdf'):
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
    # probably also need to divide the correlation matrix by 1+K?
    if corr:
        covcols[0] += 1
        suffix = 'corr'
        vmin = -0.15
        vmax = 0.7
        log = False
        cbarticks = numpy.arange(-0.1, 0.71, 0.1)
    else:
        suffix = 'cov'
        vmin = -2.5
        vmax = 3.5
        #log = True
        log = False
        cbarticks = numpy.arange(-2, 3.1, 1, dtype=int)
    #vmin = 0
    #vmax = 1
    #log = False
    cov = readfile.table(covfile, cols=covcols)
    if len(covcols) == 2:
        cov = cov[0] / cov[1]
    if corr:
        print 'max(corr) =', cov[cov < 1].max()
        print 'min(corr) =', cov.min()
    cov = cov.reshape((shape[0],shape[0],shape[1],shape[1]))

    ## this is what I have in run.py
    #cov = numpy.loadtxt(covfile, usecols=[covcol])
    ## 4-d matrix
    #cov = cov.reshape((shape[0],shape[0],shape[1],shape[1]))
    #print cov[0][1][4], cov.shape
    #exit()
    ## switch axes to have the diagonals aligned consistently
    #cov = numpy.transpose(cov, axes=(0,2,1,3))

    # now plot full covariances
    if log:
        cov = numpy.log10(cov)
    fig, axes = pylab.subplots(figsize=(4*shape[0],4*shape[0]),
                               nrows=shape[0], ncols=shape[0])
    for m, row in enumerate(axes):
        for n, ax in enumerate(row):
            title = r'$({0},{1})$'.format(shape[0]-m, n+1)
            img = ax.imshow(cov[m][2-n], extent=(0.02,2,0.02,2),
                            origin='lower', cmap=cm.CMRmap_r,
                            interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.set_title(title, fontsize=20)
    axes[2][1].set_xlabel(r'$R\,(h^{-1}{\rm Mpc})$', fontsize=30)
    axes[1][0].set_ylabel(r'$R\,(h^{-1}{\rm Mpc})$', fontsize=30)
    #fake_ax = pylab.axes([0.06, 0.07, 0.93, 0.92],
                         #xticks=[], yticks=[],
                         #axisbg='none', frameon=False)
    #fake_ax.set_xlabel(r'$R\,(h^{-1}{\rm Mpc})$', fontsize=30)
    #fake_ax.set_ylabel(r'$R\,(h^{-1}{\rm Mpc})$', fontsize=30)
    fig.subplots_adjust(left=0.10, bottom=0.10, right=0.96, top=0.96)
    fig.tight_layout()
    if save_output:
        output = os.path.join(output_path, output.split('/')[-1])
        output = output.replace('.fits', '_{0}.{1}'.format(suffix, ext))
        pylab.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    fig = pylab.figure(figsize=(2.5,10))
    ax = pylab.axes([0.01, 0.10, 0.88, 0.88], frameon=False,
                    xticks=[], yticks=[], axisbg='none')
    if log:
        label = r'$\log\,C_{mnij}$'
    else:
        label = r"$\boldsymbol{C'}_{mnij}$"
    cbar = fig.colorbar(img, ax=ax, fraction=.8, aspect=18, ticks=cbarticks)
    cbar.set_label(label=label, fontsize=30)
    for tl in cbar.ax.get_yticklabels():
        tl.set_fontsize(18)
    if save_output:
        output = output.replace('_{0}.{1}'.format(suffix, ext),
                                '_{0}_cbar.{1}'.format(suffix, ext))
        pylab.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    if log:
        return 10**cov
    return cov

def plot_esd_verbose(chainfile, esds, esd_keys, best, burn=10000,
                     percentiles=(2.5,16,84,97.5), show=False,
                     save_output=False, output_path='mcmcplots', ext='png'):
    # plot esd with extra information
    out = utils.read_header(chainfile.replace('.fits', '.hdr'))
    params, prior_types, sat_profile_name, group_profile_name, \
        val1, val2, val3, val4, \
        datafiles, cols, covfile, covcol, \
        model, nwalkers, nsteps, nburn = out
    if (type(datafiles) == str and len(esd_keys) == 1) or \
        len(datafiles) != len(esd_keys):
        msg = 'ERROR: number of data files does not match number of ESDs'
        print msg
        return
    n = len(datafiles)
    signal = [[] for i in xrange(n)]
    cross = [[] for i in xrange(n)]
    signal_err = [[] for i in xrange(n)]
    residuals = []
    for i, datafile in enumerate(datafiles):
        data = read_datafile(datafile, cols, covfile, covcol, i)
        R, Ro, signal[i], cross[i], signal_err[i], used = data
    pylab.figure(figsize=(5*n,4.5))
    xo = [0.12, 0.09, 0.06][n-1]
    axw = [0.85, 0.45, 0.305][n-1]
    yo = 0.16
    axh = 0.82
    inset_ylim = [(0, 50), (-10, 30), (-10, 30)]
    for i, s, serr, esd in itertools.izip(itertools.count(),
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
        ax.set_xlabel(r'$R\,(h^{-1}{\rm Mpc})$')
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
        output = os.path.join(output_path, output.split('/')[-1])
        output = output.replace('.fits', '_esd_verbose.' + ext)
        pylab.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    # plot normalized residuals
    #t = numpy.linspace(-3, 3, 100)
    #pylab.hist(res, bins=7, histtype='step', color=(0,0,1))
    #pylab.plot(t, numpy.exp(-t**2/2), 'k-')
    #pylab.show()
    residuals = numpy.array(residuals)
    out = (Ro, signal, signal_err, used,
           median_signal, per_signal, residuals)
    return out

def plot_esd(chainfile, chain, keys, esds, esd_keys, best, burn=10000,
             percentiles=(2.5,16,84,97.5),
             show=False, save_output=False,
             output_path='mcmcplots', ext='png'):
    # plot the ESD in the format that will go in the paper
    out = utils.read_header(chainfile.replace('.fits', '.hdr'))
    params, prior_types, sat_profile_name, group_profile_name, \
        val1, val2, val3, val4, \
        datafiles, cols, covfile, covcol, \
        model, nwalkers, nsteps, nburn = out
    if (type(datafiles) == str and len(esd_keys) == 1) or \
        len(datafiles) != len(esd_keys):
        msg = 'ERROR: number of data files does not match number of ESDs'
        print msg
        return
    n = len(datafiles)
    signal = [[] for i in xrange(n)]
    cross = [[] for i in xrange(n)]
    signal_err = [[] for i in xrange(n)]
    residuals = []
    res_bins = numpy.arange(-3, 3.1, 0.5)
    t = numpy.linspace(-3, 3, 100)
    gauss = numpy.exp(-t**2/2) / numpy.sqrt(2*numpy.pi)
    for i, datafile in enumerate(datafiles):
        data = read_datafile(datafile, cols, covfile, covcol, i)
        R, Ro, signal[i], cross[i], signal_err[i], used = data
    pylab.figure(figsize=(5*n,5.5))
    xo = [0.12, 0.09, 0.07][n-1]
    axw = [0.85, 0.45, 0.308][n-1]
    yo = 0.15
    axh = 0.82
    Rsat = [0.05, 0.20, 0.35, 1.0]
    Mstar = (10.45, 10.51, 10.66)
    t = numpy.logspace(-2, 0.7, 100)
    for i, s, x, serr, esd in itertools.izip(itertools.count(),
                                             signal, cross, signal_err, esds):
        ax = pylab.axes([xo+i*axw, yo, axw, axh], xscale='log')
        #for Mo in xrange(8, 13):
            #j = numpy.argmin(abs(chain[keys == 'Msat%d' %(i+1)][0] - Mo))
            #ax.plot(Ro[used], chain[keys == 'esd%i' %(i+1)][0][j], ':',
                    #color=red, zorder=20)
            #print 'Mo =', Mo, 'chi2 =', chain[keys == 'chi2'][0][j]
        #bestfit = numpy.median(esd[burn:], axis=0)
        bestfit = esd[burn:][best]
        residuals.append(s[used] - bestfit)
        per_signal = [numpy.percentile(esd[burn:], p, axis=0)
                      for p in percentiles]
        ax.errorbar(Ro[used], s[used], yerr=serr[used],
                    fmt='o', color='k', mec='k', mfc='k',
                    ms=10, mew=2, elinewidth=2, zorder=10)
        #ax.errorbar(Ro[used], x[used], yerr=serr[used],
                    #fmt='o', color='0.7', mec='0.7', mfc='none',
                    #ms=6, mew=1, elinewidth=1, zorder=9)
        #ax.plot(Ro[used], esd[best], '-', color='0.5')
        ax.plot(Ro[used], bestfit, '-', color='k')
        #ax.plot(Ro[used], per_signal[0], '-', color='k')
        #ax.plot(Ro[used], per_signal[3], '-', color='k')
        #ax.plot(Ro[used], per_signal[1], '-', color='k')
        #ax.plot(Ro[used], per_signal[2], '-', color='k')
        ax.fill_between(Ro[used], per_signal[0], per_signal[3],
                        color=bcolor[0], lw=0)
        ax.fill_between(Ro[used], per_signal[0], per_signal[1],
                        color=bcolor[1])
        ax.fill_between(Ro[used], per_signal[2], per_signal[3],
                        color=bcolor[1])
        ax.plot(t, 10**Mstar[i]/(numpy.pi*(1e6*t)**2), 'k--')
        #ax.plot(Ro[used], per_signal[0], '-', color='0.2', lw=1)
        #ax.plot(Ro[used], per_signal[3], '-', color='0.2', lw=1)
        label = r'$%.2f<R_{\rm sat}\leq%.2f$' %(Rsat[i], Rsat[i+1])
        if i == 2:
            label = label[:-2] + label[-1]
        ax.annotate(label, xy=(1.2,80), ha='right', va='center',
                    fontsize=22)
        if i == 0:
            ylabel = r'$\Delta\Sigma_{\rm sat}\,(h\,\mathrm{M_\odot pc^{-2}})$'
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
            ax.get_xticklabels()[1].set_visible(False)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if i == 0:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_xlabel(r'$R\,(h^{-1}{\rm Mpc})$')
        ax.set_xlim(0.02, 2)
        ax.set_ylim(-20, 100)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    if save_output:
        output = os.path.join(output_path, output.split('/')[-1])
        output = output.replace('.fits', '_esd.' + ext)
        pylab.savefig(output, format=ext)
        pylab.close()
        print 'Saved to', output
    elif show:
        pylab.show()
    residuals = numpy.array(residuals)
    out = (Ro, signal, signal_err, used,
           bestfit, per_signal, residuals)
    return out

def plot_samples(chainfile, data, keys, best, chi2, npts,
                 burn=10000, corner_params=None,
                 save_output=False, output_path='mcmcplots', ext='png'):
    from numpy import arange
    if 'chi2_total' in keys:
        chi2_key = 'chi2_total'
    else:
        chi2_key = 'chi2'
    data, keys = numpy.transpose([(data[keys == key][0], key)
                                  for key in keys if 'esd' not in key])
    Nparams = len(keys)
    avgs = []
    xmax = max([len(value) for value in data])
    fig, axes = pylab.subplots(figsize=(8,Nparams),
                               nrows=Nparams, sharex=True)
    for ax, value, key in itertools.izip(axes, data, keys):
        ax.plot(value, ',', color='0.7')
        to, avg = runningavg(value, thin=len(value)/50)
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
    pylab.tight_layout()
    fig.subplots_adjust(hspace=0)
    if save_output:
        output = os.path.join(output_path, output.split('/')[-1])
        output = output.replace('.fits', '_trace.png')
        pylab.savefig(output, format='png')
        pylab.close()
        print 'Saved to', output
        output = output.replace('.png', '.'+ext)
    else:
        output = ''
        pylab.show()
    # to use the results from above
    bgweight = numpy.arange(len(data[0]))
    data = numpy.array([x[burn:] for x in data])
    if save_output:
        output = output.replace('_trace', '_corner')
    if 'lnlike' in keys:
        best = numpy.argmax(data[keys == 'lnlike'][0])
    else:
        chi2sort = numpy.argsort(data[keys == chi2_key][0])
        best = numpy.argmin(data[keys == chi2_key][0])
    x = [numpy.median(data[keys == chi2_key])]
    x.append(x[0] - numpy.percentile(data[keys == chi2_key], 16))
    x.append(numpy.percentile(data[keys == chi2_key], 84) - x[0])
    #print x
    #print 'median chi^2 = %.2f_{-%.2f}^{+%.2f}' %tuple(x)
    #cat = chainfile[-6]
    #x = data[keys == chi2_key][0]
    data1 = numpy.array([data[keys == key][0] for key in keys1])
    #truths1 = [numpy.median(data[keys == key][0][best]/a)
               #for key, a in itertools.izip(keys1, norm)]
    #truths1 = [d[best] for d in data1]
    if 'offcenter' in chainfile:
        line = ''
    else:
        line = '0 & '
    logfile = chainfile.replace('.fits', '.out')
    log = open(logfile, 'w')
    print >>log, '# param  median  err_lo  err_hi'
    #logx1 = numpy.log10(numpy.array(x1))
    #logx2 = numpy.log10(numpy.array(x2))
    #x1err = numpy.log10(numpy.array(x1)+numpy.array(x1err)) - logx1
    #x2err = numpy.log10(numpy.array(x2)+numpy.array(x2err)) - logx2
    truths1 = []
    for key, d in itertools.izip(keys1, data1):
        m = numpy.median(d)
        if 'M' in key:
            logm = numpy.log10(m)
            vals = (logm, logm - numpy.log10(numpy.percentile(d, 16)),
                    numpy.log10(numpy.percentile(d, 84)) - logm)
            truths1.append(numpy.log10(d[best]))
        else:
            truths1.append(d[best])
        vals = (m, m - numpy.percentile(d, 16),
                numpy.percentile(d, 84) - m)
        print >>log, key.ljust(10),
        print >>log, '%.2e  %.2e  %.2e' %vals
        line += '$%.2f_{-%.2f}^{+%.2f}$' %vals
        if key == 'fc':
            line += '}'
        line += ' & '
        if 'fc' in key:
            line = '\n' + line + '\n'
    log.close()
    print 'Saved to', logfile
    if 'fiducial' in chainfile or 'rt' in chainfile:
        for i in xrange(1, 4):
            key = 'rt%d' %i
            d = data[keys == key][0]
            m = numpy.median(d)
            vals = (m, m - numpy.percentile(d, 16),
                    numpy.percentile(d, 84) - m)
            print key, '%.3f -%.3f +%.3f' %vals
    fc = numpy.median(data[keys == 'fc_group'][0])
    z = [0.17, 0.19, 0.21]
    for i in xrange(1, 4):
        m = numpy.median(data[keys == 'Mgroup%d' %i][0])
        c = fc * utils.cM_duffy08(m, z[i-1], h=h)
        rhom = utils.density_average(z[i-1], h=h, Om=Om)
        r200 = (m / (4*numpy.pi/3 * 200 * rhom)) ** (1./3.)
        print '%d) cgroup = %.1f, rsgroup = %.2f' %(i, c, r200/c)
    print '(chi2, npts, nparams) =', chi2, npts, len(keys1)
    line += '%.1f/%d \\\\' %(chi2, npts - len(keys1) - 1)
    print line
    print '**', len(data1), len((0.5,0.5,0.5,0.1,0.1,0.1,0.1))
    #lnlike = data[keys == 'lnlike'][0]
    #print truths1
    #for i, key in enumerate(keys1):
        #if 'M' in key:
            #data1[i] = numpy.log10(data1[i])
    #data1 = 
    corner = plottools.corner(data1, labels=labels1, bins=25, bins1d=50,
                              #clevels=(0.68,0.95,0.99),
                              #output=output,
                              ticks=ticks, limits=limits,
                              truths=truths1, truth_color=red,
                              truths_in_1d=True, medians1d=False,
                              percentiles1d=False,
                              #likelihood=lnlike, likesmooth=1,
                              style1d='step',
                              smooth=(0.5,0.5,0.5,0.1,0.2,0.2,0.2),
                              #smooth=(1,1,1,0.35,0.35,0.35,0.35),
                              background='filled',
                              linewidths=1, show_contour=True,
                              bcolor=bcolor[:2], verbose=True)
    for i, key in enumerate(keys1):
        if 'M' in key:
            data1[i] = 10 ** data1[i]
    fig, diagonals, offdiag = corner
    for ax in offdiag:
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        if ax.get_xlabel():
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if ax.get_ylabel():
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    for ax, key, d in itertools.izip(diagonals, keys1, data1):
        m = numpy.median(d)
        v68 = [numpy.percentile(d, i) for i in (16, 84)]
        v95 = [numpy.percentile(d, i) for i in (2.5, 97.5)]
        if 'M' in key:
            m = numpy.log10(m)
            v68 = [numpy.log10(v) for v in v68]
            v95 = [numpy.log10(v) for v in v95]
        #j = numpy.argmin(abs(
        #ax.axvline(m, color='k', lw=2)
        for v in v68:
            line = ax.axvline(v, color='k', lw=2)
            line.set_dashes([10,6])
        for v in v95:
            line = ax.axvline(v, color='k', lw=2)
            line.set_dashes([3,4])
        ax.xaxis.set_tick_params(width=2)
        if ax.get_xlabel():
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    if save_output:
        pylab.savefig(output, format=output[-3:])
        pylab.close()
        print 'Saved to', output

    # plot rt's and Msub's
    if 'fiducial' in chainfile:
        keys1 = ('rt1', 'rt2', 'rt3', 'Msat1_rt', 'Msat2_rt', 'Msat3_rt')
        data1 = numpy.array([data[keys == key][0] for key in keys1])
        for i in xrange(3, 6):
            #data1[i] /= 1e12
            data1[i] = numpy.log10(data1[i])
        labels = (r'$r_{\rm t,1}$', r'$r_{\rm t,2}$', r'$r_{\rm t,3}$',
                  r'$\log M_{{\rm sub},1}$', r'$\log M_{{\rm sub},2}$',
                  r'$\log M_{{\rm sub},3}$')
        truths1 = [d[best] for d in data1]
        limits = [(0, 0.25), (0, 0.25), (0, 0.3),
                  (10, 12.5), (10, 12.5), (10.5, 13)]
        ticks = [arange(0, 0.25, 0.1), arange(0, 0.25, 0.1),
                 arange(0.05, 0.26, 0.1), arange(10.5, 12.1, 0.5),
                 arange(10.5, 12.1, 0.5), arange(11, 12.6, 0.5)]
        corner = plottools.corner(data1, labels=labels, bins=25, bins1d=25,
                                  ticks=ticks, limits=limits,
                                  truths=truths1, truth_color=red,
                                  style1d='step',
                                  truths_in_1d=True, likelihood=lnlike,
                                  background='filled', linewidths=1,
                                  show_contour=True, bcolor=bcolor[:2])
        fig, diagonals, offdiag = corner
        for ax in offdiag:
            ax.xaxis.set_tick_params(width=2)
            ax.yaxis.set_tick_params(width=2)
            if ax.get_xlabel():
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            if ax.get_ylabel():
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        for ax in diagonals:
            ax.xaxis.set_tick_params(width=2)
            if ax.get_xlabel():
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if save_output:
            output1 = output.replace('_corner', '_corner_rt')
            pylab.savefig(output1, format=output1[-3:])
            pylab.close()
            print 'Saved to', output1

    if save_output:
        output = output.replace('_corner', '_corner_all')
        if burn == 0:
            output = output.replace('corner_all', 'corner_all_noburn')

    j = [i for i, d in enumerate(data) if numpy.std(d)]
    truths = [d[best] for d in data[j]]
    keys1 = [k.replace('_', '\_') for k in keys[j]]
    plottools.corner(data[j], labels=keys1, bins=25, bins1d=50,
                     clevels=(0.68,0.95,0.99), likelihood=lnlike,
                     truth_color=red, style1d='step',
                     truths=truths, background='filled', output=output,
                     top_labels=True, bcolor=bcolor, verbose=True)
    if save_output:
        print 'Saved to', output
    return

def plot_satsignal(chainfile, chain, keys, Ro, signal, signal_err,
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
    R = numpy.logspace(-2, 1, 100)
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
    ax.set_xlabel(r'$R\,(h^{-1}{\rm Mpc})$')
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
    output = chainfile.replace('outputs/', 'plots/')
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

def read_datafile(datafile, cols, covfile, covcols, obsbin):
    cols = numpy.append(cols, cols[1]+1)
    data = readfile.table(datafile, cols=cols)
    if len(cols) == 3:
        Ro, esd, cross = data
    elif len(cols) == 4:
        Ro, esd, oneplusk, cross = data
        esd /= oneplusk
        cross /= oneplusk
    if Ro[-1] > 500:
        Ro /= 1000
    R = numpy.append(0, Ro)
    cov = readfile.table(covfile, cols=covcols)
    if len(covcols) == 2:
        # cov[1] is (1+K)
        cov = cov[0] / cov[1]
    cov = cov.reshape((3,3,len(Ro),len(Ro)))
    err = numpy.sqrt(numpy.diag(cov[obsbin][obsbin]))
    used = numpy.ones(len(esd), dtype=bool)
    return R, Ro, esd, cross, err, used

def chi2(model, esd, esd_err):
    return (((esd - model) / esd_err) ** 2).sum()

def convert_groupmasses(chain, keys, hdr):
    from time import time
    from astro.clusters import profiles
    izip = itertools.izip
    h = cosmology.h
    keys = list(keys)

    to = time()
    hdr = utils.read_header(hdr)
    jfc = keys.index('fc_group')
    for i in xrange(1, 4):
        z = hdr[4][hdr[0] == 'zgal%d' %i][0]
        jm = keys.index('Mgroup%d' %i)
        m200c = 10**chain[jm]
        m200a = [profiles.nfw(M, z, c=fc, ref_in='200c', ref_out='200a')
                for M, fc in izip(m200c, chain[jfc])]
        m200a = numpy.array(m200a)
        chain[jm] = numpy.log10(m200a)
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
           for i in numpy.linspace(0, len(value), samples)]
    return numpy.array(avg)

def runningavg(value, thin=1000):
    t = numpy.arange(0, len(value), thin, dtype=int)
    avgs = [numpy.median(value[i*thin:(i+1)*thin]) for i in xrange(len(t)-1)]
    x = [(t[i]+t[i-1])/2 for i in xrange(1, len(t))]
    return numpy.array(x), numpy.array(avgs)

def values(data, burn=0, log=False):
    data = data[burn:]
    m = numpy.median(data)
    if log:
        m = numpy.log10(m)
        p16 = m - numpy.log10(numpy.percentile(data, 16))
        p84 = numpy.log10(numpy.percentile(data, 84)) - m
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

if __name__ == '__main__':
    main()
