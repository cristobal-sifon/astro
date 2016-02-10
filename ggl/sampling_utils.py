import numpy
import os
from glob import glob
#from ConfigParser import SafeConfigParser

def load_datapoints(datafile, datacols, exclude_bins=None):
    if type(datafile) == str:
        R, esd = numpy.loadtxt(datafile, usecols=datacols[:2]).T
        # better in Mpc
        if R[-1] > 500:
            R /= 1000
        if len(datacols) == 3:
            oneplusk = numpy.loadtxt(datafile, usecols=[datacols[2]]).T
            esd /= oneplusk
        if exclude_bins is not None:
            R = numpy.array([R[i] for i in xrange(len(R))
                             if i not in exclude_bins])
            esd = numpy.array([esd[i] for i in xrange(len(R))
                               if i not in exclude_bins])
    else:
        R, esd = numpy.transpose([numpy.loadtxt(df, usecols=datacols[:2])
                                  for df in datafile], axes=(2,0,1))
        if len(datacols) == 3:
            oneplusk = array([numpy.loadtxt(df, usecols=[datacols[2]])
                              for df in datafile])
            esd /= oneplusk
        for i in xrange(len(R)):
            if R[i][-1] > 500:
                R[i] /= 1000
        if exclude_bins is not None:
            R = numpy.array([[Ri[j] for j in xrange(len(Ri))
                              if j not in exclude_bins] for Ri in R])
            esd = numpy.array([[esdi[j] for j in xrange(len(esdi))
                                if j not in exclude_bins] for esdi in esd])
    return R, esd

def load_covariance(covfile, covcols, Nobsbins, Nrbins, exclude_bins=None):
    def format_covariance(C):
        C = C.reshape((Nobsbins,Nobsbins,Nrbins+nexcl,Nrbins+nexcl))
        if exclude_bins is not None:
            for b in exclude_bins[::-1]:
                C = numpy.delete(C, b, axis=3)
                C = numpy.delete(C, b, axis=2)
        return C
    cov, icov = numpy.loadtxt(covfile, usecols=covcols, unpack=True)
    if exclude_bins is None:
        nexcl = 0
    else:
        nexcl = len(exclude_bins)
    # 4-d matrix
    cov = format_covariance(cov)
    icov = format_covariance(icov)
    detC = numpy.array([numpy.linalg.det(cov[m][n])
                        for n in xrange(Nobsbins)
                        for m in xrange(Nobsbins)])
    norm = numpy.log(detC[detC > 0].prod())
    likenorm = -0.5 * (Nobsbins**2*numpy.log(2*numpy.pi) + norm)
    cov2d = numpy.hstack(numpy.hstack(cov))
    icov2d = numpy.hstack(numpy.hstack(icov))
    import pylab
    pylab.imshow(numpy.dot(cov2d, icov2d), interpolation='nearest')
    esd_err = numpy.sqrt(numpy.diag(cov2d)).reshape((Nobsbins,Nrbins))
    return cov, icov, likenorm, esd_err, cov2d
    #import pylab
    #from matplotlib import cm
    #fig, axes = pylab.subplots(figsize=(8,8), nrows=cov.shape[0],
                               #ncols=cov.shape[0])
    #p = numpy.percentile(cov, [1, 99])
    #for m, axm in enumerate(axes):
        #for n, axmn in enumerate(axm):
            #axmn.imshow(cov[m][-n-1][::-1], interpolation='nearest',
                        #cmap=cm.CMRmap_r, vmin=p[0], vmax=p[1])
                        ##vmin=-2, vmax=14)
    #fig.tight_layout()
    #pylab.draw()
    #cov2d = cov.transpose(0,2,1,3)
    #cov2d = cov2d.reshape((Nobsbins*Nrbins,Nobsbins*Nrbins))
    cov2d = numpy.hstack(numpy.hstack(cov[::-1])[::-1])
    #fig2, ax2 = pylab.subplots(figsize=(6,6))
    #ax2.imshow(cov2d[::-1], interpolation='nearest',
               #cmap=cm.CMRmap_r, vmin=p[0], vmax=p[1])
    #pylab.draw()
    #pylab.show()
    #exit()
    # product of the determinants
    detC = numpy.array([numpy.linalg.det(cov[m][n])
                        for n in xrange(Nobsbins)
                        for m in xrange(Nobsbins)])
    prod_detC = detC[detC > 0].prod()
    # likelihood normalization
    likenorm = -(Nobsbins**2*numpy.log(2*numpy.pi) + numpy.log(prod_detC)) / 2
    # errors are sqrt of the diagonal of the covariance matrix
    esd_err = numpy.sqrt(numpy.diag(cov2d)).reshape((Nobsbins,Nrbins))
    # reshape back into the desired shape (with the right axes order)
    icov = numpy.linalg.inv(cov2d)
    #p = numpy.percentile(icov, [1,99])
    #fig, ax = pylab.subplots(figsize=(5,5))
    #ax.imshow(icov[::-1], interpolation='nearest',
              #cmap=cm.CMRmap_r, vmin=p[0], vmax=p[1])
    #pylab.draw()
    icov = icov.reshape((Nobsbins,Nrbins,Nobsbins,Nrbins))
    icov = icov.transpose(0,2,1,3)
    icov = numpy.flipud(numpy.fliplr(icov))
    #icov = icov.transpose(2,0,3,1)
    #fig, axes = pylab.subplots(figsize=(7,7), ncols=icov.shape[0],
                               #nrows=icov.shape[0])
    #for m, axm in enumerate(axes):
        #for n, axmn in enumerate(axm):
            #axmn.imshow(icov[m][-n-1][::-1], interpolation='nearest',
                        #cmap=cm.CMRmap_r, vmin=p[0], vmax=p[1])
    #pylab.show()
    #exit()
    return cov, icov, likenorm, esd_err, cov2d

def read_config(config_file, version='0.5.7',
                path_data='', path_covariance=''):
    valid_types = ('normal', 'lognormal', 'uniform', 'exp',
                   'fixed', 'read', 'function')
    exclude_bins = None
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'path_data':
            path_data = line[1]
        if line[0] == 'data':
            datafiles = line[1]
            datacols = [int(i) for i in line[2].split(',')]
            if len(datacols) not in (2,3):
                msg = 'datacols must have either two or three elements'
                raise ValueError(msg)
        if line[0] == 'exclude_bins':
            exclude_bins = numpy.sort([int(b) for b in line[1].split(',')])
        if line[0] == 'path_covariance':
            path_covariance = line[1]
        elif line[0] == 'covariance':
            covfile = line[1]
            covcols = numpy.sort([int(i) for i in line[2].split(',')])
            if len(covcols) not in (1,2):
                msg = 'covcols must have either one or two elements'
                raise ValueError(msg)
        elif line[0] == 'sampler_output':
            output = line[1]
            if output[-5:].lower() != '.fits' and \
                output[-4:].lower() != '.fit':
                output += '.fits'
        # all of this will have to be made more flexible to allow for
        # non-emcee options
        elif line[0] == 'sampler':
            sampler = line[1]
        elif line[0] == 'nwalkers':
            nwalkers = int(line[1])
        elif line[0] == 'nsteps':
            nsteps = int(line[1])
        elif line[0] == 'nburn':
            nburn = int(line[1])
        elif line[0] == 'thin':
            thin = int(line[1])
        # this k is only needed for mis-centred groups in my implementation
        # so maybe it should go in the hm_utils?
        elif line[0] == 'k':
            k = int(line[1])
        elif line[0] == 'threads':
            threads = int(line[1])
        elif line[0] == 'sampler_type':
            sampler_type = line[1]
    if path_data:
        datafiles = os.path.join(path_data, datafiles)
    datafiles = sorted(glob(datafiles))
    if path_covariance:
        covfile = os.path.join(path_covariance, covfile)
    covfile = glob(covfile)
    if len(covfile) > 1:
        msg = 'ambiguous covariance filename'
        raise ValueError(msg)
    covfile = covfile[0]

    out = (datafiles, datacols, covfile, covcols, exclude_bins, output,
           sampler, nwalkers, nsteps, nburn, thin, k, threads,
           sampler_type)
    return out

def read_function(function):
    print 'Reading function', function
    function_path = function.split('.')
    if len(function_path) < 2:
        msg = 'ERROR: the parent module(s) must be given with'
        msg += 'a function'
        print msg
        exit()
    else:
        module = __import__(function)
        for attr in function_path:
            func = getattr(func, attr)
    return func

def setup_integrand(R, k=7):
    """
    These are needed for integration and interpolation and should always
    be used. k=7 gives a precision better than 1% at all radii

    """
    if R.shape[0] == 1:
        Rrange = numpy.logspace(numpy.log10(0.99*R.min()),
                                numpy.log10(1.01*R.max()), 2**k)
        # this assumes that a value at R=0 will never be provided, which is
        # obviously true in real observations
        R = numpy.append(0, R)
        Rrange = numpy.append(0, Rrange)
    else:
        Rrange = [numpy.logspace(numpy.log10(0.99*Ri.min()),
                                 numpy.log10(1.01*Ri.max()), 2**k)
                  for Ri in R]
        R = [numpy.append(0, Ri) for Ri in R]
        Rrange = [numpy.append(0, Ri) for Ri in Rrange]
        R = numpy.array(R)
        Rrange = numpy.array(Rrange)
    return R, Rrange
