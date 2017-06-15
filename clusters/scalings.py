#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate the mass of a galaxy cluster given an observable, for several scaling
relations found by different studies.

Needs the uncertainties package.

When the observable is not given at the radius where the scaling relation was
calculated...

"""



import numpy
import uncertainties
from uncertainties import ufloat, umath, unumpy, Variable

# local
from astro.clusters import arnaud_profile
from astro import cosmology


def cM(M, z=0, ref='200c', profile='NFW', scaling='duffy08', redshift=0,
       errors=False):
    """
    get the concentration from a given mass, redshift and density profile.
    Requires the uncertainties package

    For the Duffy et al. relation, only the "Full" sample relation is
    implemented.

    Returns the concentration for a given mass, redshift and overdensity.

    Parameters
    ----------
        M    : cluster mass
                Both *m* and *z* can either be floats (if errors==False),
                or lists of two floats or uncertainties.ufloat
                objects (if errors==True).
        z    : cluster redshift
        ref  : reference radius. Options:
                -Duffy et al. (2008): ref = {'200a','200c'}
                -Dutton et al. (2014): ref = '200c'
        profile : {'NFW', 'Einasto'} Default 'NFW'
        scaling : {'duffy08', 'dutton14'}
        redshift : redshift of the c-M relation. Options:
                -Duffy et al. (2008): redshift = {0,2}. The second option
                 refers to the redshift-dependent scaling relation computed
                 for halos at z = 0-2.
                -Dutton et al. (2014): redshift = {0,0.5,1,2,3,4,5}. No
                 interpolation is available at the moment (therefore the
                 argument *z* is ignored for scaling='dutton14')

    Returns
    -------
        c    : concentration. Has the same type and format of both M and z

    """
    if errors:
        if type(M) != Variable:
            try:
                if len(M) == 1:
                    M = (M[0], 0)
            except TypeError:
                M = (M, 0)
        if type(z) != Variable:
            try:
                if len(z) == 1:
                    z = (z[0], 0)
            except TypeError:
                z = (z, 0)

    zscaling = {'duffy08': [0, 2], 'dutton14': [0, 0.5, 1, 2, 3, 4, 5]}
    Mo = {'duffy08': 2e12, 'dutton14': 1e12}
    if profile.lower() == 'nfw':
        A = {'duffy08': {'200c': ((5.74, 0.12), (5.71, 0.12)),
                         '200a': ((10.39, 0.22), (10.14, 0.22))},
             'dutton14': {'200c': ((0.905, 0.001), (0.814, 0.001),
                                   (0.728, 0.001), (0.612, 0.001),
                                   (0.557, 0.003), (0.528, 0.004),
                                   (0.539, 0.006))}}
        B = {'duffy08': {'200c': ((-0.097, 0.006), (-0.084, 0.006)),
                         '200a': ((-0.089, 0.007), (-0.081, 0.006))},
             'dutton14': {'200c': ((-0.101, 0.001), (-0.086, 0.001),
                                   (-0.073, 0.001), (-0.050, 0.001),
                                   (-0.021, 0.002), (0.000, 0.003),
                                   (0.027, 0.005))}}
        C = {'duffy08': {'200c': ((0, 0), (-0.47, 0.04)),
                         '200a': ((0, 0), (-1.01, 0.04))}}
    elif profile.lower() == 'einasto':
        A = {'duffy08': {'200c': ((6.48, 0.15), (6.40, 0.16)),
                         '200a': ((11.84, 0.30), (11.39, 0.30))},
             'dutton14': {'200c': ((0.978, 0.006), (0.884, 0.005),
                                   (0.775, 0.004), (0.613, 0.004),
                                   (0.533, 0.005), (0.481, 0.009),
                                   (0.478, 0.022))}}
        B = {'duffy08': {'200c': ((-0.127, 0.009), (-0.108, 0.007)),
                         '200a': ((-0.124, 0.008), (-0.107, 0.007))},
             'dutton14': {'200c': ((-0.125, 0.004), (-0.117, 0.004),
                                   (-0.100, 0.004), (-0.073, 0.006),
                                   (-0.027, 0.009), (-0.020, 0.014),
                                   (0.013, 0.032))}}
        C = {'duffy08': {'200c': ((0, 0), (-0.62, 0.04)),
                         '200a': ((0, 0), (-1.16, 0.05))}}
    try:
        i = zscaling[scaling].index(redshift)
    except ValueError:
        msg = 'ERROR: redshift %.2f not implemented for scaling %s;' \
                %(redshift, scaling)
        msg += 'see help page.'
        print(msg)
        exit()
    try:
        a = ufloat(*A[scaling][ref][i])
        b = ufloat(*B[scaling][ref][i])
        #m = M / Mo[scaling] / cosmology.h
        m = M / (Mo[scaling]/cosmology.h)
    except KeyError:
        msg = 'ERROR: combination of scaling=%s, ref=%s is not' \
                %(scaling, ref)
        msg += ' implemented; see help page.'
        print(msg)
        exit()
    if scaling == 'dutton14':
        c = 10**a * m**b
    elif scaling == 'duffy08':
        c = a * m**b * (1+z) ** ufloat(*C[scaling][ref][i])
    if errors:
        return c
    elif type(c) == Variable:
        return c.nominal_value
    else:
        return unumpy.nominal_values(c)


def csigma(sigma, z=0, dsigma=0, scaling='duffy08'):
    """
    Estimate the concentration from a given velocity dispersion measured at
    r200c.

    Available scalings:
        Dolag et al. (2004)
        Duffy et al. (2008)

    """

    if float(uncertainties.__version__.split('.')[0]) < 2:
        s = ufloat((sigma, dsigma))
    else:
        s = ufloat(sigma, dsigma)
    if scaling == 'duffy08':
        if float(uncertainties.__version__.split('.')[0]) < 2:
            A = ufloat((5.71, 0.12))
            B = ufloat((-0.084, 0.006))
            C = ufloat((-0.47, 0.04))
        else:
            A = ufloat(5.71, 0.12)
            B = ufloat(-0.084, 0.006)
            C = ufloat(-0.47, 0.04)
        c = A * (s * cosmology.E(z)**(1./3) / 125)**(3*B) * (1+z)**C
    elif scaling == 'dolag04':
        c = 4 / (s/700.)**0.306
    return c.nominal_value, c.std_dev


def Mgas(mgas, z, dmgas=0, dz=0, radius='500c', scaling='okabe10'):
    """
    Estiamte the mass from the gas mass, generally estimated through X-ray
    observations. Input units are Msun.

    Available scalings:
        Okabe et al. (2010, ApJ, 721, 875) -- Assuming self-similar slope
        Mahdavi et al. (2012, arXiv)

    """

    if dmgas > 0:
        mgas = ufloat((mgas, dmgas))
    if dz > 0:
        z = ufloat((z, dz))

    if scaling == 'okabe10':
        if radius == '500c':
            A = umath.log10(ufloat((13.10, 0.77)))
            #B = ufloat((1.00, 0.15))
            B = 1
    elif scaling == 'mahdavi12':
        if radius == '500c':
            A = ufloat((0.90, 0.02))
            B = ufloat((1.04, 0.10))

    m = 10 ** A * mgas ** B
    return m.nominal_value, m.std_dev()


def sigma(s, z, ds=None, dz=None,
          radius='200c', scaling='evrard08', bias=(1,0.),
          zscaling=0, separate_errors=False):
    """
    Estimate the mass from the galaxy velocity dispersion, sigma, in km/s.

    The following scalings are implemented:
        evrard08 : Evrard et al. (2008), DM particles
        lau10 : Lau, Nagai, & Kravtsov (2010), DM particles in hydro-
                simulation with cooling and star formation (no AGN feedback)
        munari13dm : Munari et al. (2013), DM particles in AGN simulation
        munari13sub : Munari et al. (2013), DM subhalos in AGN simulation
        munari13gal : Munari et al. (2013), galaxies in AGN simulation
        munari13 : alias for munari13gal
        saro13 : Saro et al. (2013), SAM galaxies

    Parameters
    ----------
      s : float
        Galaxy cluster velocity dispersion
      z : float
        Cluster redshift
      ds : float or list of 2 floats (optional)
        Uncertainty on the velocity dispersion. If 2 floats are given then
        these are assymetric errors, and assymetric errors are returned.
      dz : float (optional)
        Uncertainty on the redshift
      radius : '200c'
        Reference radius at which to calculate the mass. Only 'r200c' is
        accepted.
      scaling : str (default 'evrard08')
        Which scaling relation to use. Accepted values are:
          evrard08 : Evrard et al. (2008), DM particles
          lau10 : Lau, Nagai, & Kravtsov (2010), DM particles in hydro-
                  simulation with cooling and star formation (no AGN feedback)
          munari13dm : Munari et al. (2013), DM particles in AGN simulation
          munari13sub : Munari et al. (2013), DM subhalos in AGN simulation
          munari13gal : Munari et al. (2013), galaxies in AGN simulation
          munari13 : alias for munari13gal
      bias : float or list of 2 floats (default 1)
        Galaxy bias, sigma_gal/sigma_DM. If a list is given, the first
        value is the bias and the second its uncertainty. A bias b==1 means
        that galaxies are unbiased tracers of the dark matter velocity
        distribution.
      zscaling : float (default 0)
        Redshift at which the scaling relation is taken. Only the lau10
        scaling has values at different redshifts, and these are: 0, 0.6, 1.0.
        If any other value is given raises a ValueError. Note that no
        message is given if a zscaling != 0 is given for the other scalings.
      separate_errors : boolean (default False)
        Whether to return statistical and systematic (from the uncertainties
        in the scaling relation parameters) uncertainties separately.

    Returns
    -------
      m : float
        Galaxy cluster mass

    Optional returns
    ----------------
      1) separate_errors == False:
          dm : float or list of 2 floats
            Total uncertainty on the mass, accounting for statistical and
            systematic errors from the scaling relation (can be 2 values if
            input ds includes 2 floats)
      2) separate_errors == True:
          dm_stat : float or list of 2 floats
            Statistical uncertainty on the mass (can be 2 values if
            input ds includes 2 floats)
          dm_syst : float
            Systematic uncertainty on the mass from the scaling relation
            parameters

    """
    h = cosmology.h
    Ez = cosmology.E(z)
    # best-fit parameters from the scaling
    if scaling == 'saro13':
        def Mdyn(s, z, A, B, C):
            return 1e15 * (s / (A * Ez**C)) ** B
        A = ufloat(939., 0.55)
        B = ufloat(2.91, 0.0021)
        C = ufloat(0.33, 0.0019)
        Msyst = Mdyn(s, z, A, B, C)
        if ds is not None:
            if type(s) == numpy.ndarray:
                s = unumpy.uarray(s, ds)
            else:
                s = ufloat(s, ds)
        if dz is not None:
            if type(z) == numpy.ndarray:
                z = unumpy.uarray(z, dz)
            else:
                z = ufloat(z, dz)
        Mstat = Mdyn(s, z, A.nominal_value,
                     B.nominal_value, C.nominal_value)
        if ds:
            if separate_errors:
                if type(s) == numpy.ndarray:
                    err = (numpy.array([M.std_dev for M in Mstat]),
                           numpy.array([M.std_dev for M in Msyst]))
                else:
                    err = (Mstat.std_dev, Msyst.std_dev)
            else:
                if type(s) == numpy.ndarray:
                    err = numpy.array([numpy.hypot(Mst.std_dev, Msy.std_dev) \
                                       for Mst, Msy in zip(Mstat, Msyst)])
                else:
                    err = numpy.hypot(Mstat.std_dev, Msyst.std_dev)
            if type(s) == numpy.ndarray:
                M = numpy.array([M.nominal_value for M in Msyst])
            else:
                M = Msys.nominal_value
            return M, err
        if type(s) == numpy.ndarray:
            return numpy.array([M.nominal_value for M in Msyst])
        return Msyst.nominal_value
    # slope
    A = {'200c': {'evrard08': (0.3361, 0.0026),
                  'lau10': ((0.2724, 0.0149), # z = 0.0
                            (0.2903, 0.0254), # z = 0.6
                            (0.3232, 0.0153)), # z = 1.0
                  'munari13dm': (0.336, 0.0015),
                  'munari13sub': (0.365, 0.0017),
                  'munari13gal': (0.364, 0.0021)},
         '500c': {'lau10': ((0.2965, 0.0144), # z = 0.0
                            (0.2888, 0.0180), # z = 0.6
                            (0.3134, 0.0108))}} # z = 1.0
    A['200c']['munari13'] = A['200c']['munari13gal']
    # normalization
    So = {'200c': {'evrard08': (1082.9, 4.0),
                   'lau10': ((692., 11.), (674., 11.), (671., 8.)),
                   'munari13dm': (1095., 4.4),
                   'munari13sub': (1199., 5.2),
                   'munari13gal': (1177., 4.2)},
          '500c': {'lau10': ((788., 10.), (762., 9.), (771., 11.))}}
    So['200c']['munari13'] = So['200c']['munari13gal']
    # pivot mass
    Mo = {'evrard08': 1e15, 'lau10': 2e14, 'munari13': 1e15}
    Mo['munari13dm'] = Mo['munari13sub'] = Mo['munari13gal'] = Mo['munari13']
    # now read
    a = A[radius][scaling]
    so = So[radius][scaling]
    mo = Mo[scaling]
    if scaling == 'lau10':
        zval = [0.0, 0.6, 1.0]
        i = zval.index(zscaling)
        a, da = a[i]
        so, dso = so[i]
    else:
        a, da = a
        so, dso = so
    try:
        b, db = bias
    except ValueError:
        b = bias
        db = 0.
    m = mo / (h*Ez) * (b*s/so)**(1/a)
    def err(s, z, dsi, dzi):
        ## statistical:
        # dM/dz
        A = 3 * h * mo / (2*Ez**3) * (s/so)**(1/a) * (1+z)**2
        # dM/ds
        B = mo / (h*Ez) * (s/so) ** (1/a) / (a*s)
        stat = numpy.hypot(A*dzi, B*dsi)
        ## systematic:
        # dM/dso
        C = -mo / (h*Ez) * (s/so)**(1/a) / (a*so)
        # dM/da
        D = -mo / (h*Ez) * (s/so)**(1/a) * numpy.log(s/so) / a**2
        # dM/db
        E = m / (b*a)
        syst = numpy.hypot(C*dso, numpy.hypot(D*da, E*db))
        return numpy.array([stat, syst])

    if ds is not None:
        if dz is None:
            dz = 0
        try:
            stat, syst = err(s, z, ds, dz)
            if separate_errors:
                return m, (stat, syst)
        except TypeError:
            st1, syst = err(s, z, ds[0], dz)
            st2, syst = err(s, z, ds[1], dz)
            if separate_errors:
                return m, (st1, st2, syst)
            else:
                return m, (numpy.hypot(st1, syst), scipy.hypot(st2, syst))
        return m, numpy.hypot(stat, syst)
    else:
      return m


def Tx(t, z, dt=0, dz=0, radius='500c', scaling='arnaud10'):
  return


def Ysz(y, z, dy=0, dz=0, r_in='500c', r_out='500c', scaling='sifon13'):
    """
    Estimate the mass from the integrated Sunyaev-Zel'dovich effect
    signal Ysz in Mpc^2, which is converted to the measure at the
    radius for each scaling relation (e.g., r200c for Sifon et al.)
    using the Universal Pressure Profile of Arnaud et al. (2010).

    Available scalings:
        Andersson et al. (2011, ApJ, 738, 48) -- X-rays
            originally at r500c
        Marrone et al. (2012, ApJ, 754, 119)  -- Weak lensing
            originally at r500c, r1000c, r2500c
        Sifon et al. (2013, ApJ, 772, 25)     -- Dynamics
            originally at r200c
    """
    # just to help if I forget not to put the "r"
    if r_in[0] == 'r':
        r_in = r_in[1:]
    if r_out[0] == 'r':
        r_out = r_out[1:]

    cosmology.Omega_M = 0.3
    cosmology.Omega_L = 0.7
    Ez = cosmology.E(z)

    y = ufloat((y, dy))
    z = ufloat((z, dz))

    if scaling == 'andersson11':
        # see Marrone et al. (2012)
        if r_in == '500c':
            A = ufloat((0.36, 0.03))
            B = ufloat((0.60, 0.12))
        m = 10 ** (14 + A) * (y / Ez ** (2./3.) / 1e-5) ** B

    elif scaling == 'marrone12':
        if r_in == '500c':
            A = ufloat((0.367, 0.099))
            B = ufloat((0.44, 0.12))
        elif r_in == '1000c':
            A = ufloat((0.254, 0.080))
            B = ufloat((0.48, 0.11))
        elif r_in == '2500c':
            A = ufloat((0.254, 0.080))
            B = ufloat((0.48, 0.11))
        else:
            A = ufloat((0.367, 0.099))
            B = ufloat((0.44, 0.12))

        m = 10 ** (14 + A) * (y / Ez ** (2./3.) / 1e-5) ** B

    # the different r_in's and r_out's still need to be well checked
    elif scaling == 'sifon13':
        A = ufloat((14.99, 0.07))
        B = ufloat((0.48, 0.11))
        if r_in != '200c':
            f = arnaud_profile.Yratio(r_in, '200c', 'sph')
            #print(f)
            y = f * y
        m = 10 ** A * (y / Ez ** (2./3.) / 5e-5) ** B
        if r_out != '200c':
            # update clusters.NFW so that it returns the error as well
            m = read_NFW(m.nominal_value, z.nominal_value, m.std_dev(),
                                    ref_in='200c', ref_out=r_out)
            m = ufloat((m[0], m[1]))

    return m.nominal_value, m.std_dev()


def Yx(y, z, dy=0, dz=0, radius='500c', scaling='arnaud10'):
    """
    Estimate the mass from the X-ray pseudo-Compton parameter Yx (see
    Kravtsov et al. 2006). Units are Msun*keV

    Available scalings:
        Arnaud et al. (2010, A&A, 517, A92)
    """
    Ez = cosmology.E(z)

    if dy > 0:
        y = ufloat((y, dy))
    if dz > 0:
        z = ufloat((z, dz))

    if scaling == 'arnaud10':
        A = ufloat((14.567, 0.010))
        B = ufloat((0.561, 0.018))

        #if radius != '500c':

        m = 10 ** A * (y / 2e14) ** B / Ez ** (2./5.)
        return m.nominal_value, m.std_dev()


