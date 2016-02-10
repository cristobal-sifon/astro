# -*- coding: utf-8 -*-

def danese80(s, z, dz, zo=0):
    """
    Calculates a correction to the velocity dispersion for errors in the
    individual measurements of the redshifts of galaxies (Danese, De Zotti & Di
    Tulio, 1980).

    Parameters
    ----------
        s         : velocity dispersion, in km/s.
        z         : float or numpy array with the redshifts of all (member) 
                    galaxies.
        dz        : float or numpy array with the measurement errors
                    corresponding 
                    to z.

    Returns
    -------
        v         : the updated velocity dispersion, in km·s⁻¹.

    """
    import numpy
    import stattools
    from astro import constants, units

    if not zo:
        try:
            zo = stattools.Cbi(z)
        except ZeroDivisionError:
            zo = numpy.median(z)
    if type(dz) == float:
        delta = constants.c * dz
    if type(dz) == list or type(dz) == numpy.ndarray:
        try:
            delta = constants.c * stattools.Cbi(dz)
        except ZeroDivisionError:
            delta = constants.c * numpy.std(dz)
    delta = delta / units.km
    return numpy.sqrt(s ** 2 - delta**2 / (1. + zo)**2)

def fsp():
  """
  Surface pressure correction term

  """
  return

def sigma_profile(sigma, aperture, r200, z=0, dsigma=0, orbits='iso',
                  concentration='duffy08', err=1e-3):
    """
    A correction for incomplete coverage based on the theoretical velocity
    dispersion profile of Mamon, Biviano & Murante (2010). The profiles
    were kindly provided by Gary Mamon

    Returns
    -------
      s200      : float
                  the corrected velocity dispersion at r200
      aperture  : float
                  the final estimate of the radial coverage, based on the
                  new r200
      r200      : float
                  the final estimate of r200

    """
    import conversions
    import scalings
    import scipy
    from scipy import integrate, interpolate

    def fraction(r0, c=4, orbits='iso'):
        ## these values kindly provided by Gary Mamon, are in a file in the
        ## folder where this module is located
        ## here, s = sigma_LOS / sqrt[GM(r_s)/r_s]. However, we are only
        ## interested in a ratio.
        # r / r_s
        x = [0.000, 0.100, 0.126, 0.158, 0.200, 0.251, 0.316, 0.398,
             0.501, 0.631, 0.794, 1.000, 1.259, 1.585, 1.995,
             2.512, 3.162, 3.981, 5.012, 6.310, 7.943, 10.000]
        # s for isotropic orbits
        if orbits == 'iso':
            s = [0.000, 0.625, 0.638, 0.650, 0.661, 0.670, 0.677, 0.682,
                 0.684, 0.682, 0.678, 0.669, 0.658, 0.642, 0.624,
                 0.603, 0.579, 0.554, 0.527, 0.499, 0.471, 0.442]
        # s for progressively more radial orbits, beta=r/2/(r+r_s)
        elif orbits == 'radial':
            s = [0.000, 0.700, 0.714, 0.727, 0.737, 0.745, 0.750, 0.751,
                 0.748, 0.740, 0.728, 0.712, 0.691, 0.667, 0.640,
                 0.611, 0.581, 0.549, 0.518, 0.486, 0.455, 0.425]
        # in units of r200
        r = scipy.array(x) / c
        profile = interpolate.interp1d(r, s)
        integrand = lambda R: profile(R) * R
        # these are integrated velocity dispersions
        #sigma_ap = 2 * integrate.romberg(integrand, 0.1, r0) / r0**2
        #sigma_r200 = 2 * integrate.romberg(integrand, 0.1, 1) # / 1**2
        sigma_ap = profile(r0)
        sigma_r200 = profile(1)
        return sigma_r200 / sigma_ap

    if type(concentration) == float:
      c = concentration
    elif concentration in ('dolag04', 'duffy08'):
      c = scalings.csigma(sigma, z, dsigma, scaling=concentration)[0]
    else:
      msg = 'Value for argument concentration not valid, see help page'
      raise ValueError(msg)
    s1 = sigma # velocity dispersion at r200 -- changing with changing r200
    r0 = aperture * r200 # initial radial coverage -- fixed!
    r1 = [r200] # r200 -- should go changing
    r2 = aperture * r200 # radial coverage
    x = []
    while abs(r1[-1]-r2) / r1[-1] > err:
      r2 = r1[-1]
      #x = fraction(r0/r1, c, orbits=orbits)
      x.append(fraction(r0/r1[-1], c, orbits=orbits))
      s1 = x[-1] * sigma
      m1 = scalings.sigma(s1, z)
      #r1 = conversions.rsph(m1, z)
      r1.append(conversions.rsph(m1, z))
    ap = r0 / r1[-1]
    if scipy.isnan(ap):
        print x
        print r1
        print c, sigma, s1, r0, r2, ap
        print ''
        #exit()
    r200new = r1[-1]
    return s1, ap, r200new
