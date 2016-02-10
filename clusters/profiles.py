# -*- coding: utf-8 -*-

def nfw(m, z, dm=0, ref_in='200c', ref_out='500c', \
        c=1., err=1e-6, scaling='duffy08', full_output=False):
  """
  Convert the mass of a cluster from one overdensity radius to another
  assuming an NFW profile.

  Parameters
  ----------
    m         : float
                mass, in units of solar mass
    z         : float
                redshift
    dm        : float (optional)
                mass uncertainty
    ref_in    : {'2500c', '500c', '200c', '180c', '100c',
                 '500a', '200a', '100a'} (default '200c')
                overdensity at which the input mass is measured. The last
                letter indicates whether the overdensity is with respect to
                the critical ('c') or average ('a') density of the Universe
                at redshift z.
    ref_out   : {'2500c', '500c', '200c', '180c', '100c',
                 '500a', '200a', '100a'} (default '500c')
                overdensity at which the output mass is measured.
    c         : float (default 1)
                either a fixed concentration or a correction factor to the
                Duffy et al. relation (useful, e.g., for estimating
                uncertainties due to the c-M relation). See parameter duffy.
    err       : float (default 1e-6)
                allowed difference for convergence
    scaling   : {'duffy08', 'dutton14'} (optional)
                If given, use the corresponding concentration relation with a
                correction factor *c*. If False, the concentration is fixed
                to the value of *c*. Only possible if ref_in is either '200c'
                or '200a'.
    full_output : bool (default False)
                If True, also return the concentration used.

  Returns
  -------
    m_out     : float
                The mass at the output overdensity. If dm>0 then an
                uncertainty on this mass is also returned (m_out is
                then a tuple of length 2).
    c         : float (optional)
                concentration. This is returned if full_output is set to True.

  """
  import scipy
  import scipy.interpolate
  from astro import cosmology
  from astro.clusters import conversions, scalings

  #if ref_in != '200c':
    #return read_nfw(m, z, dm, ref_in=ref_in, ref_out=ref_out)
  if ref_in not in ('200a', '200c'):
      duffy = False

  # iteratively calculate output mass
  def _mass(c, m, scale, err):
    m_out = 0
    mass = m
    while abs(m_out/mass - 1) > err:
      m_out = mass
      r_out = (3 * m_out / (4 * scipy.pi * rho_out)) ** (1./3.)
      x = r_out / scale
      mass = m * (scipy.log(1 + x) - x / (1 + x)) / \
             (scipy.log(1 + c) - c / (1 + c))
    return mass

  # density contrasts
  rho_in = int(ref_in[:-1]) * cosmology.density(z, ref=ref_in[-1])
  rho_out = int(ref_out[:-1]) * cosmology.density(z, ref=ref_out[-1])

  # radii
  r_in = conversions.rsph(m, z, ref=ref_in, unit='Mpc')
  if scaling in ('duffy08', 'dutton14'):
    c = c * scalings.cM(m, z, ref=ref_in, scaling=scaling)
  scale = r_in / c
  # mass and uncertainty (if defined)
  m_out = _mass(c, m, scale, err)
  if dm > 0:
    dm_hi = _mass(c, m+dm, scale, err)
    dm_lo = _mass(c, m-dm, scale, err)
    m_out = (m_out, (dm_hi+dm_lo)/2)

  if full_output:
    return m_out, c
  return m_out

def read_nfw(m, z, dm=0, ref_in='500c', ref_out='200c'):
  """
  read the file nfw_conversion.dat located in this folder, and interpolate
  linearly between the closest values of m for ref_in. The mass range in the
  file is M200a=[1e14, 5e15] and includes the redshift range [0, 2] in steps
  of 0.01. The values in this file assume the Duffy et al. concentration
  relation

  """
  import readfile
  import scipy

  import scalings
  path = scalings.__file__.replace('scalings.pyc', '')

  col_in  = 'm' + ref_in
  col_out = 'm' + ref_out
  # format redshift
  z0 = str(round(z, 2))
  z0 += '0' * (4 - len(z0))

  data = readfile.dict(path + '/nfw_conversion.dat',
                       cols=scipy.arange(1, 9), include=z0,
                       lower=True)
  m_in = data[col_in]
  m_out = data[col_out]
  # identify closest value
  i = scipy.argmin(scipy.absolute(m - data[col_in]))
  # do linear interpolation
  if m == m_in[i]:
    mass = m_out[i]
  elif m > m_in[i]:
    order = scipy.array([i, i + 1])
    mass = scipy.interp(m, m_in[order], m_out[order])
  else:
    order = scipy.array([i - 1, i])
    mass = scipy.interp(m, m_in[order], m_out[order])

  # do the same for dm
  if dm > 0:
    # identify closest value
    i = scipy.argmin(abs(dm - data[col_in]))
    # do linear interpolation
    if dm == m_in[i]:
      return mass, m_out[i]
    elif dm > m_in[i]:
      order = scipy.array([i, i + 1])
    else:
      order = scipy.array([i - 1, i])
    return mass, scipy.interp(dm, m_in[order], m_out[order])

  return mass

#def sis(r, re, output=('m,s'), re_unit='kpc'):
    #from astro import cosmology

    #if re_unit == 'Mpc':
        #re *= 1e3
    #beta = cosmology.dA(
    #f = {'s': (re/28.9

def upp(r, m500, z, runit='Mpc', unit='astro',
        self_similar=True, profile='sph'):
  """
  The Universal Pressure Profile (UPP) from Arnaud et al. (2010)

  Parameters
  ----------
    r         : float
                Radius at which to estimate the pressure
    m500      : float
                Cluster mass within r500
    z         : float
                Cluster redshift
    runit     : {'Mpc', 'kpc', 'r500'} (default 'Mpc')
                Whether R is in absolute units ('Mpc', 'kpc') or in units
                relative to r500 ('r500'). In the first case, r500 will be
                estimated from m500 assuming a spherical cluster.
    unit      : {'astro', 'cgs', 'mks'} (default 'astro')
                Pressure units. Default value is astro, which returns the
                pressure in units of Msun·Mpc⁻¹·s⁻². In cgs, the units are
                erg·cm⁻³ = g·cm⁻¹·s⁻² while in mks they are kg·m⁻¹·s⁻².
    self_similar : bool (default True)
                Whether to use the "self-similar" scaling or the "universal"
                scaling.
    profile   : {'sph', 'cyl'} (default 'sph')
                Whether the used profile is spherical or cylindrical, i.e.,
                integrated along the line of sight.

  """
  import scipy
  import scipy.integrate
  from astro import constants, cosmology, units
  # local
  import conversions

  h = cosmology.h
  h70 = h / 0.7
  if self_similar:
    alpha_p = 0
    Po = 8.130 / h70**1.5
    c500 = 1.156
    gamma = 0.3292
    alpha = 1.0620
    beta = 5.4807
  else:
    alpha_p = 0.12
    Po = 8.403 / h70**1.5
    c500 = 1.177
    gamma = 0.3081
    alpha = 1.0510
    beta = 5.4905

  def alpha_prime(x):
    """ Eq. 8 """
    if self_similar:
      return 0
    f = (2*x) ** 3
    f /= (1+f)
    return 0.10 - f * (alpha_p+0.10)

  def I(x):
    """ Eq. 24 """
    f = lambda u: p(u, c500) * u**2
    return 3 * scipy.integrate.quad(f, 0, x)[0]

  def J(x):
    """ Eq. 27 """
    f = lambda u: p(u) * scipy.sqrt(u**2 - x**2) * u
    return I(5) - 3*scipy.integrate.quad(f, x, 5)[0]

  def P(x, m500, z):
    """ UPP itself, Eq. 13 """
    exp = alpha_p + alpha_prime(x, alpha_p, self_similar)
    return p(x) * P500(m500, z) * (m500/3e14) ** exp

  def P500(m500, z, unit='astro'):
    """ Eq. 5 """
    # in units of keV·cm⁻³
    pp = 1.65e-3 * cosmology.E(z)**(8./3.) * (m500 / 3e14)**(2./3.) * h70**2
    # in units of erg·cm⁻³ = g·cm⁻¹·s⁻²
    pp *= units.keV
    if unit == 'cgs':
      return pp
    # in units of kg·m⁻¹·s⁻²
    if unit == 'mks':
      return pp / units.kg * units.m
    # in units of Msun·Mpc⁻¹·s⁻²
    if unit == 'astro':
      return pp / units.Msun * units.Mpc
    return pp

  def p(x, c500):
    """ Eq. 11 """
    cx = c500 * x
    exp = (beta-gamma) / alpha
    return Po / (cx**gamma * (1+cx**alpha)**exp)

  def Ysph(R, r500, m500, z):
    f = lambda r: P(r/r500, m500, z) * r**2
    i = 4 * scipy.pi * scipy.integrate.quad(f, 0, R)[0]
    return i * constants.Msun * constants.sigmaT / \
           (constants.me * constants.c**2)

  def Ycyl(R, r500, m500, z):
    """ Eq. 15 """
    rb = 5 * r500
    f = lambda x, r: 4 * scipy.pi * r * P(x/r500, m500, z) * \
                     x / scipy.sqrt(x**2-r**2)
    i = scipy.integrate.dblquad(f, 0, R, lambda r: r, lambda r: Rb)[0]
    return i * constants.Msun * constants.sigmaT / \
           (constants.me * constants.c**2)

  # the calculation itself
  if runit == 'r500':
    x = r
  else:
    r500 = conversions.rsph(m500, z, ref='500c', unit=runit)
    x = r / r500
  if profile == 'sph':
    Ax = 2.925e-5 * I(x) / h70
    y = Ax * (m500/3e14)**1.78
  elif profile == 'cyl':
    Bx = 2.925e-5 * J(x) / h70
    y = Bx * (m500/3e14)**1.78
  return y

#def sigma(r, M, orbits='iso', concentration='duffy08'):
    #"""
    #Velocity dispersion profile measured in numerical simulations by
    #Mamon, Biviano, & Murante (2010). Kindly provided by Gary Mamon.

    #Parameters
    #----------
        #r : array-like of floats
            #radial

    #"""
    #x = [0.100, 0.126, 0.158, 0.200, 0.251, 0.316, 0.398,
            #0.501, 0.631, 0.794, 1.000, 1.259, 1.585, 1.995,
            #2.512, 3.162, 3.981, 5.012, 6.310, 7.943, 10.000]
    ## s for isotropic orbits
    #if orbits == 'iso':
        #s = [0.625, 0.638, 0.650, 0.661, 0.670, 0.677, 0.682,
                #0.684, 0.682, 0.678, 0.669, 0.658, 0.642, 0.624,
                #0.603, 0.579, 0.554, 0.527, 0.499, 0.471, 0.442]
    ## s for progressively more radial orbits, beta=r/2/(r+r_s)
    #elif orbits == 'radial':
        #s = [0.700, 0.714, 0.727, 0.737, 0.745, 0.750, 0.751,
                #0.748, 0.740, 0.728, 0.712, 0.691, 0.667, 0.640,
                #0.611, 0.581, 0.549, 0.518, 0.486, 0.455, 0.425]
    #if concentration == 'duffy08':
      #c = scalings.cM(M, z, dsigma, scaling=concentration,
                      #ref='200c', profile='NFW', scaling='duffy08')[0]
    #else:
      #c = concentration
    #x = scipy.array(x) / c