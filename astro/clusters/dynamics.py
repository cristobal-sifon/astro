import pylab
import scipy
from astro import cosmology
from astLib import astCoords

c = 299792.458

def m200(z, x, y=[], zo=0, xycenter=None, xyunit='deg', zunit='redshift',
         membership='shifting gapper', correct_profile='mbm10',
         mass_scaling='evrard08', converge=True, bootstrap=1000):
    """
    Get the dynamical mass of a cluster given the positions and redshifts,
    or velocities, of galaxies. This module uses the Evrard et
    al. scaling relation to get the mass, including a correction for
    incomplete coverage.

    xyunit can be {'Mpc', 'kpc', 'deg', 'arcmin', 'arcsec'}, and have to
    be with respect to a center (i.e., cannot be coordinates on the sky).

    """
    if zo == 0:
        zo = scipy.median(z)
    if zunit == 'velocity':
        v = z
        zo /= c
    else:
        v = c * (z-zo)/(1+zo)
    # then x corresponds to cluster-centric distance:
    if len(y) == 0:
        if xyunit == 'kpc':
            r = x / 1e3
        elif xyunit in ('deg', 'arcmin', 'arcsec'):
            r = cosmology.dProj(zo, x, input_unit=xyunit, unit='Mpc')
    # otherwise use the given center to calculate distances
    elif xyunit in ('kpc', 'Mpc'):
        r = scipy.hypot(x, y)
        if xyunit == 'kpc':
            r /= 1e3
    else:
        if xyunit == 'arcsec':
            x /= 3600.
            y /= 3600.
        if xyunit == 'arcmin':
            x /= 60.
            y /= 60.
        dist = astCoords.calcAngSepDeg(x, y, xycenter[0], xycenter[1])
        r = cosmology.dProj(zo, dist, input_unit='deg', unit='Mpc')

    


    
    return
