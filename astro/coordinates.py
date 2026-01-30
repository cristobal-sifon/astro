import numpy as np
from astLib import astCoords


"""
Various coordinates utilities

Variables
---------
  epoch      : Epoch. Right now only 'J2000' is implemented
  aNGP       : Right Ascension of the North Galactic Pole, for the
               chosen epoch
  dNGP       : Declination of the North Galactic Pole, for the chosen
               epoch

"""

epoch = 'J2000'

if epoch == 'J2000':
    aNGP = astCoords.hms2decimal('12:51:26.28', ':')
    dNGP = astCoords.dms2decimal('+27:07:42.0', ':')



def eq2gal(ra, dec):
    """
    WARNING: the output galactic longitude has a ~4 arcmin difference
    with the NED output. Will correct later

    Convert from Equatorial to Galactic coordinates. Currently only
    implemented for epoch=J2000

    Parameters
    ----------
        ra        : float
                    Right Ascension in decimal degrees
        dec       : float
                    Declination in decimal degrees

    Returns
    -------
        b         : float
                    Galactic latitude, in decimal degrees
        l         : float
                    Galactic longitude, in decimal degrees

    """
    a0 = np.radians(aNGP)
    d0 = np.radians(dNGP)
    a = np.radians(ra)
    d = np.radians(dec)
    # Galactic latitude
    b = np.cos(d) * np.cos(d0) * np.cos(a - a0) + np.sin(d) * np.sin(d0)
    b = np.arcsin(b)
    # Galactic longitude
    l0 = np.radians(33)
    l = (np.sin(d) - np.sin(b) * np.sin(d0)) / (np.cos(d) * np.sin(a - a0) * np.cos(d0))
    l = np.arctan(l) + l0
    return np.degrees(l) + 180, np.degrees(b)


def zhelio(z, ra, dec):
    c = 299792.458
    # from NED
    Vapex = 232.3 # km/s
    rad = np.pi / 180
    lapex = 87.8 * rad
    bapex = 1.7 * rad
    gal = [astCoords.convertCoords('J2000', 'GALACTIC', x, y, 2000.0)
           for x, y in zip(ra, dec)]
    l, b = np.transpose(gal)
    l *= rad
    b *= rad
    Vh = np.sin(b) * np.sin(bapex) - \
         np.cos(b) * np.cos(bapex)*np.cos(l-lapex)
    Vh *= Vapex
    return (c * z - Vh) / c

