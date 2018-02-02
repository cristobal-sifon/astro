from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


def name_from_coords(coords=None, ra=None, dec=None):
    """Create name from coordinates following IAU convention

    Name can be created by passing either an
    `astropy.coordinates.SkyCoord` object or an RA and Dec.

    Parameters
    ----------
    coords : `astropy.coordinates.SkyCoord` object
    ra, dec : `float` or `astropy.units.Quantity` objects.
        if `float`, they are assumed to be in degrees.
    """
    if coords is not None:
        name_ra = '{0:02.0f}{1:02.0f}{1:04.1f}'.format(*coords.ra.hms)
