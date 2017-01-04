"""
Generic astronomical utilities

"""
#import anydbm
import os
import re
import readfile
import scipy
import string
import urllib
from astLib.astWCS import WCS


def makereg(filename, rakey='ra', deckey='dec', xkey='x', ykey='y',
            shape='circle', size='3"', color='magenta',
            commentkeys='', wcs=None, frame='default', output=None):
    """
    Take a galaxy catalog and create a region file. See
    http://ds9.si.edu/doc/ref/region.html.

    Parameters
    ----------
        filename    : str
            input filename, containing at least two positional columns,
            (x,y) or (ra,dec)
        rakey       : str, int or None
            Right ascension column name (if header is provided in the
            first row) or column number. Will skip if name not present
            in header.
        deckey      : str, int or None
            Declination column name (if header is provided in the first
            row) or column number. Will skip if name not present in
            header.
        xkey, ykey  : str or int
            analogous to rakey, deckey for image coordinates. If rakey
            and deckey are valid entries then xkey and ykey are
            ignored.
        shape       : str
            any shape accepted by DS9 region files.
        size        : str
            comma-separated size(s) and rotation angle, as appropirate
            for the selected shape. For instance, for `shape=circle`,
            `size` could be '3"' (three arcsec) or '4' (four pixels);
            and for `shape=box`, `size` could be '3",4",45'
            (three-by-four sq. arcsec rotated by 45 degrees).
        color       : str
            any color accepted by DS9 region files.
        commentkeys : str, int, or lists of them (optional)
                      analogous to the above for any text to be added
                      to the region file on top of the objects.
        wcs         : str or astLib.astWCS.WCS object (optional)
            image with which to convert coordinates
        frame       : {'fk5', 'image'} (optional)
            frame in which the region file will be written (WCS
            or image coordinates). If not specified, will use the same
            input frame.
        output      : str (optional)
            output file name. If not provided, will simply replace
            the extension of the original file by .reg.

    """
    hdr = open(filename).readline().split()
    # are the keys given as column names?
    if (isinstance(rakey, basestring) \
            and (rakey in hdr and deckey in hdr)) \
            or (isinstance(xkey, basestring) \
                and (xkey in hdr and ykey in hdr)):
        if rakey in hdr and deckey in hdr:
            keys = ','.join((rakey, deckey))
        else:
            keys = ','.join((xkey, ykey))
        if hasattr(commentkeys, '__iter__'):
            commentkeys = ','.join(commentkeys)
        if isinstance(commentkeys, basestring) and len(commentkeys) > 0:
            keys = ','.join((keys, commentkeys))

    return

