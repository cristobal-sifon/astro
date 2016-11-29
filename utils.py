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


def Av(ra, dec, verbose=False, bands='UBVRIugrizJHKL'):
    """
    retrieve SFD dust extinction and galactic coordinates from NED
    """
    msg = 'Retrieveng dust extinction at RA=%.6f, Dec=%.6f from NED' \
            %(ra, dec)
    if verbose:
        print msg
    form = 'in_csys=Equatorial'
    form += '&in_equinox=J2000.0'
    form += '&obs_epoch=2000.0'
    form += '&lon=%.7fd' %ra
    form += '&lat=%.7fd' %dec
    form += '&pa=0.0'
    form += '&out_csys=Equatorial'
    form += '&out_equinox=J2000.0'
    cmd = 'http://nedwww.ipac.caltech.edu/cgi-bin/nph-calc?' + form
    response = urllib.urlopen(cmd)
    text = response.read()
    # find extinction in each band
    ext = scipy.zeros(len(bands))
    for line in text.split('\n'):
        l = re.split('\s+', line)
        if l[0] in ('Landolt', 'SDSS', 'UKIRT'):
            j = string.find(bands, l[1])
            if j != -1:
                ext[j] = float(l[3])
    return ext


def absmag(mag, z, band1='megacam_r', band2='megacam_r',
           model='cb07_burst_0.1_z_0.02_salp.model', zf=5):
  """
  Takes a set of magnitudes all at the same redshift z
  and returns the aboslute magnitudes. This does not need to be
  done for each object since the conversion from apparent
  to absolute is an additive term that depends only on redshift
  """
  mo = 20. # irrelevant, but to be consistent
  sed = ezgal.model(model)
  sed.set_normalization(band1, z, mo, apparent=True)
  mabs = sed.get_absolute_mags(zf, band2, z)
  return mag + (mabs - mo)


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

