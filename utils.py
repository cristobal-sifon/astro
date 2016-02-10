"""
Generic astronomical utilities

"""
#import anydbm
import os
import re
import scipy
import string
import urllib

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
