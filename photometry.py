import ezgal
import numpy
import os
import re
import string
import urllib
from astLib import astWCS
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits
from scipy.interpolate import interp1d

# local
import coordinates


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


def extinction(ra, dec, use_ned=True, bands='UBVRIugrizJHKL',
               path_sfd='/disks/shear7/sifon/', verbose=False):
    """
    Get extinction given an RA and Dec using the Schlegel et al. (1998)
    maps (if use_ned==False) or the Schlafly et al. (2011) maps (if
    use_ned==True).

    Set use_ned to False to get the extinction for the Megacam bands in
    the format I have been using for CCCP. Otherwise NED will be
    queried and the extinction in the selected bands will be returned
    (and ugriz refer to SDSS bands).

    """
    if use_ned:
        if verbose:
            msg = 'Retrieveng dust extinction at RA=%.6f, Dec=%.6f from NED' \
                  %(ra, dec)
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
        ext = numpy.zeros(len(bands))
        for line in text.split('\n'):
            l = re.split('\s+', line)
            if l[0] in ('Landolt', 'SDSS', 'UKIRT'):
                j = string.find(bands, l[1])
                if j != -1:
                    ext[j] = float(l[3])
        return ext
    # note that the conversions at the bottom only work for Megacam
    else:
        galcoords = coordinates.eq2gal(ra, dec)
        if galcoords[1] >= 0:
            #fitsfile = path_sfd + 'schlegel/SFD_dust_4096_ngp.fits'
            hemisphere = 'ngp'
        else:
            hemisphere = 'sgp'
        fitsfile = os.path.join(path_sfd, 'schlegel',
                                'SFD_dust_4096_{0}.fits'.format(hemisphere))
        wcs = astWCS.WCS(fitsfile)
        pix = wcs.wcs2pix(galcoords[0], galcoords[1])
        dustmap = fits.getdata(fitsfile)
        ebv = dustmap[int(pix[1]),int(pix[0])]
        # copied these from Henk's code -- not sure about the source
        Ag = 3.793 * ebv
        Ar = 2.751 * ebv
        #Ar = ebv / 2.751
        return ebv, Ag, Ar


def mstar(z, band='megacam_r', zf=5.,
          apparent=True, model='cb07_burst_0.1_z_0.02_chab.model'):
    """
    Uses mstar(z) measured by Lu et al. (2009) for CHFTLS

    keep alpha=1.2 from Blanton et al. (2001), for consistency with Lu et al.
    """
    sed = ezgal.model(model)
    #t = [17.65, 18.10, 18.48, 18.81, 19.12, 19.40, 19.67]
    #mstar_z = interp1d(numpy.arange(0.19, 0.38, 0.03), t, kind='cubic')
    #if 0.19 <= z <= 0.37 and band == 'megacam_r':
        #return mstar_z(z)
    ## use them to "evolve" mstar using EzGal
    #if z < 0.19:
        #znorm = 0.19
    #else:
        #znorm = 0.37
    #mstar = mstar_z(znorm)
    #sed.set_normalization('megacam_r', znorm, mstar, apparent=apparent)
    #mstar = sed.get_apparent_mags(zf, band, z)
    #return mstar
    t = numpy.linspace(0.05, 0.35, 100)
    # these are valid for 0.05 < z < 0.7 (Rykoff et al. 2014)
    if z <= 0.5:
        mstar_z = lambda lnz: 22.44 + 3.36*lnz + 0.273*lnz**2 - \
                              0.0618*lnz**3 - 0.0227*lnz**4
    else:
        mstar_z = lambda lnz: 22.94 + 3.08*lnz - 11.22*lnz**2 - \
                              27.11*lnz**3 - 18.02*lnz**4
    if 0.05 <= z <= 0.7:
        znorm = z
    elif z < 0.05:
        znorm = 0.05
    else:
        znorm = 0.7
    mstar = mstar_z(numpy.log(znorm))
    if band == 'sloan_i':
        return mstar
    sed.set_normalization('sloan_i', znorm, mstar, apparent=apparent)
    mstar = sed.get_apparent_mags(zf, band, z)
    return mstar


def sdss2megacam(sdss, bands_sdss, band_megacam='r'):
    """
    Taken from
    http://www2.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/megapipe/docs/filt.html

    """
    if not isinstance(bands_sdss, basestring):
        bands_sdss = ''.join(bands_sdss)
    if band_megacam == 'u':
        if not bands_sdss == 'ug':
            return
        c = -0.241
    elif band_megacam == 'g':
        if not bands_sdss == 'gr':
            return
        c = -0.153
    elif band_megacam == 'r':
        if not bands_sdss == 'gr':
            return
        c = -0.024
    elif band_megacam == 'i':
        if not bands_sdss == 'ri':
            return
        c = -0.003
    elif band_megacam == 'z':
        if not bands_sdss == 'iz':
            return
        c = 0.074
    if band_megacam in 'ug':
        m = sdss[0]
    else:
        m = sdss[1]
    return m + c * (sdss[0]-sdss[1])