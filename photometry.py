import ezgal
import numpy
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

def extinction(ra, dec, path='../'):
    galcoords = coordinates.eq2gal(ra, dec)
    if galcoords[1] >= 0:
        fitsfile = path + 'schlegel/SFD_dust_4096_ngp.fits'
    else:
        fitsfile = path + 'schlegel/SFD_dust_4096_sgp.fits'
    wcs = astWCS.WCS(fitsfile)
    pix = wcs.wcs2pix(galcoords[0], galcoords[1])
    dustmap = fits.getdata(fitsfile)
    ebv = dustmap[int(pix[1]),int(pix[0])]
    # copied these from Henk's code -- not sure about the source
    Ag = 3.793 * ebv
    Ar = 2.751 * ebv
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

