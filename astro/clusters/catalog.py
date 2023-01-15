"""Utility to work with locally-stored cluster catalogs"""
from astro import cosmology
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.io.fits import getdata
from astropy.table import Table
from itertools import count
import numpy as np
import os
import urllib
import six
import sys

# these should not be modified
_available = (
    'abell', 'act-dr4', 'act-dr5', 'codex', 'gmbcg', 'hecs2013', 'madcows',
    'maxbcg', 'mcxc', 'orca', 'psz1', 'psz2', 'redmapper', 'spt-sz', 'whl'
    )
_filenames = {
    'abell': 'abell/aco1989_ned.tbl',
    'act-dr4': 'actpol/E-D56Clusters.fits',
    'act-dr5': 'advact/DR5_cluster-catalog_v1.1.fits',
    'codex': 'codex/J_A+A_638_A114_catalog.dat.gz.fits.gz',
    'gmbcg': 'gmbcg/GMBCG_SDSS_DR7_PUB.fit',
    'hecs2013': 'hecs/2013/data.fits',
    'madcows': 'madcows/wise_panstarrs.txt',
    'maxbcg': 'maxbcg/maxBCG.fits',
    'mcxc': 'mcxc/mcxc.fits',
    'orca': 'orca/fullstripe82.fits',
    'psz1': 'planck/PSZ-2013/PLCK-DR1-SZ/COM_PCCS_SZ-union_R1.11.fits',
    'psz2': 'planck/PSZ-2015/HFI_PCCS_SZ-union_R2.08.fits',
    'redmapper': 'redmapper/redmapper_dr8_public_v6.3_catalog.fits',
    'spt-sz': 'spt/bleem2015.txt',
    'whl': 'whl/whl2015.fits'
    }

# the user may choose to modify these
columns = {
    'abell': 'Object Name,RA(deg),DEC(deg),Redshift',
    'act-dr4': 'name,RADeg,decDeg,z',
    'act-dr5': 'name,RADeg,decDeg,redshift',
    'codex': 'CODEX,RAdeg,DEdeg,z',
    'gmbcg': 'OBJID,RA,DEC,PHOTOZ',
    'hecs2013': 'Name,RAJ2000,DEJ2000,z',
    'madcows': 'Cluster,Rahms,Dechms,Photz',
    'maxbcg': 'none,RAJ2000,DEJ2000,zph',
    'mcxc': 'MCXC,RAdeg,DEdeg,z',
    'orca': 'ID,ra_bcg,dec_bcg,redshift',
    'psz1': 'NAME,RA,DEC,REDSHIFT',
    'psz2': 'NAME,RA,DEC,REDSHIFT',
    'redmapper': 'NAME,RA,DEC,Z_LAMBDA',
    'spt-sz': 'SPT,RAdeg,DEdeg,z',
    'whl': 'WHL,RAJ2000,DEJ2000,zph'
    }
labels = {
    'abell': 'Abell',
    'act-dr5': 'ACT-DR5',
    'act-dr4': 'ACT-DR4',
    'codex': 'CODEX',
    'gmbcg': 'GMBCG',
    'hecs2013': 'HeCS',
    'hecs2016': 'HeCS-SZ',
    'madcows': 'MaDCoWS',
    'maxbcg': 'maxBCG',
    'mcxc': 'MCXC',
    'orca': 'ORCA',
    'psz1': 'PSZ1',
    'psz2': 'PSZ2',
    'redmapper': 'redMaPPer',
    'spt-sz': 'SPT-SZ',
    'whl': 'WHL'
    }
masscols = {
    'abell': None,
    'act-dr5': 'M500cCal',
    'act-dr4': 'M500cCal',
    'codex': 'lambda',
    'gmbcg': None,
    'hecs2013': None,
    'hecs2016': None,
    'madcows': None,
    'maxbcg': None,
    'mcxc': 'M500',
    'orca': None,
    'psz1': 'MSZ',
    'psz2': 'MSZ',
    'redmapper': 'LAMBDA_CHISQ',
    'spt-sz': 'M500c',
    'whl': None
    }
references  = {
    # optical
    'abell': 'Abell 1958',
    'gmbcg': 'Hao et al. 2010',
    'hecs2013': 'Rines et al. 2013',
    'hecs2016': 'Rines et al. 2016',
    'maxbcg': 'Koester et al. 2007',
    'orca': 'Geach, Murphy & Bower 2011',
    'redmapper-sdss': 'Rykoff et al. 2014',
    'whl': 'Wen, Han & Liu 2012; Wen & Han 2015',
    # sz
    'act-dr4': 'Hilton et al. 2018',
    'act-dr5': 'Hilton et al. 2021',
    'psz1': 'Planck Collaboration XXIX 2014',
    'psz2': 'Planck Collaboration XXVII 2016',
    'spt-sz': 'Bleem et al. 2015',
    # x-ray
    'codex': 'Finoguenov et al. 2020',
    'mcxc': 'Piffaretti et al. 2011'
    }
# all catalogs are here -- this should not be hard-coded
if 'DATA' in os.environ:
    path = os.environ['DATA']
elif 'DOCS' in os.environ:
    path = os.environ['DOCS']
else:
    path = os.path.join(os.environ['HOME'], 'Documents')
path = os.path.join(path, 'catalogs')
# these serve to restore the above attributes if necessary
_columns = columns.copy()
_labels = labels.copy()
_path = '{0}'.format(path)


class Catalog:
    """Catalog object
    
    The following attributes are defined at initialization:

    """

    def __init__(self, name, catalog=None, indices=None, cols=None,
                 label=None, base_cols='default', masscol=None,
                 coord_unit='deg'):
        """
        Define a ``Catalog`` object

        Parameters
        ----------
        name : str
            name of the catalog
        catalog : ``astropy.table.Table`` (optional)
            catalog table
        indices : array of int (optional)
            indices of the objects whose information is requested.
            These indices may be obtained by running `query()`
            using the positions of the objects first. If not given,
            the full catalog will be returned.
        cols : str or list of str (optional)
            list or comma-separated names of the columns wanted. If
            not given, all columns are returned.
        base_cols : list of str (optional)
            column names for name, ra, dec, and redshift, in that order
        masscol : str
            name of column containing mass or mass-like quantity of interest
            (e.g., luminosity). 

        """
        if not isinstance(name, six.string_types):
            msg = 'argument name must be a string'
            raise TypeError(msg)
        if catalog is None and name not in _available:
            err = f'catalog {name} not available.' \
                f' Available catalogs are {_available}'
            raise ValueError(err)
        self.name = name
        self.coord_unit = coord_unit
        self._indices = indices
        self._cols = cols
        if name in _available:
            self.label = labels[self.name] if label is None else label
            self.reference = references[self.name]
            fname = self.filename()
            # load. Some may have special formats
            if catalog is None:
                if self.name in ('madcows','spt-sz'):
                    catalog = ascii.read(fname, format='cds')
                elif self.name == 'abell':
                    catalog = ascii.read(fname, format='ipac')
                    # fill masked elements
                    catalog = catalog.filled(-999)
                    #noz = (data[columns[data][3]].values == 1e20)
                    #data[noz] = -1
                else:
                    catalog = Table(getdata(fname, ext=1, ignore_missing_end=True))
            else:
                catalog = Table(catalog)
            base_cols = columns[self.name] if base_cols in (None, 'default') \
                else base_cols
            self.masscol = masscols[name] if masscol is None else masscol
        else:
            self.label = self.name if label is None else label
            self.reference = None
            if base_cols == 'default':
                base_cols = ('name', 'ra', 'dec', 'z')
            if not isinstance(catalog, Table):
                catalog = Table(catalog)
            self.masscol = masscol
        if self.masscol is not None and self.masscol not in catalog.colnames:
            raise KeyError(f'masscol {self.masscol} not in catalog')
        
        # if necessary, adding an index column should happen before we define
        # which columns to return
        self.base_cols = base_cols.split(',') if isinstance(base_cols, str) \
            else base_cols
        try:
            _nobj = catalog[self.base_cols[-1]].size
        except KeyError:
            err = f'key {self.base_cols[-1]} not found in catalog {name}'
            raise ValueError(err)
        if self.base_cols[0] not in catalog.colnames:
            # add the id column at the beginning
            catalog.add_column(
                np.arange(_nobj, dtype=int), 0, self.base_cols[0])
        if cols is None:
            cols = catalog.colnames
        elif isinstance(cols, six.string_types):
            cols = cols.split(',')
        if self.masscol is not None and self.masscol not in cols:
            cols.append(self.masscol)
        if indices is None:
            indices = np.ones(catalog[cols[0]].size, dtype=bool)
        catalog = catalog[cols][indices]
        self.catalog = catalog
        # this only tests that the attributes can be accessed
        # so we raise an issue immediately if they cannot
        try:
            self.catalog.rename_columns(
                self.base_cols[:4], ('name', 'ra', 'dec', 'z'))
        except KeyError:
            err = f'at least one of base_cols {self.base_cols} does not exist.\n' \
                f'available columns:\n{np.sort(self.catalog.colnames)}'
            raise KeyError(err)
        self.mass = self.catalog[self.masscol] if self.masscol is not None else None
        self._coords = None
        self._galactic = None

    def __repr__(self):
        return f'Catalog("{self.name}", indices={self._indices},' \
            f' cols={self._cols})\n' \
            f'{self.catalog}'

    def __str__(self):
        msg = f'{self.label} catalog'
        if self.reference is not None:
            msg = f'{msg} ({self.reference})'
        return f'{msg}\n{self.catalog}'

    def __getitem__(self, key):
        # return Catalog(f'{self.name}[{key}]', self.catalog[key],
        #                base_cols=self.base_cols, masscol=self.masscol)
        return self.catalog[key]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.nobj:
            #i = self.catalog[self.n]
            i = self.__getitem__(self.n)
            self.n += 1
            # return Catalog(f'{self.name}[{i}]', i, base_cols=self.base_cols,
            #                masscol=self.masscol)
            return i
        else:
            raise StopIteration

    @staticmethod
    def list_available():
        print(_available)

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, value):
        self._catalog = value
        # reset coords so that they are recalculated
        self._coords = None

    @property
    def colnames(self):
        return self.catalog.colnames

    @property
    def coords(self):
        """SkyCoord object"""
        if self._coords is None or self._coords.size != self.ra.size:
            self._coords = SkyCoord(
                ra=self.ra, dec=self.dec, unit=self.coord_unit, frame='icrs')
            # reset Galactic coordinates
            self._galactic = None
        return self._coords

    @property
    def galactic(self):
        if self._galactic is None or self._galactic.size != self.ra.size:
            self._galactic = self.coords.transform_to('galactic')
        return self._galactic

    @property
    def l(self):
        return self.galactic.l

    @property
    def b(self):
        return self.galactic.b

    # base properties

    @property
    def obj(self):
        return self.catalog['name'].value

    @property
    def ra(self):
        return u.Quantity(self.catalog['ra'].value, unit=u.deg)

    @property
    def dec(self):
        return u.Quantity(self.catalog['dec'].value, unit=u.deg)

    @property
    def z(self):
        return self.catalog['z'].value

    @property
    def size(self):
        return self.obj.size

    # methods

    def filename(self, relative=False):
        """
        Return the file name of the corresponding catalogs

        Parameters
        ----------
        relative  : bool
                    whether to return the absolute or relative path. Mostly
                    used when downloading catalogs, otherwise it should in
                    general not be changed.

        Returns
        -------
        filename : str
            local filename containing the catalog

        """
        if relative:
            fnames = _filenames.copy()
        if not relative:
            fnames = {key: os.path.join(path, filename)
                      for key, filename in _filenames.items()}
        return fnames[self.name]

    # @staticmethod
    # def download(filename):
    #     """
    #     Download a catalog from the web
    #     
    #     Need to post them somewhere or write down the original website

    #     """
    #     # online location
    #     www = r'http://www.astro.princeton.edu/~sifon/cluster_catalogs/'
    #     online = os.path.join(www, fname)
    #     # local path
    #     path_local = os.path.join(path, os.path.dirname(fname))
    #     if not os.path.isdir(path_local):
    #         os.makedirs(path_local)
    #     local = os.path.join(path, fname)
    #     urllib.urlretrieve(online, local)
    #     return

    def _crossmatch(self, catalog, radius=1*u.arcmin, z=0, cosmo=None, z_width=None):
        assert isinstance(catalog, Catalog)
        raise NotImplementedError

    def query(self, coords=None, ra=None, dec=None, radius=1*u.arcmin, z=0,
              cosmo=None, z_width=None):
        """
        Query the catalog for specific coordinates

        Parameters
        ----------
        coords : ``astropy.coordinates.SkyCoord``
            coordinates to query
        ra, dec : array-like
            Right ascension and declination. Will be ignored if ``coords``
            is provided
        radius : ``astropy.units.Quantity``
            maximum matching radius in angular or physical units
        z : float
            redshift
        cosmo : ``astropy.cosmology.FLRW``
            cosmology to use when ``radius`` is in physical units
        z_width : float
            set in order to allow a maximum redshift difference

        Returns
        -------
        matches : ``astropy.table.Table``
            matching objects
        """
        assert isinstance(radius, u.Quantity)
        if coords is None:
            if ra is None or dec is None:
                raise ValueError('Need to specify either ra, dec or coords')
            coords = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        assert isinstance(coords, SkyCoord)
        distances = self.coords.separation(coords[:,None])
        # matching in physical distance
        if u.get_physical_type(radius.unit) == 'length':
            raise NotImplementedError('matching in physical distance not implemented')
        closest = np.min(distances, axis=0)
        matches = (closest <= radius)
        return self.catalog[matches]
