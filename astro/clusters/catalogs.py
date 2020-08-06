"""Query locally-stored cluster catalogs

The following attributes may be modified by the user depending on their
needs:

    columns : list of columns to be loaded from each catalog by `query`
    labels : label of each available catalog

"""
from __future__ import absolute_import, division, print_function

from astLib.astCoords import calcAngSepDeg, dms2decimal, hms2decimal
from astro import cosmology
from astropy.io import ascii
from astropy.io.fits import getdata
from astropy.table import Table
from itertools import count
import numpy as np
import os
import urllib
import six
import sys

if sys.version_info[0] == 2:
    from itertools import izip as zip

# these should not be modified
_available = (
    'abell', 'actpol', 'advact', 'gmbcg', 'hecs2013', 'madcows', 'maxbcg',
    'mcxc', 'orca', 'psz1', 'psz2', 'redmapper', 'spt-sz', 'whl')
_filenames = {
    'abell': 'abell/aco1989_ned.tbl',
    'advact': 'advact/AdvACT_fixedSNR_gtr_5_20190304.fits',
    'actpol': 'actpol/E-D56Clusters.fits',
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
    'whl': 'whl/whl2015.fits'}

# the user may choose to modify these
columns = {
    'abell': 'Object Name,RA(deg),DEC(deg),Redshift',
    'advact': 'name,RADeg,decDeg,redshift',
    'actpol': 'name,RADeg,decDeg,z',
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
    'whl': 'WHL,RAJ2000,DEJ2000,zph'}
labels = {
    'abell': 'Abell',
    'advact': 'AdvACT',
    'actpol': 'ACTPol',
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
    'whl': 'WHL'}
# all catalogs are here
if 'DOCS' in os.environ:
    path = os.environ['DOCS']
else:
    path = os.path.join(os.environ['HOME'], 'Documents')
path = os.path.join(path, 'catalogs')
# these serve to restore the above attributes if necessary
_columns = columns.copy()
_labels = labels.copy()
_path = '{0}'.format(path)



def download(fname):
    """
    Download a catalog from the web

    """
    # online location
    www = r'http://www.astro.princeton.edu/~sifon/cluster_catalogs/'
    online = os.path.join(www, fname)
    # local path
    path_local = os.path.join(path, os.path.dirname(fname))
    if not os.path.isdir(path_local):
        os.makedirs(path_local)
    local = os.path.join(path, fname)
    urllib.urlretrieve(online, local)
    return


def filename(catalogs, as_dict=True, squeeze=False, relative=False):
    """
    Return the file name of the corresponding catalogs

    Parameters
    ----------
    catalogs  : str or list of strings
                see description in `query()`
    as_dict   : bool
                whether to return a list or a dictionary
    squeeze   : bool
                whether to return a str instead of a list if only one
                catalog is requested. `as_dict` takes precedence over
                `squeeze`
    relative  : bool
                whether to return the absolute or relative path. Mostly
                used when downloading catalogs, otherwise it should in
                general not be changed.

    Returns
    -------
    fnames : list or dict

    """
    if relative:
        fnames = _filenames.copy()
    if not relative:
        fnames = {key: os.path.join(path, filename)
                  for key, filename in _filenames.items()}
    if isinstance(catalogs, six.string_types):
        catalogs = catalogs.split(',')
    if as_dict:
        return {key: fnames[key] for key in catalogs}
    if len(catalogs) == 1 and squeeze:
        return fnames[catalogs[0]]
    return [fnames[key] for key in catalogs]


def load(catalog, indices=None, cols=None, squeeze=False):
    """
    Retrieve data from a catalog using indices within it (which may
    be obtained using `query()`)

    Parameters
    ----------
    catalog   : str
                name of the catalog
    indices   : array of int (optional)
                indices of the objects whose information is requested.
                These indices may be obtained by running `query()`
                using the positions of the objects first. If not given,
                the full catalog will be returned.
    cols      : str or list of str (optional)
                list or comma-separated names of the columns wanted. If
                not given, all columns are returned.
    squeeze   : bool
                whether to return a one-element list if only one column
                is given

    Returns
    -------
    data      : list of arrays
                requested catalog entries

    """
    if not isinstance(catalog, six.string_types):
        msg = 'argument catalog must be a string'
        raise TypeError(msg)
    fname = filename(catalog, as_dict=False, squeeze=True)
    # load. Some may have special formats
    if catalog in ('madcows','spt-sz'):
        data = ascii.read(fname, format='cds')
    elif catalog == 'abell':
        data = ascii.read(fname, format='ipac')
        # fill masked elements
        data = data.filled()
        #noz = (data[columns[data][3]].values == 1e20)
        #data[noz] = -1
    else:
        data = Table(getdata(fname, ext=1, ignore_missing_end=True))

    if cols is None:
        cols = data.colnames
    elif isinstance(cols, six.string_types):
        cols = cols.split(',')
    if indices is None:
        indices = np.ones(data[cols[0]].size, dtype=bool)
    #data = [data[col][indices] for col in cols]
    data = data[cols][indices]
    if len(cols) == 1 and squeeze:
        return data[0]
    return data


def query(ra, dec, radius=2., unit='arcmin', z=0., cosmo=None,
          catalogs=None, return_single=True, squeeze=False,
          return_values=('name','ra','dec','z','index','dist','dz'),
          error_if_missing=False):
    """
    Query different catalogs for clusters close to a given set of coordinates.

    To-do's:
        -possibility to return survey observables

    Parameters
    ----------
      ra        : (array of) float or str
                  if float, should be the RA in decimal degrees; if str,
                  should be in hms format ('hh:mm:ss', requires astLib)
      dec       : (array of) float or str
                  if float, should be the Dec in decimal degrees; if str,
                  should be in dms format ('dd:mm:ss', requires astLib)
      radius    : float (default 2)
                  the search radius, in units given by argumetn "unit"
      unit      : {'arcmin', 'Mpc'}
                  the units of argument "radius". If Mpc, then argument "z"
                  must be larger than zero.
      z         : (array of) float (optional)
                  redshift(s) of the cluster(s). Must be >0 if unit=='Mpc'.
      cosmo     : module astro.cosmology (optional)
                  if the matching is done in physical distance then pass
                  this module to make sure all cosmological parameters are
                  used consistently with the parent code!
      catalogs  : str or list of str (optional)
                  list or comma-separated names of catalogs to be searched.
                  If not given, all available catalogs are searched. Allowed
                  values are:
                        Optical catalogs:
                        * 'maxbcg' (Koester et al. 2007)
                        * 'gmbcg' (Hao et al. 2010)
                        * 'hecs2013' (Rines et al. 2013)
                        * 'hecs2016' (Rines et al. 2016) NOT YET
                        * 'orca' (Geach, Murphy & Bower 2011)
                        * 'redmapper' (Rykoff et al. 2014, v6.3)
                        * 'whl' (Wen, Han & Liu 2012, Wen & Han 2015)
                        SZ catalogs:
                        * 'actpol' (Hilton et al. 2018)
                        * 'psz1' (Planck Collaboration XXIX 2014)
                        * 'psz2' (Planck Collaboration XXVII 2016)
                        * 'spt-sz' (Bleem et al. 2015)
      return_single : bool
                  whether to return the single closest matching cluster (if
                  within the search radius) or all those falling within the
                  search radius
      return_values : any subset of ('name','ra','dec','z','index','dist','dz')
                  what elements to return. 'index', 'dist' and 'dz' refer to
                  the index in the catalog and the distances of the matching
                  cluster(s) in arcmin and redshift space, respectively.
                  NOTE: altering the order of the elements in return_values
                  will have no effect in the order in which they are returned!
      squeeze   : bool
                  whether to return a list instead of a dictionary if only
                  one catalog is provided
      error_if_missing : bool
                  if True, will raise an IOError if *any* of the
                  requested catalogs are missing. Otherwise they will
                  just be skipped.

    Returns
    -------
      matches   : dict
                  matching elements per catalog. Each requested catalog is
                  a key of this dictionary if more than one catalog was
                  searched or if squeeze==False. If only one catalog was
                  provided and squeeze==True, then return a list with
                  matching entry/ies. Distances are in the units specified
                  in `unit` ('arcmin' by default)
      withmatch : dict
                  for each searched catalog, contains the indices *in the
                  input list* for which at least one match was found.
                  The same formatting as for "matches" applies.

    Notes
    -----
      If none of the catalogs exists and the user chooses not to
      download any of them, then an IOError will be raised.

    """
    # some formatting for convenience
    if not np.iterable(ra):
        ra = np.array([ra])
        dec = np.array([dec])
    assert ra.size > 0, 'Received empty catalog'
    if not np.iterable(z):
        z = np.array([z])
    # in the case of matching by physical radius, demand z > 0
    if unit == 'Mpc' and np.any(z <= 0):
        msg = "ERROR: in catalogs.query:"
        msg += " if unit=='Mpc' then z must be larger than 0"
        print(msg)
        exit()
    if unit == 'Mpc':
        if cosmo is None:
            cosmo = cosmology
        dproj = cosmo.dProj
    # will this fail for np.string_?
    if isinstance(ra[0], six.string_types):
        ra = np.array([hms2decimal(x, ':') for x in ra])
        dec = np.array([dms2decimal(y, ':') for y in dec])
    # if using physical units, each matching radius is different
    if unit in ('arcsec','arcmin','deg'):
        radius = np.ones(ra.size) * radius
    else:
        radius = np.array(
            [dproj(zi, radius, input_unit=unit, unit='deg') for zi in z])
    if catalogs is None:
        catalogs = available
    else:
        try:
            catalogs = catalogs.split(',')
        # if this happens then we assume catalogs is already a list
        except AttributeError:
            pass
    for name in catalogs:
        if name not in _available:
            msg = 'WARNING: catalog {0} not available'.format(name)
            print(msg)
    fnames = filename(catalogs)

    # if any of the requested catalogs does not exist, download
    bad = []
    for i, cat in enumerate(catalogs):
        if not os.path.isfile(fnames[cat]):
            msg = 'WARNING: there does not seem to be a local copy'
            msg = '{0} of the {1} catalog'.format(msg, cat)
            msg = '{0}. You need a local copy to query the'.format(msg)
            msg = '{0} catalog; would you like to download it?'.format(msg)
            msg = '{0} [y/N] '.format(msg)
            do_download = raw_input(msg)
            if do_download[0].lower() == 'y':
                download(filename(cat, relative=True, as_dict=False)[0])
            elif error_if_missing:
                raise IOError('catalog {0} does not exist'.format(cat))
            else:
                bad.append(i)
    # if some haven't been found, remove
    # these two loops need to be separate because otherwise removing
    # items from `catalogs` will mess up the first instruction
    for i in bad:
        fnames.pop(catalogs[i])
    for i in bad:
        catalogs.pop(i)
    if len(catalogs) == 0:
        msg = 'No catalogs available for query'
        raise IOError(msg)

    matches = {}
    withmatch = {}
    # distance conversion
    dc = {'arcsec': 3600, 'arcmin': 60, 'deg': 1}
    # the "else" is for physical units, which are converted to deg above
    dc = dc[unit] if unit in dc else 1
    for cat in _available:
        if cat not in catalogs:
            continue
        data = load(cat)
        aux = {}
        for name in data.colnames:
            aux[name] = data[name]
        data = aux
        if isinstance(columns[cat], six.string_types):
            columns[cat] = columns[cat].split(',')
        # if the catalog doesn't give a name
        if columns[cat][0] == 'none':
            columns[cat][0] = 'Name'
            data['Name'] = np.chararray(data[columns[cat][1]].size, itemsize=4)
            data['Name'][:] = 'none'
        data = [data[v] for v in columns[cat]]
        name, xcat, ycat, zcat = data
        colnames = 'name,ra,dec,z'.split(',')
        close = [(abs(xcat - x) < 2*r/dc) & (abs(ycat - y) < 2*r/dc)
                 for x, y, r in zip(ra, dec, radius)]
        # objects in the target catalog that match something in the input
        withmatch[cat] = np.array(
            [j for j, c in enumerate(close) if name[c].size])
        # distances from each input object to all matches
        dist = [dc * np.array(calcAngSepDeg(xcat[j], ycat[j], x, y))
                for j, x, y in zip(close, ra, dec)]
        # I don't understand why some elements are scalars but just fixing for now
        dist = [d if np.iterable(d) else np.array([d]) for d in dist]

        # whether the matches meet the distance criterion
        match = np.array([(d <= r) for d, r in zip(dist, radius)])
        withmatch[cat] = np.array(
            [w for w, m in enumerate(match)
             if w in withmatch[cat] and m.sum()])

        if return_single:
            match = [np.argmin(d) if d.size else None for d in dist]
        matches[cat] = {}
        # keeping them all now because they may be needed for other properties
        for name, x in zip(colnames, data):
            matches[cat][name] = np.array(
                [x[j][mj] for w, j, mj in zip(count(), close, match)
                 if w in withmatch[cat]])
        if 'index' in return_values:
            matches[cat]['index'] = np.array(
                [np.arange(xcat.size)[j][m]
                 for w, j, m in zip(count(), close, match)
                 if w in withmatch[cat]])
        if 'dist' in return_values:
            matches[cat]['dist'] = np.array(
                [d[m] for w, d, m in zip(count(), dist, match)
                 if w in withmatch[cat]])
            if unit == 'Mpc':
                matches[cat]['dist'] *= np.array(
                    [dproj(zi, 1, unit='Mpc', input_unit='arcmin')
                     for zi in matches[cat]['z']])
        if 'dz' in return_values:
            matches[cat]['dz'] = np.array(
                [zcat[j][m] - zj for w, j, m, zj
                 in zip(count(), close, match, z) if w in withmatch[cat]])
        for key in matches[cat].keys():
            if key not in return_values:
                matches[cat].pop(key)
        if not return_single and name[j][match].size == 1:
            for key in matches[cat].keys():
                matches[cat][key] = matches[cat][key][0]
        print('{0}: {1} matches'.format(labels[cat], withmatch[cat].size))
    if len(catalogs) == 1 and squeeze:
        return matches[catalogs[0]], withmatch[catalogs[0]]
    return matches, withmatch


#### Auxiliary functions


def list_available():
    print('Available catalogs:')
    print(_available)
    return


def reset_columns():
    global columns
    columns = _columns.copy()
    return


def reset_labels():
    global labels
    labels = _labels.copy()
    return


def reset_path():
    global path
    path = _path.copy()
    return



