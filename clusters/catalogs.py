"""
Query cluster catalogs

"""
from astLib.astCoords import calcAngSepDeg, dms2decimal, hms2decimal
from astro import cosmology
from itertools import count, izip
from numpy import any as npany, arange, argmin, array, chararray, ones
from os.path import join as osjoin
from astropy.io.fits import getdata
from readfile import table


def catalog_files(catalogs, as_dict=True, squeeze=False):
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

    Returns
    -------
    filenames : list or dict

    """
    path = '/Users/cristobal/Documents/catalogs'
    filenames = {'maxbcg': 'maxbcg/maxBCG.fits',
                 'gmbcg': 'gmbcg/GMBCG_SDSS_DR7_PUB.fit',
                 'hecs2013': 'hecs/2013/data.fits',
                 'orca': 'orca/fullstripe82.fits',
                 'psz1': osjoin('planck', 'PSZ-2013', 'PLCK-DR1-SZ',
                                'COM_PCCS_SZ-union_R1.11.fits'),
                 'psz2': osjoin('planck', 'PSZ-2015',
                                'HFI_PCCS_SZ-union_R2.08.fits'),
                 'redmapper': 'redmapper/redmapper_dr8_public' + \
                              '_v5.10_catalog.fits',
                 'whl': 'whl/whl2015.fits'}
    filenames = {key: osjoin(path, filenames[key]) for key in filenames}
    if isinstance(catalogs, basestring):
        catalogs = catalogs.split(',')
    if as_dict:
        return {key: filenames[key] for key in catalogs}
    if len(catalogs) == 1 and squeeze:
        return filenames[catalogs[0]]
    return [filenames[key] for key in catalogs]


def query(ra, dec, radius=2., unit='arcmin', z=0., cosmo=None,
          catalogs=None, return_single=True, squeeze=False,
          return_values=('name','ra','dec','z','index','dist','dz')):
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
                        * 'maxbcg' (Koester et al. 2007)
                        * 'gmbcg' (Hao et al. 2010)
                        * 'hecs2013' (Rines et al. 2013)
                        * 'hecs2016' (Rines et al. 2016) NOT YET
                        * 'orca' (Geach, Murphy & Bower 2011)
                        * 'psz1' (Planck Collaboration XXIX 2014)
                        * 'psz2' (Planck Collaboration XXVII 2016)
                        * 'redmapper' (Rykoff et al. 2014, v5.10)
                        * 'whl' (Wen, Han & Liu 2012, Wen & Han 2015)
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

    Returns
    -------
      matches   : dict
                  matching elements per catalog. Each requested catalog is
                  a key of this dictionary if more than one catalog was
                  searched or if squeeze==False. If only one catalog was
                  provided and squeeze==True, then return a list with
                  matching entry/ies.
      withmatch : dict
                  for each searched catalog, contains hte indices of the
                  provided clusters for which at least one match was found.
                  The same formatting as for "matches" applies.

    """
    available = ('maxbcg', 'gmbcg', 'hecs2013', 'orca', 'psz1', 'psz2',
                 'redmapper', 'whl')
    # some formatting for convenience
    if not hasattr(ra, '__iter__'):
        ra = [ra]
        dec = [dec]
    if not hasattr(z, '__iter__'):
        z = array([z])
    # in the case of matching by physical radius, demand z > 0
    if unit == 'Mpc' and npany(z <= 0):
        msg = "ERROR: in catalogs.query:"
        msg += " if unit=='Mpc' then z must be larger than 0"
        print msg
        exit()
    if unit == 'Mpc':
        if cosmo is None:
            cosmo = cosmology
        dproj = cosmo.dProj
    # will this fail for numpy.string_?
    if isinstance(ra[0], basestring):
        ra = array([hms2decimal(x, ':') for x in ra])
        dec = array([dms2decimal(y, ':') for y in dec])
    if unit == 'arcmin':
        radius = ones(ra.size) * radius# / 60
    else:
        radius = array([dproj(zi, radius, input_unit='Mpc', unit='arcmin')
                        for zi in z])
    if catalogs is None:
        catalogs = available
    else:
        try:
            catalogs = catalogs.split(',')
        # if this happens then we assume catalogs is already a list
        except ValueError:
            pass
    for name in catalogs:
        if name not in available:
            msg = 'WARNING: catalog {0} not available'.format(name)
            print msg
    filenames = catalog_files(catalogs)
    labels = {'maxbcg': 'maxBCG', 'gmbcg': 'GMBCG', 'hecs2013': 'HeCS',
              'hecs2016': 'HeCS-SZ', 'orca': 'ORCA', 'psz1': 'PSZ1',
              'psz2': 'PSZ2', 'redmapper': 'redMaPPer', 'whl': 'WHL'}
    columns = {'maxbcg': 'none,RAJ2000,DEJ2000,zph',
               'gmbcg': 'OBJID,RA,DEC,PHOTOZ',
               'hecs2013': 'Name,RAJ2000,DEJ2000,z',
               'orca': 'ID,ra_bcg,dec_bcg,redshift',
               'psz1': 'NAME,RA,DEC,REDSHIFT',
               'psz2': 'NAME,RA,DEC,REDSHIFT',
               'redmapper': 'NAME,RA,DEC,Z_LAMBDA',
               'whl': 'WHL,RAJ2000,DEJ2000,zph'}
    for cat in catalogs:
        #filenames[cat] = osjoin(path, filenames[cat])
        columns[cat] = columns[cat].split(',')
    matches = {}
    withmatch = {}
    for cat in available:
        if cat not in catalogs:
            continue
        data = getdata(filenames[cat], ext=1)
        aux = {}
        for name in data.names:
            aux[name] = data[name]
        data = aux
        # if the catalog doesn't give a name
        if columns[cat][0] == 'none':
            columns[cat][0] = 'Name'
            data['Name'] = chararray(data[columns[cat][1]].size, itemsize=4)
            data['Name'][:] = 'none'
        data = [data[v] for v in columns[cat]]
        name, xcat, ycat, zcat = data
        colnames = 'name,ra,dec,z'.split(',')
        close = [(abs(xcat - x) < 2*r/60.) & (abs(ycat - y) < 2*r/60.)
                 for x, y, r in izip(ra, dec, radius)]
        withmatch[cat] = [j for j, c in enumerate(close) if name[c].size]
        dist = [60 * calcAngSepDeg(xcat[j], ycat[j], x, y)
                for j, x, y in izip(close, ra, dec)]
        match = [(d <= r) for d, r in izip(dist, radius)]
        withmatch[cat] = array([w for w, m in izip(count(), match)
                                if w in withmatch[cat] and name[m].size])
        if return_single:
            match = [argmin(d) if d.size else None for d in dist]
        matches[cat] = {}
        # keeping them all now because they may be needed for other properties
        for name, x in izip(colnames, data):
            matches[cat][name] = array([x[j][mj] for w, j, mj
                                        in izip(count(), close, match)
                                        if w in withmatch[cat]])
        if 'index' in return_values:
            matches[cat]['index'] = array([arange(xcat.size)[j][m]
                                           for w, j, m in izip(count(),
                                                               close, match)
                                           if w in withmatch[cat]])
        if 'dist' in return_values:
            matches[cat]['dist'] = array([d[m] for w, d, m
                                          in izip(count(), dist, match)
                                          if w in withmatch[cat]])
            if unit == 'Mpc':
                matches[cat]['dist'] *= array([dproj(zi, 1, unit='Mpc',
                                                     input_unit='arcmin')
                                               for zi in matches[cat]['z']])
        if 'dz' in return_values:
            matches[cat]['dz'] = array([zcat[j][m] - zj for w, j, m, zj
                                        in izip(count(), close, match, z)
                                        if w in withmatch[cat]])
        for key in matches[cat].keys():
            if key not in return_values:
                matches[cat].pop(key)
        if not return_single and name[j][match].size == 1:
            for key in matches[cat].keys():
                matches[cat][key] = matches[cat][key][0]
    if len(catalogs) == 1 and squeeze:
        return matches[catalogs[0]], withmatch[catalogs[0]]
    return matches, withmatch


def retrieve_objects(catalog, indices=None, cols=None, squeeze=False):
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
    filename = catalog_files(catalog, as_dict=False, squeeze=True)
    print 'filename =', filename
    data = getdata(filename, ext=1)
    if cols is None:
        cols = data.names
    elif isinstance(cols, basestring):
        cols = cols.split(',')
    if indices is None:
        indices = numpy.ones(data[cols[0]].size, dtype=bool)
    data = [data[col][indices] for col in cols]
    if len(cols) == 1 and squeeze:
        return data[0]
    return data


