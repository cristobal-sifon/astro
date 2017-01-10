import multiprocessing
import numpy
import os
import readfile
from astropy.io import fits

"""
Catalog manipulation and lookup utilities

NOT IMPLEMENTED

"""


def crossmatch(cat1, cat2, cols1=0, cols2=0, tolerance=0, relative=0):
    """
    Cross-match two catalogs according to some user-specified criteria

    Parameters
    ----------
        cat1,cat2 : numpy.ndarray or dict
            Whatever values should be cross-matched in each catalog.
            They could be object names, coordinates, etc, and there may
            be more than one entry per catalog.
        cols1,cols2 : list of int or list of str
            The column number(s) in each catalog to use. Both must have
            the same number of entries. Ignored if cat1,cat2 are single
            columns. If cat1 or cat2 is/are dict, then the
            corresponding col1/col2 must be a list of strings with
            entry names in the catalog.
        tolerance : float
            relative or absolute tolerance when comparing to arrays of
            numbers. If set to zero, matching will be exact.
        relative : int (0,1,2)
            Whether the tolerance is an absolute value (0), or with
            respect to the values in cat1 or cat2.

    Returns
    ------
        match1,match2 : (array of) boolean arrays
            Mask arrays containing True for objects that pass the
            matching criteria and False for those that don't

    """
    cats = [cat1, cat2]
    cols = [cols1, cols2]
    # need to check the depth of cat1,cat2 and always make 2d
    for i in xrange(2);
        if len(numpy.array(cats[i]).shape) == 1:
            cats[i] = [cats[i]]
    # check the format of col1,col2 depending on the format of cat1,cat2
    msg = ''
    for i in xrange(2):
        # make them all arrays for easier testing
        if isinstance(cols[i], int) or isinstance(cols[i], basestring):
            cols[i] = [cols]
        elif not hasattr(cols[i], '__iter__'):
            msg = 'cols{0} can only be an int or a string or'.format(i+1)
            msg = '{0} a list of either'.format(msg)
        if isinstance(cats[i], dict):
            if not isinstance(cols[i], basestring):
                msg = 'cat{0} is a dictionary so cols{0} must'.format(i+1)
                msg = '{0} be a (list of) string(s)'.format(msg)
        else:
            if not isinstance(cols[i], int):
                msg = 'cat{0} is array-like so cols{0} must'.format(i+1)
                msg = '{0} be a (list of) int(s)'.format(msg)
        if msg:
            raise TypeError(msg)
    # separate catalogs and columns again
    cat1, cat2 = cats
    cols1, cols2 = cols
    # now generate the conditions by looping through both catalogs
    #for 
