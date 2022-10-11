"""
Generic astronomical utilities

"""
from __future__ import absolute_import, division, print_function

#import anydbm
from astLib.astWCS import WCS
import warnings


def makereg(x, y, text=None, frame='fk5',
            shape='circle', size='3"',
            properties={'color': 'yellow'},
            commentkeys='', output='ds9.reg'):
    """
    Take a galaxy catalog and create a region file. See
    http://ds9.si.edu/doc/ref/region.html.

    **Only circles for now**

    Parameters
    ----------
    x, y : array-like
        coordinates in appropriate system

    Optional Parameters
    -------------------
    text : array-like
        text to add next to each region
    frame : one of ('fk5', 'image')
        frame in which the region file will be written.
    shape : str
        any shape accepted by DS9 region files.
    size : str
        comma-separated size(s) and rotation angle, as appropirate
        for the selected shape. For instance, for `shape=circle`,
        `size` could be '3"' (three arcsec) or '4' (four pixels);
        and for `shape=box`, `size` could be '3",4",45'
        (three-by-four sq. arcsec rotated by 45 degrees).
    color : str
        any color accepted by DS9 region files.
    output : str
        output file name.

    """
    if shape != 'circle':
        wrn = 'only circle implemented for now'
        warnings.warn(wrn)
        shape = 'circle'
    if text is None:
        X = (x, y)
    else:
        X = (x, y, text)
    with open(output, 'w') as reg:
        print(frame, file=reg)
        props = ' '.join([f'{key}={val}' for key, val in properties.items()])
        print(f'global {props}', file=reg)
        for i in zip(*X):
            xi, yi = i[:2]
            msg = f'{shape}({xi},{yi},{size})'
            if text is not None:
                msg = f'{msg} # text={i[2]}'
            print(msg, file=reg)
    return
