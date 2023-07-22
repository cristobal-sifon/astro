"""
Catalog manipulation and lookup utilities

NOT IMPLEMENTED

"""
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import numpy as np
import os
import sys

class SourceCatalog(Table):
    """SourceCatalog class
    
    An astropy table with additional methods tailored for astronomical
    source catalogs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ellipses(self, frame='world', size=5):
        frame = frame.upper()
        assert frame in ('IMAGE', 'WORLD'), "frame must be 'IMAGE' or 'WORLD'"
        xcol = 'X_{0}'.format(frame)
        ycol = 'Y_{0}'.format(frame)
        acol = 'A_{0}'.format(frame)
        bcol = 'B_{0}'.format(frame)
        tcol = 'THETA_{0}'.format(frame)
        ellipses \
            = [Ellipse((x,y), size*a, size*b, angle=theta)
               for x, y, a, b, theta 
               in zip(*[self.data[col] for col in (xcol,ycol,acol,bcol,tcol)])]
        return ellipses
    
    def patch_collection(self, frame='world', size=5, facecolor='none',
                         **kwargs):
        shapes = ellipses(frame=frame, size=size)
        return PatchCollection(shapes, facecolor=facecolor, **kwargs)

    def plot(self, ax=None, frame='world', size=5, facecolor='none', **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.add_collection(self.patch_collection(
            frame=frame, size=size, facecolor=facecolor, **kwargs))
        return ax
