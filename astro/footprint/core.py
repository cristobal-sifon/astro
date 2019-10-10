from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


class Footprint:

    def __init__(self, name=None, filename=None, format=None, footprint=None,
                 default_plot_wrap=180):
        """
        Parameters
        ----------
        name : str, optional
            arbitrary name
        file : str, optional
            
        """
        self.name = name
        self.filename = filename
        self.format = format
        if self.filename is not None:
            self.file = FootprintFile(self.filename, self.format)
            self.footprint = self.file.read()
        else:
            self.footprint = footprint
        self.default_plot_wrap = default_plot_wrap

    #@property
    #def footprint(self):
        #"""(RA, Dec) coordinates in decimal degrees"""
        #return self._footprint

    """
    @footprint.setter
    def footprint(self, coords, format='array'):
        assert format in ('array', 'reg')
        if format == 'array':
            # a footprint might consist of multiple polygons
            if not hasattr(coords[0][0], '__iter__'):
                coords = [coords]
            for i in range(len(coords)):
                coords[i] = np.array(coords[i])
                assert coords[i].shape[1] == 2, \
                    'Each polygon within the footprint must have shae (N,2);' \
                    ' polygon #{0} has shape {1}'.format(i, coords[i].shape)
        else:
            coords = self.file.read()
        self._footprint = np.array(coords)
    """


    #def _read_footprint(self):
        


    def _count_crossings(self, x, y):
        """Auxiliary function used by ``self.in_footprint``

        Count the number of times a line joining the point (x,y) and
        infinity cross the survey boundaries

        Perhaps it makes sense to have a new repository `geometry` for this

        Parameters
        ----------
        x, y : float
            coordinates

        """
        crossings = 0
        for p, polygon in enumerate(self.footprint):
            for i in range(1, len(polygon)):
                side = np.array([polygon[i-1], polygon[i]])
                # only need to move in one direction. If the entire side is below
                # the point then we don't need to test it
                if side[:,1].max() < y:
                    continue
                # similarly if the point is to the left or right of the footprint
                if (side[:,0].min() > x or side[:,0].max() < x):
                    continue
                crossings += (np.prod(x - side[:,0]) < 0)
        return crossings


    def in_footprint(self, ra, dec):
        """Identify which objects lie within the footprint."""
        if not np.iterable(ra):
            ra = [ra]
            dec = [dec]
        ncrossings = np.array(
            [self._count_crossings(x, y) for x, y in zip(ra, dec)])
        return (ncrossings % 2 == 1)


    def plot(self, ax=None, wrap=None, **kwargs):
        if wrap is None:
            wrap = self.default_plot_wrap
        if ax is not None and 'label' in kwargs:
            label = kwargs.pop('label')
            ax.plot([], [], label=label, **kwargs)
        patches = []
        for field in self.footprint:
            patches.append(Polygon(field, closed=True, **kwargs))
            if np.max(field[:,0]) > wrap:
                wrapped = field.copy()
                wrapped[:,0] = wrapped[:,0] - 2*wrap
                patches.append(Polygon(wrapped, closed=True, **kwargs))
        if ax is not None:
            for patch in patches:
                ax.add_patch(patch)
        return patches


class FootprintFile:

    def __init__(self, filename, format=None):
        """Initialize footprint file

        if ``format`` is not provided, it will be guessed from
        ``filename``
        """
        self.filename = filename
        self.format = format


    def read(self):
        """Only .reg implemented"""
        footprint = []
        if self.format is None:
            ext = self.filename.split('.')[-1]
            if ext in ('dat', 'txt'):
                self.format = 'array'
            elif ext == 'reg':
                self.format = 'reg'
        if self.format == 'reg':
            footprint = self.read_reg()
        return footprint


    def read_reg(self):
        footprint = []
        with open(self.filename) as f:
            for line in f:
                if line.startswith('polygon'):
                    line = line[line.index('(')+1:line.index(')')].split(',')
                    ra = np.array(line[::2], dtype=float)
                    dec = np.array(line[1::2], dtype=float)
                    footprint.append(np.transpose([ra, dec]))
        return np.array(footprint)


