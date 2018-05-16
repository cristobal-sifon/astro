from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np


class Survey:

    def __init__(self, name=None):
        """
        add option to read stored footprint for known surveys
        """
        self.name = name
        self._footprint = None
        if self.name in self.known_surveys.keys():
            self.footprint = self.known_surveys[self.name]


    @property
    def footprint(self):
        """(RA, Dec) coordinates in decimal degrees"""
        return self._footprint


    @footprint.setter
    def footprint(self, coords):
        # a footprint might consist of multiple polygons
        if not hasattr(coords[0][0], '__iter__'):
            coords = [coords]
        for i in range(len(coords)):
            coords[i] = np.array(coords[i])
            assert coords[i].shape[1] == 2, \
                'Each polygon within the footprint must have shae (N,2);' \
                ' polygon #{0} has shape {1}'.format(i, coords[i].shape)
        self._footprint = coords


    @property
    def known_surveys(self):
        # add footprints of known surveys here - or in a folder?
        return {}


    def in_footprint(self, ra, dec):
        """Identify which objects lie within the footprint."""
        ncrossings = np.array(
            [self.count_crossings(x, y) for x, y in zip(ra, dec)])
        return (ncrossings % 2 == 1)


    def count_crossings(self, x, y):
        """Perhaps it makes sense to have a new repository `geometry` for this"""
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

