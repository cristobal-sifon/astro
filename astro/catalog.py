"""Catalog manipulation and lookup utilities"""

from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import numpy as np
import os
import sys


class SourceCatalog(Table):
    """SourceCatalog class

    An ``astropy.table.Table`` object with additional methods tailored for astronomical
    source catalogs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ellipses(self, frame="world", size=5):
        frame = frame.upper()
        assert frame in ("IMAGE", "WORLD"), "frame must be 'IMAGE' or 'WORLD'"
        xcol = f"X_{frame}"
        ycol = f"Y_{frame}"
        acol = f"A_{frame}"
        bcol = f"B_{frame}"
        tcol = f"THETA_{frame}"
        ellipses = [
            Ellipse((x, y), size * a, size * b, angle=theta)
            for x, y, a, b, theta in zip(
                *[self.__getitem__(col) for col in (xcol, ycol, acol, bcol, tcol)]
            )
        ]
        return ellipses

    def patch_collection(
        self, frame="world", size=5, facecolor="none", edgecolor="C3", **kwargs
    ):
        ellipses = self.ellipses(frame=frame, size=size)
        return PatchCollection(
            ellipses, facecolor=facecolor, edgecolor=edgecolor, **kwargs
        )

    def plot(
        self,
        ax=None,
        frame="world",
        size=5,
        edgecolor="C3",
        facecolor="none",
        names=None,
        names_kwargs=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        ax : ``matplotlib.Axes``
        frame : ``str``
            coordinate frame. Should correspond to a coordinate frame included
            in the table used to generate ``self`` (e.g. ``"image"`` if the
            catalog contains columns ``X_IMAGE`` and ``Y_IMAGE``)
        size : ``float``
            size to rescale the ellipses, relative to the values included in the table
        facecolor : any value accepted as a color by ``matplotlib``
        names : ``str``
            name of a column, which must be present in the table, shown next to each
            ellipse in the plot
        kwargs : ``dict``, optional
            passed to ``matplotlib.collections.PatchCollection``
        names_kwargs : ``dict``, optional
            passed to ``ax.annotate`` if ``names`` is provided
        """
        if ax is None:
            ax = plt.gca()
        if names is not None:
            assert names in self.colnames, f"column {names} not in catalog"
        if frame == "world" and "transform" not in kwargs:
            kwargs["transform"] = ax.get_transform("world")
        ax.add_collection(
            self.patch_collection(
                frame=frame,
                size=size,
                edgecolor=edgecolor,
                facecolor=facecolor,
                **kwargs,
            )
        )
        if names is not None:
            frame = frame.upper()
            if names_kwargs is None:
                names_kwargs = {}
            if "color" not in names_kwargs:
                names_kwargs["color"] = edgecolor
            if frame == "world" and "transform" not in names_kwargs:
                names_kwargs["transform"] = ax.get_transform("world")
            for name, x, y in self.iterrows(names, f"X_{frame}", f"Y_{frame}"):
                ax.annotate(name, xy=(x, y), **names_kwargs)
        return ax
