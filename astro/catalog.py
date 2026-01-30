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
        """
        Initialize the SourceCatalog object

        Parameters
        ----------
        *args : tuple
            List of positional arguments to be passed to the Table constructor.
        **kwargs : dict
            List of keyword arguments to be passed to the Table constructor.

        Notes
        -----
        The constructor is a thin wrapper around the astropy.table.Table constructor.
        Please refer to the astropy.table.Table documentation for more information.
        """
        super().__init__(*args, **kwargs)

    @property
    def size(self):
        """
        Number of sources in the catalog.

        Returns
        -------
        int
            number of sources in the catalog
        """
        return self.__getitem__(self.colnames[0]).shape[0]

    def ellipses(self, mask: np.ndarray = None, frame: str = "world", size: float = 5):
        """
        Return a list of Ellipse objects representing the sources in the catalog.

        Parameters
        ----------
        mask : np.ndarray, optional
            Boolean array to select a subset of sources.
        frame : str, optional
            Coordinate frame to use for the ellipses, either "image" or "world".
        size : float, optional
            Scaling factor for the size of the ellipses.

        Returns
        -------
        list of Ellipse objects
            Each ellipse is defined by its center coordinates (x, y), semi-major axis
            length, semi-minor axis length, and orientation angle (in degrees).
        """
        frame = frame.upper()
        assert frame in ("IMAGE", "WORLD"), "frame must be 'IMAGE' or 'WORLD'"
        xcol = f"X_{frame}"
        ycol = f"Y_{frame}"
        acol = f"A_{frame}"
        bcol = f"B_{frame}"
        tcol = f"THETA_{frame}"
        if mask is None:
            mask = np.ones(self.size, dtype=bool)
        ellipses = [
            Ellipse((x, y), size * a, size * b, angle=theta)
            for x, y, a, b, theta in zip(
                *[self.__getitem__(col)[mask] for col in (xcol, ycol, acol, bcol, tcol)]
            )
        ]
        return ellipses

    def patch_collection(
        self,
        frame="world",
        size=5,
        facecolors="none",
        edgecolors="C3",
        mask=None,
        **kwargs,
    ):
        """
        Create a `PatchCollection` of ellipses representing the sources in the catalog.

        Parameters
        ----------
        frame : str, optional
            Coordinate frame to use for the ellipses, either "image" or "world".
        size : float, optional
            Scaling factor for the size of the ellipses.
        facecolors : color or sequence of colors, optional
            Color of the ellipse faces.
        edgecolors : color or sequence of colors, optional
            Color of the ellipse edges.
        mask : np.ndarray, optional
            Boolean array to select a subset of sources.
        kwargs : dict, optional
            Additional keyword arguments passed to `matplotlib.collections.PatchCollection`.

        Returns
        -------
        PatchCollection
            A `matplotlib.collections.PatchCollection` object containing the ellipses.
        """
        ellipses = self.ellipses(frame=frame, size=size, mask=mask)
        return PatchCollection(
            ellipses, facecolors=facecolors, edgecolors=edgecolors, **kwargs
        )

    def plot(
        self,
        ax=None,
        frame="world",
        size=5,
        edgecolors="C3",
        facecolors="none",
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
                edgecolors=edgecolors,
                facecolors=facecolors,
                **kwargs,
            )
        )
        if names is not None:
            frame = frame.upper()
            if names_kwargs is None:
                names_kwargs = {}
            if "color" not in names_kwargs:
                names_kwargs["color"] = edgecolors
            if frame == "WORLD" and "transform" not in names_kwargs:
                names_kwargs["transform"] = ax.get_transform("world")
            for name, x, y in self.iterrows(names, f"X_{frame}", f"Y_{frame}"):
                ax.annotate(name, xy=(x, y), **names_kwargs)
        return ax

    def save_regions(
        self,
        output: str,
        mask: np.ndarray = None,
        header: dict = None,
        frame: str = "world",
        size: float = 5,
        textcol: str = None,
        overwrite: bool = False,
    ):
        """Save catalog as region file. Each source is represented as an ellipse

        see https://ds9.si.edu/doc/ref/region.html for conventions and definitions

        Parameters
        ----------
        output : ``str``
            output file name
        mask : ``array``, optional
            mask to apply to the catalog before saving
        header : ``dict``, optional
            global region properties such as color, width or font size
        frame : ``str``
            coordinate frame. Should correspond to a coordinate frame included
            in the table used to generate ``self`` (e.g. ``"image"`` if the
            catalog contains columns ``X_IMAGE`` and ``Y_IMAGE``)
        size : ``float``
            size to rescale the ellipses, relative to the values included in the table
        textcol : ``str``, optional
            column name to use as text for each ellipse
        """
        if os.path.isfile(output) and not overwrite:
            raise OSError(
                f'File {output} already exists. If you mean to replace it then use the argument "overwrite=True".'
            )
        ell = self.ellipses(mask=mask, frame=frame, size=size)
        # if properties is not None:
        #     for key, val in properties.items():
        #         assert len(val) == len(
        #             ell
        #         ), "properties must have the same length as the catalog, received"
        if mask is None:
            mask = np.ones(self.size, dtype=bool)
        if textcol is None:
            add_text = False
        else:
            add_text = True
            text = self.__getitem__(textcol)[mask]
        with open(output, "w") as f:
            if header is not None:
                line = "global"
                for key, val in header.items():
                    line = f"{line} {key}={val}"
                print(line, file=f)
            if frame == "world":
                print("fk5", file=f)
            for i, e in enumerate(ell):
                line = f"ellipse({e.center[0]},{e.center[1]},{e.width},{e.height},{e.angle})"
                if add_text:
                    line = f"{line} # text={{{text[i]}}}"
                print(line, file=f)
        return
