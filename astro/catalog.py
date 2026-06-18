"""Catalog manipulation and lookup utilities"""

from astropy import units as u
from astropy.coordinates import SkyCoord
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

    def query(
        self,
        coords=None,
        ra=None,
        dec=None,
        radius=1 * u.arcsec,
    ):
        """
        Query the catalog for specific sources

        Parameters
        ----------
        name : array-like, dtype string
            name(s) to query. In this case, ``radius`` is ignored. Name(s) will be
            matched against the column given in the first element of ``base_cols``
        coords : ``astropy.coordinates.SkyCoord``
            coordinates to query. Will be ignored if ``name`` is provided
        ra, dec : array-like
            Right ascension and declination. Will be ignored if ``coords``
            or ``name`` are provided
        radius : ``astropy.units.Quantity``
            maximum matching radius in angular or physical units

        Returns
        -------
        matches : ``astropy.table.Table``
            matching objects
        """
        assert isinstance(radius, u.Quantity)
        if coords is None:
            if ra is None or dec is None:
                raise ValueError("Need to specify either ra, dec or coords")
            coords = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        assert isinstance(coords, SkyCoord)
        if coords.isscalar:
            coords = coords.reshape((1,))
        coords = coords.icrs
        if "RA" in self.colnames and "DEC" in self.colnames:
            cat_ra = np.asarray(self["RA"])
            cat_dec = np.asarray(self["DEC"])
        elif "X_WORLD" in self.colnames and "Y_WORLD" in self.colnames:
            cat_ra = np.asarray(self["X_WORLD"])
            cat_dec = np.asarray(self["Y_WORLD"])
        else:
            raise ValueError(
                "Catalog must contain either (RA, DEC) or (X_WORLD, Y_WORLD) columns"
            )
        if u.get_physical_type(radius.unit) == "length":
            raise NotImplementedError("matching in physical distance not implemented")
        radius_deg = (1.2 * radius).to(u.deg).value
        if radius_deg > 0:
            if coords.size == 1:
                ra0 = coords.ra.deg[0]
                dec0 = coords.dec.deg[0]
                dra = np.abs((cat_ra - ra0 + 180.0) % 360.0 - 180.0)
                pre_mask = (dra <= radius_deg) & (np.abs(cat_dec - dec0) <= radius_deg)
            else:
                pre_mask = np.zeros(cat_ra.size, dtype=bool)
                coords_ra = coords.ra.deg
                coords_dec = coords.dec.deg
                chunk_size = 256
                for i in range(0, coords_ra.size, chunk_size):
                    ra_chunk = coords_ra[i : i + chunk_size][:, None]
                    dec_chunk = coords_dec[i : i + chunk_size][:, None]
                    dra = np.abs((cat_ra[None, :] - ra_chunk + 180.0) % 360.0 - 180.0)
                    dec_ok = np.abs(cat_dec[None, :] - dec_chunk) <= radius_deg
                    pre_mask |= np.any((dra <= radius_deg) & dec_ok, axis=0)
            if not np.any(pre_mask):
                return self[pre_mask]
            cat_ra = cat_ra[pre_mask]
            cat_dec = cat_dec[pre_mask]
            cat_view = self[pre_mask]
        else:
            cat_view = self
        cat_coords = SkyCoord(ra=cat_ra, dec=cat_dec, unit="deg", frame="icrs")
        distances = cat_coords.separation(coords[:, None])
        closest = np.min(distances, axis=0)
        matches = closest <= radius
        return cat_view[matches]

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
