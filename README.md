# astro

Common astronomical utilities, with a focus on galaxy cluster analyses

(c) Cristóbal Sifón, 2013-2024

---

*This package has been developed following my own interests and requirements and is made public here as-is. Some portions of it may be incomplete or (partially) untested, and I cannot guarantee that it will work in other operating systems or python versions*

---


## Changelog
* **v0.5.0:**
    - Updated Abell catalog to have richness class "Rich"
    - Created `ClusterCatalog` objects for direct import for all catalogs available by default
* **v0.4.0:**
    - `Catalog` class replaced by `clusters.ClusterCatalog` and `catalog.SourceCatalog`
* **v0.3.2:**
    - Bug fix when initializing ``Catalog`` object with hms coordinates
* **v0.3.0:**
    - Simpler ``Catalog`` API
    - Added Madcows catalog
    - Renamed ACT catalogs to ``act-dr4`` and ``act-dr5``
    - Removed Python 2 compatibility
* **v0.2.0:**
    - New ``footprint`` module
    - Added Abell, ACT, MCXC catalogs
* **v0.1.1**
