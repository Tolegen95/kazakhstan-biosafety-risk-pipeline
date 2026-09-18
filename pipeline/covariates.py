"""
Minimal raster covariate utilities backing the MaxEnt module: load a set of
single-band GeoTIFFs clipped to a bounding box, stack them into one array
sharing a common grid, and sample values at arbitrary (lon, lat) points or
draw random background points within the valid-data area.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import from_bounds


@dataclass
class CovariateStack:
    names: list[str]
    data: np.ndarray  # (n_bands, rows, cols), NaN where any band is nodata
    transform: rasterio.Affine
    crs: object

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1:]

    def rowcol(self, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = rasterio.transform.rowcol(self.transform, lons, lats)
        return np.asarray(rows), np.asarray(cols)

    def sample(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Nearest-cell covariate values at each point -> (n_points, n_bands), NaN if out of bounds/nodata."""
        rows, cols = self.rowcol(np.asarray(lons), np.asarray(lats))
        out = np.full((len(rows), len(self.names)), np.nan)
        n_rows, n_cols = self.shape
        valid = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        out[valid] = self.data[:, rows[valid], cols[valid]].T
        return out

    def valid_mask(self) -> np.ndarray:
        return ~np.isnan(self.data).any(axis=0)

    def random_background(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Uniform-random points among valid (non-nodata) cells -> (lons, lats)."""
        mask = self.valid_mask()
        row_idx, col_idx = np.nonzero(mask)
        if len(row_idx) == 0:
            raise ValueError("no valid (non-nodata) cells in the covariate stack")
        choice = rng.choice(len(row_idx), size=min(n, len(row_idx)), replace=len(row_idx) < n)
        rows, cols = row_idx[choice], col_idx[choice]
        xs, ys = rasterio.transform.xy(self.transform, rows, cols)
        return np.asarray(xs), np.asarray(ys)


def load_covariate_stack_from_file(path: str) -> CovariateStack:
    """Load a pre-clipped multi-band GeoTIFF (see etl/export_covariates.py)
    where each band's description is its covariate name -- no download or
    windowed clipping needed, just this one small file."""
    with rasterio.open(path) as src:
        names = [src.descriptions[i] or f"band{i+1}" for i in range(src.count)]
        data = src.read(masked=True).astype(np.float32).filled(np.nan)
        return CovariateStack(names=names, data=data, transform=src.transform, crs=src.crs)


def load_covariate_stack(paths: dict[str, str], bounds: tuple[float, float, float, float]) -> CovariateStack:
    """paths: {covariate_name: geotiff_path}. bounds: (min_lon, min_lat, max_lon, max_lat).
    All rasters must share the same CRS and resolution (true for a WorldClim bio+elev bundle)."""
    names = list(paths)
    bands = []
    transform = crs = None
    for name in names:
        with rasterio.open(paths[name]) as src:
            window = from_bounds(*bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            arr = src.read(1, window=window, masked=True).astype(np.float32).filled(np.nan)
            win_transform = src.window_transform(window)
            if transform is None:
                transform, crs = win_transform, src.crs
            elif arr.shape != bands[0].shape:
                raise ValueError(f"covariate {name!r} grid shape {arr.shape} does not match {names[0]!r} {bands[0].shape}")
            bands.append(arr)
    return CovariateStack(names=names, data=np.stack(bands), transform=transform, crs=crs)
