"""Lazy arrays on the ``(time, baseline_id, ...)`` grid, read through xarray's outer indexing.

Nothing that scales with ``n_time * n_baseline`` is built when a file is opened: VISIBILITY is read from HDF5,
and the derived arrays (FLAG, WEIGHT, UVW, ...) are computed on read from small per-file and per-snapshot
tables. Imports only numpy and xarray, because ``backend.py`` imports this module.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
from xarray.backends import BackendArray
from xarray.core import indexing


class GridArray(BackendArray):
    """Base for lazy ``(time, baseline_id, *trailing)`` arrays.

    Subclasses implement :meth:`_block`, which returns the full trailing axes for the selected time and
    baseline indices. This class turns any outer-indexing key (ints, slices, unsorted or repeated index
    arrays) into those index arrays, then applies the trailing keys.
    """

    shape: tuple[int, ...]
    dtype: np.dtype

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER, self._raw_getitem)

    def _raw_getitem(self, key: tuple) -> np.ndarray:
        t = np.arange(self.shape[0])[key[0]]
        b = np.arange(self.shape[1])[key[1]]
        out = self._block(np.atleast_1d(t), np.atleast_1d(b))
        # Trailing keys go last-axis-first, so an integer key never shifts the axes still to be indexed.
        for axis in range(len(key) - 1, 1, -1):
            out = out[(slice(None),) * axis + (key[axis],)]
        if np.ndim(b) == 0:
            out = out[:, 0]
        if np.ndim(t) == 0:
            out = out[0]
        return out

    def _block(self, t: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the ``(t.size, b.size, *trailing)`` block for 1-d time and baseline index arrays."""
        raise NotImplementedError


class DerivedArray(GridArray):
    """A grid array computed on read by ``fn(t, b)``.

    ``fn`` must be picklable (e.g. a ``functools.partial`` of a module-level function over small arrays), so
    that Dask and Ray can ship the array to workers.
    """

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray], shape: tuple[int, ...], dtype: type):
        self.fn = fn
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)

    def _block(self, t: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.fn(t, b)


def table_rows(table: np.ndarray, row_index: np.ndarray, t: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Expand a small table to the grid: snapshot ``t`` takes row ``row_index[t]`` of ``table``.

    ``table`` has shape ``(n_rows, n_baseline or 1, *trailing)``. A size-1 baseline axis is broadcast.
    """
    rows = table[row_index[t]]
    return rows[:, b] if rows.shape[1] > 1 else np.repeat(rows, b.size, axis=1)


def expand_table(table: np.ndarray, row_index: np.ndarray, shape: tuple[int, ...]) -> indexing.LazilyIndexedArray:
    """Return a lazy grid array of ``shape`` that expands ``table`` with :func:`table_rows` on read."""
    return lazy(DerivedArray(partial(table_rows, table, row_index), shape, table.dtype))


def lazy(array: GridArray) -> indexing.LazilyIndexedArray:
    """Wrap a grid array so that xarray indexes it lazily."""
    return indexing.LazilyIndexedArray(array)
