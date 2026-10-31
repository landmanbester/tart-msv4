"""The ``tart`` xarray backend: TART HDF5 files opened lazily as MSv4 DataTrees.

Module-level imports are limited to numpy, xarray and h5py, because xarray imports every registered backend
whenever it resolves any engine. The schema code (and pyerfa) is imported on first use.
"""

import json
import os
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint, CachingFileManager
from xarray.backends.locks import HDF5_LOCK

from tart_msv4.lazy import GridArray, lazy

PathLike = str | os.PathLike
_SIGNATURE = ("vis", "baselines", "config", "timestamp")
# xarray forwards its CF decoding options to every backend; TART data has nothing to decode.
_IGNORED_DECODERS = frozenset(
    {"mask_and_scale", "decode_times", "decode_timedelta", "use_cftime", "concat_characters", "decode_coords"}
)


def parse_timestamps(raw: Iterable[bytes | str]) -> np.ndarray:
    """Parse TART ISO 8601 timestamps to float64 unix seconds. Timezone-naive values are taken as UTC."""
    out = []
    for value in raw:
        stamp = datetime.fromisoformat(value.decode() if isinstance(value, bytes) else value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        out.append(stamp.timestamp())
    return np.array(out, dtype=np.float64)


def read_metadata(path: PathLike) -> dict[str, Any]:
    """Eagerly read the small arrays of a TART file (everything except ``vis``).

    The path is made absolute, so lazy reads still work after a ``chdir`` or on a worker.

    Raises:
        ValueError: If the shape of ``vis`` disagrees with ``timestamp`` and ``baselines``.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    with h5py.File(path, "r") as f:
        raw_config = f["config"][0]
        meta = {
            "path": path,
            "config": json.loads(raw_config.decode() if isinstance(raw_config, bytes) else raw_config),
            "time": parse_timestamps(f["timestamp"][:]),
            "antenna_positions": f["antenna_positions"][:].astype(np.float64),
            "baselines": f["baselines"][:].astype(np.int64),
            "gains": f["gains"][:].astype(np.float64),
            "phases": f["phases"][:].astype(np.float64),
            "vis_shape": f["vis"].shape,
        }
    expected = (meta["time"].size, len(meta["baselines"]))
    if tuple(meta["vis_shape"]) != expected:
        raise ValueError(f"{path}: vis has shape {meta['vis_shape']}, expected (n_timestamp, n_baseline) = {expected}")
    return meta


class TartVisibilityArray(GridArray):
    """Lazy ``(time, baseline_id, frequency, polarization)`` view over the ``vis`` datasets of TART files.

    Each file is opened through a :class:`~xarray.backends.CachingFileManager`, so the array pickles cleanly
    and every Dask or Ray worker reopens its own handles. Reads are serialised with ``HDF5_LOCK``.
    """

    def __init__(self, paths: Sequence[str], n_times: Sequence[int], n_baseline: int):
        self.managers = [CachingFileManager(h5py.File, p, mode="r") for p in paths]
        self.offsets = np.concatenate([[0], np.cumsum(n_times)])
        self.shape = (int(self.offsets[-1]), int(n_baseline), 1, 1)
        self.dtype = np.dtype(np.complex64)

    def _block(self, t: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.empty((t.size, b.size), dtype=self.dtype)
        if t.size and b.size:
            b_lo, b_hi = b.min(), b.max() + 1
            for i, manager in enumerate(self.managers):
                lo, hi = self.offsets[i], self.offsets[i + 1]
                sel = np.flatnonzero((t >= lo) & (t < hi))
                if not sel.size:
                    continue
                rows = t[sel] - lo
                # h5py reads one contiguous block; the outer selection is applied in memory.
                with HDF5_LOCK:
                    block = manager.acquire()["vis"][rows.min() : rows.max() + 1, b_lo:b_hi]
                out[sel] = _to_complex(block)[np.ix_(rows - rows.min(), b - b_lo)]
        return out[:, :, None, None]


def _to_complex(block: np.ndarray) -> np.ndarray:
    """Return complex64 values whether h5py presents ``vis`` as complex or as an ``(r, i)`` compound."""
    if block.dtype.names:
        return (block["r"] + 1j * block["i"]).astype(np.complex64)
    return block.astype(np.complex64, copy=False)


def open_tart_groups(
    paths: PathLike | Sequence[PathLike],
    *,
    drop_variables: str | Iterable[str] | None = None,
    partition_name: str | None = None,
    creation_date: str | None = None,
    dut1: float = 0.0,
    position_tolerance: float = 0.01,
) -> dict[str, xr.Dataset]:
    """Open one or more TART files of a single observation as a dict of MSv4 groups.

    Files are sorted by their first timestamp and must not overlap in time. They must share ``config`` and
    ``baselines`` exactly, while their ``antenna_positions`` may differ by up to ``position_tolerance``: TART
    recalibrates positions along with gains, so UVW is computed with each file's own positions. Arrays on the
    (time, baseline) grid stay lazy; the small per-file metadata is read eagerly.

    Args:
        paths: One file path or a sequence of file paths. Globs and directories are not expanded.
        drop_variables: Variable names to drop from every group.
        partition_name: Partition node name. Defaults to ``<stem of the first file>_partition_000``.
        creation_date: ISO creation date. Defaults to the last snapshot time (deterministic).
        dut1: UT1 - UTC in seconds.
        position_tolerance: Largest allowed antenna position difference between files, in metres.

    Returns:
        Mapping from DataTree group path to dataset.

    Raises:
        ValueError: If no paths are given, a file is malformed, files overlap in time (or repeat), or they do
            not share one configuration.
    """
    from tart_msv4.schema import build_msv4_groups  # deferred: imports pyerfa

    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    if not paths:
        raise ValueError("no TART files given")
    metas = sorted((read_metadata(p) for p in paths), key=lambda m: m["time"][0])
    _check_compatible(metas, position_tolerance)

    first = metas[0]
    for prev, meta in zip(metas, metas[1:]):
        if meta["time"][0] <= prev["time"][-1]:
            raise ValueError(f"TART files overlap in time (or repeat): {prev['path']} and {meta['path']}")

    vis = TartVisibilityArray([m["path"] for m in metas], [m["time"].size for m in metas], first["vis_shape"][1])
    name = partition_name or f"{Path(first['path']).stem}_partition_000"
    groups = build_msv4_groups(
        lazy(vis),
        np.concatenate([m["time"] for m in metas]),
        np.concatenate([np.full(m["time"].size, i) for i, m in enumerate(metas)]),
        first["config"],
        np.stack([m["antenna_positions"] for m in metas]),
        first["baselines"],
        np.stack([m["gains"] for m in metas]),
        np.stack([m["phases"] for m in metas]),
        partition_name=name,
        creation_date=creation_date,
        dut1=dut1,
    )
    if drop_variables is not None:
        drop = [drop_variables] if isinstance(drop_variables, str) else list(drop_variables)
        groups = {k: ds.drop_vars(drop, errors="ignore") for k, ds in groups.items()}
    return groups


def _check_compatible(metas: list[dict[str, Any]], position_tolerance: float) -> None:
    """Raise ``ValueError`` naming the first key and file that differ from the first file."""
    ref = metas[0]
    for meta in metas[1:]:
        a, b = ref["antenna_positions"], meta["antenna_positions"]
        checks = {
            "config": ref["config"] == meta["config"],
            "baselines": np.array_equal(ref["baselines"], meta["baselines"]),
            "antenna_positions": a.shape == b.shape and bool(np.all(np.abs(a - b) <= position_tolerance)),
        }
        for key, same in checks.items():
            if not same:
                raise ValueError(
                    f"TART files are not one observation: {key!r} of {meta['path']} differs from {ref['path']}"
                )


def open_tart(
    paths: PathLike | Sequence[PathLike], *, chunks: int | dict | str | None = None, **kwargs: Any
) -> xr.DataTree:
    """Open one or more TART files of a single observation as one MSv4 DataTree.

    Args:
        paths: One file path or a sequence of paths (expand globs and directories yourself, e.g. with
            ``sorted(Path(d).glob("*.hdf"))``).
        chunks: ``None`` (default) keeps arrays lazy without Dask. ``{}`` makes Dask chunks of one file
            each along ``time``. Any other value is passed to :meth:`xarray.Dataset.chunk` (dimensions
            missing from a group are ignored).
        **kwargs: Passed to :func:`open_tart_groups` (``partition_name``, ``creation_date``, ``dut1``,
            ``position_tolerance``, ``drop_variables``).

    Returns:
        The MSv4 DataTree.
    """
    groups = open_tart_groups(paths, **kwargs)
    if chunks is not None:
        groups = {k: _chunk(ds, chunks) for k, ds in groups.items()}
    return xr.DataTree.from_dict(groups)


def _chunk(ds: xr.Dataset, chunks: int | dict | str) -> xr.Dataset:
    """Chunk a group; ``{}`` means one ``time`` chunk per file (scan) in the visibility partition."""
    if not ds.data_vars:
        return ds
    if isinstance(chunks, dict):
        if not chunks and "scan_name" in ds.coords:
            scans = ds["scan_name"].values
            starts = np.flatnonzero(np.r_[True, scans[1:] != scans[:-1]])
            chunks = {"time": tuple(np.diff(np.r_[starts, scans.size]))}
        chunks = {k: v for k, v in chunks.items() if k in ds.dims}
    return ds.chunk(chunks)


class TartBackendEntrypoint(BackendEntrypoint):
    """xarray backend for TART visibility HDF5 files (``engine="tart"``)."""

    description = "Open TART visibility HDF5 files as Measurement Set v4 DataTrees"
    url = "https://github.com/landmanbester/tart-msv4"
    open_dataset_parameters = ("filename_or_obj", "drop_variables", "partition_name", "creation_date", "dut1")

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        """Return True only for an HDF5 file carrying the TART signature. Never raises."""
        try:
            if not isinstance(filename_or_obj, (str, os.PathLike)) or not os.path.isfile(filename_or_obj):
                return False
            if not h5py.is_hdf5(filename_or_obj):
                return False
            with h5py.File(filename_or_obj, "r") as f:
                if not all(k in f for k in _SIGNATURE):
                    return False
                raw = f["config"][0]
                return "num_antenna" in json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            return False

    def open_groups_as_dict(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        partition_name: str | None = None,
        creation_date: str | None = None,
        dut1: float = 0.0,
        **decoders: Any,
    ) -> dict[str, xr.Dataset]:
        """Open a single TART file as a dict of MSv4 groups (see :func:`open_tart_groups`).

        xarray's CF decoding options (``decode_times`` etc.) are accepted and ignored.
        """
        unknown = set(decoders) - _IGNORED_DECODERS
        if unknown:
            raise TypeError(f"engine='tart' got unexpected keyword arguments {sorted(unknown)}")
        return open_tart_groups(
            _single_path(filename_or_obj),
            drop_variables=drop_variables,
            partition_name=partition_name,
            creation_date=creation_date,
            dut1=dut1,
        )

    def open_datatree(self, filename_or_obj: Any, **kwargs: Any) -> xr.DataTree:
        """Open a single TART file as an MSv4 DataTree."""
        return xr.DataTree.from_dict(self.open_groups_as_dict(filename_or_obj, **kwargs))

    def open_dataset(self, filename_or_obj: Any, **kwargs: Any) -> xr.Dataset:
        """Open the visibility partition dataset of a single TART file."""
        groups = self.open_groups_as_dict(filename_or_obj, **kwargs)
        return next(ds for ds in groups.values() if ds.attrs.get("type") == "visibility")


def _single_path(filename_or_obj: Any) -> str:
    """Validate that the backend was given one existing file (directories and globs are rejected)."""
    if not isinstance(filename_or_obj, (str, os.PathLike)):
        raise ValueError(f"engine='tart' needs a file path, got {type(filename_or_obj).__name__}")
    path = os.path.expanduser(os.fspath(filename_or_obj))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"not a file: {path!r} (use tart_msv4.open_tart for several files)")
    return path
