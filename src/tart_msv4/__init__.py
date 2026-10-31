"""TART visibility HDF5 files as Measurement Set v4 xarray DataTrees.

>>> import xarray as xr
>>> dt = xr.open_datatree("vis_2026-06-09_08_11_43.476804.hdf", engine="tart")
"""

from typing import Any

__version__ = "0.1.0"

from tart_msv4.backend import open_tart  # noqa: E402  (backend reads __version__ lazily)

__all__ = ["__version__", "build_msv4_groups", "open_tart"]


def __getattr__(name: str) -> Any:
    # Lazy: xarray imports this package whenever it resolves any engine, and the schema pulls in pyerfa.
    if name == "build_msv4_groups":
        from tart_msv4.schema import build_msv4_groups

        return build_msv4_groups
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
