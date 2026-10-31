# Changelog

## 0.1.0

- First release. The `tart` xarray backend opens TART HDF5 files as MSv4 DataTrees: one field per snapshot,
  fk5 UVW toward each snapshot's zenith computed with pyerfa, and a `gain_xds` extension.
- `open_tart` opens several files of one observation as a single lazy partition.
