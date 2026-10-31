# MSv4 and pyerfa notes

Facts about xradio's MSv4 schema checker and pyerfa that are not obvious from their docs. They were checked
against xradio 1.3.0 and pyerfa 2.0.1.5.

## xradio schema (`xradio/measurement_set/schema.py`)

- **Optional means `| None`.** A field is optional only if its type includes `| None`. A default value
  (`= None`, `= "Jy"`) does not make it optional. FLAG, WEIGHT and UVW are required.
- **Register before checking.** `check_datatree` reports "Unknown dataset type!" for every node unless
  `xradio.measurement_set` has been imported. Import it explicitly in tests.
- **Matched by `type`, not by name.** The checker visits every node with data and looks up its schema by
  `attrs["type"]`. Node names and tree shape are not checked, and neither is `data_groups.field_and_source`.
- **Extension nodes** must use `type="extension:<name>.<namespace>"`, matching
  `^extension:[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$`. Those nodes are skipped with an `ExtensionTypeWarning`. Any
  other unknown type is an error.
- **Dims are exact.** A node's dims must match a schema dim set. Extra data variables on existing dims pass;
  a variable that adds a new dim fails. That is why `ANTENNA_POSITION_ENU` lives in `gain_xds`.
- **Units are free strings in places.** VISIBILITY `units` is `Attr[str]` (default "Jy"), so any string passes.
  The schema spells unitless quantities `"dimensionless"`. Direction `units` must be a single string (`"rad"`);
  a list like `["rad", "rad"]` fails.
- **String formats aren't validated.** `creation_date`, `data_groups.*.date` and
  `observation_info.release_date` are only type-checked as strings.
- **Field coordinates.** A time-indexed field coordinate needs `type="field_and_source_ephemeris"`. The
  `field_and_source` type requires the `field_name` dim.

## xarray-ms conventions (0.5.13)

- Partitions are named `<ms stem>_partition_NNN`, and multiple fields share one partition (FIELD_ID is not a
  partition key).
- UVW and FIELD_PHASE_CENTER_DIRECTION carry `frame="fk5"` for J2000 MSs.
- VISIBILITY always has `units="Jy"`, and dates are `datetime.now()`.

## pyerfa

- **`atoi13` returns CIRS, not ICRS.** The observed-to-ICRS function is `atoc13`; using `atoi13` puts the
  zenith about 230″ off.
- **Leap-second days.** Build two-part UTC dates with `dtf2d` from calendar fields; seconds-of-day / 86400 is
  wrong on those days.
- **Unpacking `d2dtf`.** It returns `(iy, im, id, ihmsf)`.

## xarray backends

- With an explicit `engine`, `xr.open_datatree` passes `filename_or_obj` through unchanged.
- xarray forwards its CF decoding keywords (`decode_times`, ...) to every backend. They are not filtered by
  `open_dataset_parameters`.
- In engine guessing, the netCDF engines go first; h5netcdf claims every HDF5 file.
- `preferred_chunks` in an encoding means the on-disk chunk layout. Setting it for contiguous data causes
  spurious "chunks separate the stored chunks" warnings.
