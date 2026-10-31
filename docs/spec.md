# tart-msv4 spec

The requirements reviewers check tart-msv4 against. The original brief comes first; **Settled decisions** records
where the maintainer changed or refined it, and overrides the brief. Change this file when a decision changes.

## Purpose

A small, standalone xarray backend (`engine="tart"`) exposing TART visibility HDF5 files as MSv4
`xarray.DataTree`s. It is a library: no CLI, no container. Priorities: small, simple, lightweight, correct.

## Requirements (original brief)

- **Dependencies.** Runtime is only numpy, xarray>=2025.1, h5py and pyerfa: no astropy, netCDF4, dask or network.
  Dask must work when installed (`chunks=`) but must not be required. Python >=3.11, uv_build, ruff (line length
  120, rules E,F,I,N,W), MIT licence.
- **Layout.** Flat: `__init__.py` (public API `open_tart`, `build_msv4_groups`, `__version__`), `backend.py`,
  `schema.py` (pure functions mapping raw arrays to MSv4 groups), `geometry.py` (pure, pyerfa), and `lazy.py`
  (the outer-indexed lazy grid arrays; numpy and xarray only). Classes only where
  xarray requires them. Type hints and Google docstrings. **Module-level imports of `backend.py` are limited to
  numpy, xarray and h5py**, because xarray imports every backend whenever it resolves an engine. Any module over
  ~300 lines must be simplified.
- **Entry point.** `TartBackendEntrypoint` with `open_groups_as_dict` (the core), `open_datatree`, `open_dataset`
  (main visibility partition only), `description`, `url` and `open_dataset_parameters`. `guess_can_open` is
  conservative: True only for HDF5 files carrying `vis`, `baselines` and a JSON `config` with `num_antenna`. It
  never claims generic .h5/.hdf files and never raises.
- **Schema authority.** The output must pass `xradio.schema.check.check_datatree` with no issues, and where kremetart
  disagrees, the MSv4 schema (xradio `measurement_set/schema.py`) wins.
- **Fields.** One field per timestamp (snapshot), phased to the instantaneous zenith, all in ONE partition.
  `field_name(time)` names are distinct, deterministic and derived from the timestamp (unique across concatenated
  files). `scan_name(time)` is one scan per file. `field_and_source_base_xds` is indexed by `field_name`
  (size n_time), with `source_name`, `sky_dir_label=[ra, dec]` and `FIELD_PHASE_CENTER_DIRECTION` = the (ra, dec)
  of the site zenith per snapshot. `data_groups["base"]` carries every `DataGroupDict` key.
- **UVW.** `(time, baseline_id, uvw_label)` in m, in the frame of the phase-centre direction, toward each snapshot's
  own phase centre: rotate the ITRS baseline into the celestial frame at time t (Earth rotation +
  precession/nutation) and project onto the (u, v, w) basis of (ra_t, dec_t). Visibilities need no rephasing.
- **Ephemeris.** pyerfa only. Without IERS data, UT1 ≈ UTC is acceptable and documented, with an optional `dut1`
  kwarg. Target agreement with tart2ms/casacore is < 1 arcsec on the phase centre and < 1 mm on UVW; if that is
  unreachable, the reason must be documented.
- **Conventions.** VISIBILITY complex64, WEIGHT float32, FLAG uint8, plus TIME_CENTROID and
  EFFECTIVE_INTEGRATION_TIME, on (time, baseline_id, frequency, polarization). `time` is float64 unix UTC;
  timezone-naive timestamps are UTC, never machine-local. Polarization is ["RR"], with a single L1 channel and
  channel_width = bandwidth. `antenna_xds.ANTENNA_POSITION` is geocentric ITRS. A dead antenna (gain amplitude 0)
  flags every baseline it touches and zeroes its weight. `observation_info`, `processor_info` and `creator` follow
  the schema.
- **Extensions.** Only what consumers need and the schema can't express: `gain_xds`, marked
  `tart_msv4_extension: True`. It must not break `check_datatree`. `PHASE_CENTER_ELAZ` is dropped.
- **Laziness.** VISIBILITY (anything that scales as n_time*n_baseline) is a `BackendArray` wrapped in
  `LazilyIndexedArray`, indexed via `explicit_indexing_adapter` with `IndexingSupport.OUTER`, with complex values
  rebuilt from h5py storage on read. Files are opened with `CachingFileManager(h5py.File, path, mode="r")` and
  reads are guarded by `HDF5_LOCK`, so arrays pickle cleanly and each Dask/Ray worker reopens its own handle.
  Broadcast-only arrays may be eager, but more than one full (time, baseline) copy must not be materialised.
  Small metadata is eager.
- **Multiple files.** `open_tart(paths)` sorts by timestamp, checks that the files share one configuration (raising
  a clear error otherwise) and concatenates per-file lazy arrays along time, with each file its own lazy unit.
  `gain_xds` gets a time-like dimension.
- **Out of scope.** Rephasing to a common centre, applying gains, satellite catalogues, IERS UT1, writing HDF5.
- **Tests.** A synthetic h5py fixture copying the real layout. Backend tests cover engine listing, agreement of
  the open methods, guess_can_open, laziness (opening does not read vis), outer indexing, pickling,
  `chunks={"time": 1}` giving dask, and multi-file behaviour. Schema tests cover `check_datatree`, one field per
  snapshot, the phase centre being the zenith (elevation 90° within 1"), w ≈ ENU up within ~1 mm,
  u²+v² ≈ e²+n² within 1 mm, dead antennas, ITRS sanity, naive timestamps, dtypes and extension markers.
  Reference-MS tests compare values against tart2ms/xarray-ms and skip if the data is missing
  (`TART_MSV4_OFFLINE=1`).
- **Tooling.** CI runs ruff format check, ruff check, and pytest on 3.11–3.13 via uv. Publishing uses PyPI trusted
  publishing on `v*` tags. The README covers usage, multi-file use, the schema table with extensions, conventions
  and out-of-scope items. Version 0.1.0.

## Settled decisions (override the brief)

1. **Conventions.** Follow xarray-ms where it has one: field-direction and UVW frame "fk5", partition name
   `<first file stem>_partition_000` (overridable), data_groups `field_and_source` path
   `<partition>/field_and_source_base_xds`, `schema_version` "4.0.0", and xarray-ms's ANTENNA_POSITION attrs. No
   SIGMA_SPECTRUM. Deliberate deviations:
   - VISIBILITY `units="dimensionless"`, because the schema's units are a free string and "dimensionless" is the
     schema's own unitless spelling.
   - `creation_date` and data-group `date` are deterministic: the last snapshot time, with a `creation_date` kwarg
     override (ADR 3).
2. **UVW sign is pos(ant1) − pos(ant2)**, verified empirically against the tart2ms MS. Visibilities are
   unconjugated. This overrides the brief's ant2 − ant1 (ADR 2).
3. **Exact WGS84 tangent-plane geometry.** tart2ms places antennas on a sphere, so u/v agree with it only to
   ~3 cm. w must agree < 1 mm, and sub-mm u/v is checked against an independent astropy computation (astropy is
   test-only).
4. **Names.** field_name `zenith_<UTC ISO µs>`, source_name = field_name, no SOURCE_DIRECTION, scan_name
   `scan_<first snapshot ISO>` with scan_intents ["OBSERVE_TARGET#ON_SOURCE"], antennas `ant00…`.
5. **observation_info** = observer ["unknown"], project_UID "TART", release_date = creation date, plus
   telescope_name and site_* keys. processor_info = {CORRELATOR, TART}.
6. **Extension.** `gain_xds` has `type="extension:gain.tart_msv4"` (required, or check_datatree errors) and
   always has a `gain_time` dim (size 1 for one file). It holds GAIN, GAIN_AMPLITUDE, GAIN_PHASE, ANTENNA_FLAG and
   per-file ANTENNA_POSITION_ENU. ANTENNA_POSITION_ENU is not on antenna_xds, because the new dim fails the
   schema.
7. **Dead antennas are judged per file**, so FLAG/WEIGHT may change at file boundaries. WEIGHT is 1.0 when live.
8. **Time.** `time` = HDF timestamp exactly, TIME_CENTROID = time, integration time = median within-file spacing.
   Real files carry "+00:00"; naive timestamps are UTC.
9. **Multi-file compatibility.** config and baselines must be identical. antenna_positions may differ up to
   `position_tolerance` (default 1 cm), because TART recalibrates positions with gains. UVW uses each file's own
   positions, and antenna_xds holds the first file's.
10. **Inputs.** `xr.open_datatree(path, engine="tart")` takes a single file only (no directories, no globs).
    `open_tart` takes a path or a sequence of paths, and the caller expands globs.
11. **No new tart2ms reference MS.** The existing `_nocal.ms` (rephased to the zenith at `time[n//2]`) is compared
    at that snapshot.
12. **Tooling.** pre-commit (ruff, standard hooks, conventional commits). The changelog is generated by git-cliff
    via tbump at release only.
13. **Docs.** MIGRATION.md tells kremetart how to switch (call sites and intended differences).
14. **Every (time, baseline) array is lazy.** This goes further than the brief's "broadcast-only arrays may be
    eager". FLAG, WEIGHT, UVW, TIME_CENTROID and EFFECTIVE_INTEGRATION_TIME are `lazy.DerivedArray`s computed on
    read from small per-file and per-snapshot tables. Opening a file and pickling the tree cost O(n_time), never
    O(n_time * n_baseline).
15. **Files must not overlap.** `open_tart` raises when a file's first snapshot is not after the previous
    file's last one. This also catches the same file passed twice.
16. **Robustness.** Paths are made absolute at open. A `vis` shape that disagrees with `timestamp`/`baselines`
    raises. A non-finite gain marks a dead antenna. xarray's CF decoding keywords are accepted and ignored.
    `engine="tart"` must be passed explicitly, because netCDF engines claim HDF5 files first.
17. **Time conversion.** The ERFA two-part UTC date is built from calendar fields (`erfa.dtf2d`), which is
    correct on leap-second days.
