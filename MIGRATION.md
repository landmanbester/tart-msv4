# Migrating kremetart to tart-msv4

## Switch

1. Add `tart-msv4` to kremetart's dependencies. It needs only numpy, xarray, h5py and pyerfa.
2. Replace `read_hdf_as_msv4(path)` with `xr.open_datatree(path, engine="tart")` for one file, or with
   `tart_msv4.open_tart(paths)` for several files of one observation.
3. Delete `read_hdf_as_msv4`, `_geodetic_to_ecef` and `_enu_to_ecef`. tart-msv4 provides
   `tart_msv4.geometry.enu_to_itrs`, `itrs_to_enu`, `geodetic_to_itrs`, `zenith_radec` and `baseline_uvw`.

## Intended differences

These come from diffing both readers on `vis_2026-06-09_08_11_43.476804.hdf`. VISIBILITY, FLAG, WEIGHT, `time`
and ANTENNA_POSITION are identical (positions to 1e-9 m).

| What | kremetart `read_hdf_as_msv4` | tart-msv4 |
|---|---|---|
| Partition name | `partition_000` | `<first file stem>_partition_000` (`partition_name=` overrides it) |
| Fields | one field `"zenith"` for every snapshot | one field per snapshot, `zenith_<UTC ISO µs>` |
| `field_and_source_base_xds` | `PHASE_CENTER_ELAZ` [[90, 0]] on `elaz_label`, attr `phase_center_convention` | `FIELD_PHASE_CENTER_DIRECTION(field_name, sky_dir_label)` = ICRS (ra, dec) of the zenith per snapshot, `{type: sky_coord, units: rad, frame: fk5}` |
| `scan_name` | `"1"` | `scan_<first snapshot ISO>` per file, `scan_intents=["OBSERVE_TARGET#ON_SOURCE"]` |
| UVW frame | ENU baselines, `frame: "ENU-zenith"`, constant in time | fk5 (u, v, w) toward each snapshot's own zenith, varying in time. w = 0 to < 1 mm; (u, v) are ENU rotated by about 0.3° (precession) |
| UVW sign | pos(ant1) − pos(ant2) (never checked) | pos(ant1) − pos(ant2), **checked** against the tart2ms MS ([ADR 2](docs/adr/0002-uvw-sign-ant1-minus-ant2.md)) |
| VISIBILITY units | `"1"` | `"dimensionless"` (the spelling the MSv4 schema itself uses) plus a `description` |
| Required attrs | no `creation_date`, `observation_info` missing `observer`/`release_date`/`project_UID`, `frequency` missing `spectral_window_*`, `time`-like vars without attrs | all present; passes `xradio.schema.check.check_datatree` |
| `source_hdf` attr | present | dropped. Use `dt.encoding["source"]` (set by `xr.open_datatree`) |
| `antenna_xds.ANTENNA_POSITION_ENU` | present | **moved** to `gain_xds.ANTENNA_POSITION_ENU(gain_time, antenna_name, enu_label)`. MSv4 rejects the extra `enu_label` dim on `antenna_xds`, and TART recalibrates positions per file |
| Dish diameter | 0.0 | 0.025 m (as tart2ms writes it) |
| `gain_xds` | `type: tart_gain_snapshot` (fails `check_datatree`), dims `(antenna_name)`, attr `gain_time_unix` | `type: extension:gain.tart_msv4`, `tart_msv4_extension: True`, dims `(gain_time, antenna_name)` with one row per file. `gain_time` is the median snapshot time of each file |
| Multi-file | not supported (one DataTree per file) | `open_tart(paths)`: one partition, files lazily concatenated along `time`. Config and baselines must match exactly; positions must match within `position_tolerance` (1 cm). UVW uses each file's own positions |

## Call sites to update

- **`src/kremetart/utils/rephasing.py`**
  - `midpoint_zenith`: the zenith at `time[n//2]` is now simply
    `field_and_source_base_xds.FIELD_PHASE_CENTER_DIRECTION[n//2]` (fk5/ICRS, in radians). astropy is no
    longer needed.
  - `rephase_to_dir`, `_instantaneous_zenith`: the current phase centre per snapshot is
    `FIELD_PHASE_CENTER_DIRECTION` indexed by `field_name(time)`, and the current UVW is `UVW` as read.
    Drop the astropy recomputation of `s_old`/`uvw_cur`. Note that `_project_baselines` uses the ITRS pole
    as "north", which leaves out precession (about 1.8 cm on TART baselines). The `UVW` from tart-msv4 is
    correct in fk5.
  - `itrs_baselines`: with multi-file trees, `antenna_xds` holds only the **first** file's positions. Use
    `UVW` directly, or per-file positions from `tart_msv4.geometry.enu_to_itrs(gain_xds.ANTENNA_POSITION_ENU[i], ...)`.
  - `_rebuild`: keep writing the rephased result as a new field `"phasecenter"`. Ideally put it in a
    separate data group (e.g. `data_groups["rephased"]` with its own `field_and_source_rephased_xds`) instead
    of overwriting `base`.
  - `common_phase_direction`: read the first and last `FIELD_PHASE_CENTER_DIRECTION` from
    `open_tart(hdf_paths)`, or keep it as is. It only uses `observation_info`, which keeps the same
    `site_*` keys.
- **`src/kremetart/core/stream_msv4.py:41`** (and `operators/io.py`, `operators/dft_lm*.py`): `UVW` is now
  fk5 toward each snapshot's own zenith, not ENU. For a per-snapshot DFT toward zenith-referenced (l, m),
  that is the right quantity, but l and m are then (ra, dec) offsets of that snapshot's field, not ENU
  direction cosines. If the operators assumed u = east and v = north, expect a ~0.3° rotation, and |w| up
  to 0.3 mm instead of exactly 0. Note that the zarr written by `prepare_msv4_zarr` has no `UVW`.
- **`src/kremetart/utils/read_tart_hdf.py:492`** (`prepare_msv4_zarr`): replace the per-file
  `read_hdf_as_msv4` loop with `open_tart(paths)`, or keep it per file using `engine="tart"`.
  `equatorial_baselines`/`BORESIGHT` can use `FIELD_PHASE_CENTER_DIRECTION`.
- **`src/kremetart/utils/calibration.py:18`**, **`scripts/validate_tart_gains.py:56`**:
  `gain_xds.GAIN` is now `(gain_time, antenna_name)`. For one file use `GAIN.values[0]`. For a multi-file
  tree pick the row for each snapshot's scan (`gain_time` follows scan order).
- **`scripts/stefcal_calibrate.py:45,49`**: use `gain_xds.ANTENNA_POSITION_ENU[0]` instead of
  `antenna_xds.ANTENNA_POSITION_ENU`, and `gain_xds.ANTENNA_FLAG[0]` instead of `ANTENNA_FLAG`.
- **`src/kremetart/utils/satellites.py:67`**, **`scripts/compare_regularisers.py:55`**: only the
  open call and the partition name change (`observation_info` keys are kept). Find the partition with
  `next(iter(dt.children))`, not `"partition_000"`.
- **Tests**:
  - `tests/test_read_tart_hdf_msv4.py`: delete it. Its port lives in tart-msv4's `tests/test_reference_ms.py`.
  - `tests/test_rephasing.py`: rebase it on `engine="tart"`. Its |Δu|, |Δv| < 3 cm tolerance against tart2ms
    stays, because tart2ms places antennas on a sphere.
  - `tests/test_calibration.py`, `tests/test_stefcal_realdata.py`, `tests/test_satellites.py` and the
    `test_smoovie_*`/`test_accuracy_*` tests: switch the open call, and use `GAIN[0]` and
    `ANTENNA_POSITION_ENU` from `gain_xds`.
- **Antenna names** remain `ant00…ant23` (tart2ms/xarray-ms use `ANTENNA-0…`). Code matching on
  `baseline_antenna{1,2}_name` keeps working.
