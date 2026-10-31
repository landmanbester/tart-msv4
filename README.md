# tart-msv4

An [xarray](https://xarray.dev) backend that opens [TART](https://tart.elec.ac.nz) (Transient Array Radio
Telescope) visibility HDF5 files as Measurement Set v4 (MSv4) `xarray.DataTree`s. Its output passes
`xradio.schema.check.check_datatree`, and it follows the conventions of
[xarray-ms](https://github.com/ratt-ru/xarray-ms).

The only runtime dependencies are numpy, xarray, h5py and pyerfa. It needs no astropy, no network and no
dask (dask is supported when installed).

## Install

```bash
pip install tart-msv4
```

## Usage

```python
import xarray as xr

dt = xr.open_datatree("vis_2026-06-09_08_11_43.476804.hdf", engine="tart")
```

`xr.open_dataset(path, engine="tart")` returns just the visibility partition. `engine="tart"` accepts a
single file only. Always pass `engine="tart"`. The backend's guessing claims only HDF5 files that carry the TART
signature (`vis`, `baselines`, `timestamp`, and a JSON `config` with `num_antenna`), but xarray tries its
netCDF engines first, and h5netcdf or netCDF4, when installed, claim every HDF5 file.

Every array on the (time, baseline) grid is lazy: VISIBILITY is read from the file, and FLAG, WEIGHT, UVW,
TIME_CENTROID and EFFECTIVE_INTEGRATION_TIME are computed on read from small per-file tables. Opening a file
reads only its metadata, and the opened tree pickles small. xarray's CF decoding keywords (`decode_times`
etc.) are accepted and ignored.

### Several files

TART writes one file per minute or so. To open the files of one observation as a single partition, lazily
concatenated along `time`:

```python
from pathlib import Path
import tart_msv4

paths = sorted(Path("data").glob("vis_*.hdf"))
dt = tart_msv4.open_tart(paths)  # lazy, no dask
dt = tart_msv4.open_tart(paths, chunks={})  # dask, one chunk per file
```

- **Inputs:** `open_tart` takes one path or a sequence of paths. You expand globs and directories
  yourself.
- **Ordering:** files are sorted by timestamp. Files that overlap in time, or repeat, raise `ValueError`.
- **Compatibility:** `config` and `baselines` must be identical. `antenna_positions` may differ by up to
  `position_tolerance` (default 1 cm), because TART recalibrates them along with its gains. Any other
  mismatch raises `ValueError`.

### Keyword arguments

The keywords below work with both `xr.open_datatree` and `open_tart`:

| kwarg | default | meaning |
|---|---|---|
| `partition_name` | `<first file stem>_partition_000` | partition node name |
| `creation_date` | last snapshot time (UTC ISO) | `creation_date` and data-group `date`. The default is deterministic ([ADR 3](docs/adr/0003-deterministic-dates.md)) |
| `dut1` | `0.0` | UT1 − UTC in seconds (see Conventions) |
| `drop_variables` | `None` | variables to drop from every group |
| `position_tolerance` | `0.01` | `open_tart` only: allowed antenna-position change between files (m) |

## Schema

```
/
└── <partition>                     visibility       (time, baseline_id, frequency, polarization, uvw_label)
    ├── antenna_xds                 antenna          (antenna_name, cartesian_pos_label, receptor_label)
    ├── field_and_source_base_xds   field_and_source (field_name, sky_dir_label)
    └── gain_xds                    TART extension   (gain_time, antenna_name, enu_label)
```

| Node | Variable | Notes |
|---|---|---|
| partition | `VISIBILITY` | complex64, `units="dimensionless"` (normalised correlation coefficients, uncalibrated). Lazy |
| | `FLAG`, `WEIGHT` | uint8 / float32. Baselines of a dead antenna (gain 0 in that file) have FLAG=1, WEIGHT=0. Otherwise 0 / 1 |
| | `UVW` | float64 m, `frame="fk5"`, toward each snapshot's own zenith |
| | `TIME_CENTROID`, `EFFECTIVE_INTEGRATION_TIME` | equal to `time` and the median snapshot spacing |
| | coords | `time` (unix s, UTC), `baseline_id`, `baseline_antenna{1,2}_name` (`ant00…`), `frequency` (one L1 channel, `channel_width` = bandwidth), `polarization` = `["RR"]`, `field_name(time)`, `scan_name(time)` |
| antenna_xds | `ANTENNA_POSITION` | geocentric ITRS (m), from the first file's ENU positions on the exact WGS84 tangent plane |
| | `ANTENNA_DISH_DIAMETER`, `ANTENNA_EFFECTIVE_DISH_DIAMETER`, `ANTENNA_RECEPTOR_ANGLE` | 0.025 m, 0.025 m, 0 rad |
| field_and_source_base_xds | `FIELD_PHASE_CENTER_DIRECTION` | ICRS (ra, dec) in rad of the zenith at each snapshot, one field per snapshot |
| gain_xds (**extension**) | `GAIN`, `GAIN_AMPLITUDE`, `GAIN_PHASE`, `ANTENNA_FLAG` | TART's per-file gain snapshot, **not** applied to VISIBILITY |
| | `ANTENNA_POSITION_ENU` | each file's East/North/Up positions (m) |

`gain_xds` has `type="extension:gain.tart_msv4"` and `tart_msv4_extension=True`. `check_datatree` skips it
with an `ExtensionTypeWarning`.

## Conventions

- **Fields.** TART points at the zenith and does not fringe-stop, so every snapshot is its own field. Fields
  are named `zenith_<UTC ISO µs>` (e.g. `zenith_2026-06-09T08:10:42.993515`), which keeps names unique
  across files. All fields share one partition, as in xradio and xarray-ms. Each file is one scan
  (`scan_<first snapshot ISO>`). See [ADR 1](docs/adr/0001-one-field-per-snapshot.md).
- **Visibilities need no rephasing.** They are already phased to each snapshot's zenith. They are passed
  through unconjugated, and are identical to the tart2ms MS DATA at a shared phase centre.
- **UVW = pos(ant1) − pos(ant2)**, the sign tart2ms writes. It is checked against the reference MS
  ([ADR 2](docs/adr/0002-uvw-sign-ant1-minus-ant2.md)). Each ITRS baseline is rotated into the celestial
  frame at its snapshot (IAU 2006/2000A, `erfa.c2t06a`) and projected onto the (u, v, w) basis of that
  snapshot's phase centre.
- **Phase centres** come from `erfa.atoc13`, which converts observed (az, zd) = (0, 0) to ICRS with
  aberration and no refraction. They match astropy to under 0.5″; the difference is polar motion, which is
  neglected. They match casacore/tart2ms to about 0.25″.
- **Time.** Timestamps are parsed as ISO 8601. Timestamps without a timezone are UTC, never machine-local.
  `time` is the TART timestamp exactly, as tart2ms writes it.
- **UT1.** No IERS data is used, so by default UT1 = UTC. An error of Δ seconds shifts right ascensions by
  about 15Δ″ (|Δ| < 0.9 s), which is negligible against TART's ~7° resolution. Pass `dut1=` for better.
- **Antenna positions.** ENU offsets are placed on the exact WGS84 tangent plane. tart2ms uses a spherical
  approximation, so its u and v differ by up to about 2 cm.

## Out of scope (TODO)

- Rephasing all snapshots to one common phase centre. That is a processing step and belongs in the consumer
  (e.g. kremetart). Its result should be a separate `field_and_source` data group.
- Applying gains.
- Satellite catalogues.
- IERS-based UT1 and polar motion.
- Writing back to HDF5.

## Development

```bash
uv sync                       # dev + test groups
uv run ruff format . && uv run ruff check .
uv run pytest                 # downloads the real-data bundle once into tests/data
```

The real-data bundle is the kremetart test bundle (nine TART files and tart2ms reference MSs). Set
`TART_MSV4_OFFLINE=1` to skip the download, or `TART_MSV4_DATA=<dir>` to use an existing copy. Tests marked
`integration` skip without it. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).

### Pre-commit hooks

Install the hooks once per clone:

```bash
uv sync
uv run pre-commit install    # installs the pre-commit and commit-msg hooks
```

| Stage | Hook | What it does |
|---|---|---|
| commit | ruff, pre-commit-hooks | `ruff check --fix`, `ruff format`, whitespace/EOF/YAML/TOML/large-file checks |
| commit message | conventional-pre-commit | requires a [Conventional Commit](https://www.conventionalcommits.org/) message: `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `deps:`, `chore:`, `ci:`, `style:`, `test:` or `build:` |

If a hook modifies files, the commit is aborted so you can review the changes. Run
`git add -u && git commit` again. Run all hooks on demand with `uv run pre-commit run --all-files`.

### Releasing

Commit messages are the changelog. `CHANGELOG.md` is generated from them by [git-cliff](https://git-cliff.org)
(`cliff.toml`) at release time, so don't edit it by hand. Releases use [tbump](https://github.com/your-tools/tbump):

```bash
uv run tbump 0.2.0
```

tbump updates the version in `pyproject.toml` and `src/tart_msv4/__init__.py`, generates `CHANGELOG.md`,
refreshes `uv.lock`, commits `chore(release): bump version to 0.2.0`, tags `v0.2.0` and pushes. Pushing the
tag publishes the package to PyPI.
