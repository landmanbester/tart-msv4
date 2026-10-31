"""Pure functions turning raw TART arrays into MSv4 datasets.

The output conforms to the xradio MSv4 schema (``xradio.schema.check.check_datatree``). Its attribute
conventions follow xarray-ms, except that VISIBILITY carries ``units="dimensionless"`` (TART visibilities are
normalised correlation coefficients, not Jy) and the dates are deterministic (see ``docs/adr``).
"""

from datetime import datetime, timezone
from typing import Any

import numpy as np
import xarray as xr

from tart_msv4 import geometry

TELESCOPE_NAME = "TART"
SCHEMA_VERSION = "4.0.0"
DISH_DIAMETER_M = 0.025  # TART patch antenna, as written by tart2ms
_TIME_ATTRS = {"type": "time", "units": "s", "format": "unix", "scale": "utc"}
_DIRECTION_ATTRS = {"type": "sky_coord", "units": "rad", "frame": "fk5"}


def unix_to_iso(time: float) -> str:
    """Format unix seconds (UTC) as a timezone-naive ISO 8601 string with microseconds."""
    return datetime.fromtimestamp(float(time), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def field_names(time: np.ndarray) -> np.ndarray:
    """Return the deterministic per-snapshot field names, e.g. ``zenith_2026-06-09T08:10:42.993515``."""
    return np.array([f"zenith_{unix_to_iso(t)}" for t in time])


def antenna_names(n_antenna: int) -> np.ndarray:
    """Return the zero-padded antenna names ``ant00, ant01, ...``."""
    return np.array([f"ant{i:02d}" for i in range(n_antenna)])


def integration_time(time: np.ndarray, scan_index: np.ndarray) -> float:
    """Return the median snapshot spacing in seconds, ignoring the gaps between files."""
    same_scan = np.diff(scan_index) == 0
    spacing = np.diff(time)[same_scan]
    return float(np.median(spacing)) if spacing.size else 1.0


def build_msv4_groups(
    vis: Any,
    time: np.ndarray,
    scan_index: np.ndarray,
    config: dict[str, Any],
    antenna_enu: np.ndarray,
    baselines: np.ndarray,
    gains: np.ndarray,
    phases: np.ndarray,
    *,
    partition_name: str = "partition_000",
    creation_date: str | None = None,
    dut1: float = 0.0,
    software_version: str = "",
) -> dict[str, xr.Dataset]:
    """Assemble the MSv4 groups of one observation.

    Args:
        vis: ``(n_time, n_baseline, 1, 1)`` complex visibilities. This may be a lazy array.
        time: ``(n_time,)`` snapshot times as unix seconds (UTC), sorted.
        scan_index: ``(n_time,)`` index of the file (scan) each snapshot came from, ``0..n_scan-1``.
        config: The TART ``config`` JSON (needs ``name, frequency, bandwidth, lat, lon, alt, num_antenna``).
        antenna_enu: ``(n_scan, n_antenna, 3)`` East/North/Up antenna positions in metres, per file. UVW uses
            each file's own positions; ``antenna_xds`` holds the first file's.
        baselines: ``(n_baseline, 2)`` antenna index pairs.
        gains: ``(n_scan, n_antenna)`` gain amplitudes; zero marks a dead antenna.
        phases: ``(n_scan, n_antenna)`` gain phases in radians.
        partition_name: Name of the partition node.
        creation_date: ISO creation date. Defaults to the last snapshot time, so the output is deterministic.
        dut1: UT1 - UTC in seconds, used for the phase centres and UVW.
        software_version: Version string recorded in ``creator``.

    Returns:
        Mapping from DataTree group path to dataset: the root, the visibility partition, and its
        ``antenna_xds``, ``field_and_source_base_xds`` and ``gain_xds`` (TART extension) children.
    """
    time = np.asarray(time, dtype=np.float64)
    scan_index = np.asarray(scan_index, dtype=np.int64)
    lat, lon, alt = float(config["lat"]), float(config["lon"]), float(config["alt"])
    names = antenna_names(int(config["num_antenna"]))
    date = creation_date or unix_to_iso(time[-1])

    antenna_enu = np.asarray(antenna_enu, dtype=np.float64)
    ra, dec = geometry.zenith_radec(time, lat, lon, alt, dut1)
    uvw = np.empty((time.size, baselines.shape[0], 3))
    for s, enu in enumerate(antenna_enu):
        itrs = geometry.enu_to_itrs(enu, lat, lon, alt)
        sel = scan_index == s
        uvw[sel] = geometry.baseline_uvw(
            itrs[baselines[:, 0]] - itrs[baselines[:, 1]], time[sel], ra[sel], dec[sel], dut1
        )

    p = f"/{partition_name}"
    return {
        "/": xr.Dataset(),
        p: visibility_xds(
            vis, time, scan_index, uvw, baselines, gains, names, config, partition_name, date, dut1, software_version
        ),
        f"{p}/antenna_xds": antenna_xds(geometry.enu_to_itrs(antenna_enu[0], lat, lon, alt), names, config["name"]),
        f"{p}/field_and_source_base_xds": field_and_source_xds(field_names(time), ra, dec),
        f"{p}/gain_xds": gain_xds(gains, phases, antenna_enu, time, scan_index, names),
    }


def visibility_xds(
    vis: Any,
    time: np.ndarray,
    scan_index: np.ndarray,
    uvw: np.ndarray,
    baselines: np.ndarray,
    gains: np.ndarray,
    names: np.ndarray,
    config: dict[str, Any],
    partition_name: str,
    date: str,
    dut1: float,
    software_version: str,
) -> xr.Dataset:
    """Build the main visibility dataset of a partition (see :func:`build_msv4_groups` for the arguments)."""
    n_time, n_bl = time.size, baselines.shape[0]
    dims4 = ("time", "baseline_id", "frequency", "polarization")
    t_int = integration_time(time, scan_index)

    # Dead antennas are judged per file; one (n_scan, n_baseline) table, expanded once to (time, baseline).
    dead = np.asarray(gains) == 0
    bad = (dead[:, baselines[:, 0]] | dead[:, baselines[:, 1]])[scan_index]
    flag = bad.astype(np.uint8)[:, :, None, None]
    weight = np.where(bad, 0.0, 1.0).astype(np.float32)[:, :, None, None]
    time_grid = np.broadcast_to(time[:, None], (n_time, n_bl))
    t_int_grid = np.broadcast_to(np.float64(t_int), (n_time, n_bl))

    scan_starts = np.array([time[scan_index == s][0] for s in range(scan_index.max() + 1)])
    frequency = float(config["frequency"])
    coords = {
        "time": ("time", time, {**_TIME_ATTRS, "integration_time": _quantity(t_int, "s")}),
        "baseline_id": ("baseline_id", np.arange(n_bl, dtype=np.int64)),
        "baseline_antenna1_name": ("baseline_id", names[baselines[:, 0]]),
        "baseline_antenna2_name": ("baseline_id", names[baselines[:, 1]]),
        "frequency": (
            "frequency",
            np.array([frequency]),
            {
                "type": "spectral_coord",
                "units": "Hz",
                "observer": "TOPO",
                "spectral_window_name": "L1",
                "spectral_window_intents": ["UNSPECIFIED"],
                "reference_frequency": {
                    "attrs": {"type": "spectral_coord", "units": "Hz", "observer": "TOPO"},
                    "data": frequency,
                },
                "channel_width": _quantity(float(config["bandwidth"]), "Hz"),
            },
        ),
        "polarization": ("polarization", np.array(["RR"])),
        "uvw_label": ("uvw_label", np.array(["u", "v", "w"])),
        "field_name": ("time", field_names(time)),
        "scan_name": (
            "time",
            np.array([f"scan_{unix_to_iso(t)}" for t in scan_starts])[scan_index],
            {"scan_intents": ["OBSERVE_TARGET#ON_SOURCE"]},
        ),
    }
    vis_attrs = {
        "type": "quantity",
        "units": "dimensionless",
        "description": "Normalised correlation coefficients, uncalibrated (TART gains are in gain_xds).",
    }
    data_vars = {
        "VISIBILITY": xr.Variable(dims4, vis, vis_attrs),
        "FLAG": (dims4, flag),
        "WEIGHT": (dims4, weight),
        "UVW": (("time", "baseline_id", "uvw_label"), uvw, {"type": "uvw", "units": "m", "frame": "fk5"}),
        "TIME_CENTROID": (("time", "baseline_id"), time_grid, _TIME_ATTRS),
        "EFFECTIVE_INTEGRATION_TIME": (("time", "baseline_id"), t_int_grid, {"type": "quantity", "units": "s"}),
    }
    attrs = {
        "type": "visibility",
        "schema_version": SCHEMA_VERSION,
        "creation_date": date,
        "creator": {"software_name": "tart-msv4", "version": software_version},
        "observation_info": {
            "observer": ["unknown"],
            "project_UID": TELESCOPE_NAME,
            "release_date": date,
            "telescope_name": TELESCOPE_NAME,
            "site_latitude_deg": float(config["lat"]),
            "site_longitude_deg": float(config["lon"]),
            "site_altitude_m": float(config["alt"]),
            "dut1_s": float(dut1),
        },
        "processor_info": {"type": "CORRELATOR", "sub_type": TELESCOPE_NAME},
        "data_groups": {
            "base": {
                "correlated_data": "VISIBILITY",
                "flag": "FLAG",
                "weight": "WEIGHT",
                "uvw": "UVW",
                "field_and_source": f"{partition_name}/field_and_source_base_xds",
                "description": "Data group associated with the VISIBILITY DataArray",
                "date": date,
            }
        },
    }
    return xr.Dataset(data_vars, coords, attrs)


def antenna_xds(itrs: np.ndarray, names: np.ndarray, station: str) -> xr.Dataset:
    """Build ``antenna_xds`` from ``(n_antenna, 3)`` geocentric ITRS positions in metres."""
    n = names.size
    coords = {
        "antenna_name": ("antenna_name", names),
        "mount": ("antenna_name", np.full(n, "X-Y")),
        "telescope_name": ("antenna_name", np.full(n, TELESCOPE_NAME)),
        "station_name": ("antenna_name", np.full(n, station)),
        "cartesian_pos_label": ("cartesian_pos_label", np.array(["x", "y", "z"])),
        "receptor_label": ("receptor_label", np.array(["pol_0"])),
        "polarization_type": (("antenna_name", "receptor_label"), np.full((n, 1), "R")),
    }
    position_attrs = {
        "type": "location",
        "units": "m",
        "frame": "ITRS",
        "coordinate_system": "geocentric",
        "origin_object_name": "earth",
    }
    metres = {"type": "quantity", "units": "m"}
    data_vars = {
        "ANTENNA_POSITION": (("antenna_name", "cartesian_pos_label"), itrs, position_attrs),
        "ANTENNA_DISH_DIAMETER": ("antenna_name", np.full(n, DISH_DIAMETER_M), metres),
        "ANTENNA_EFFECTIVE_DISH_DIAMETER": ("antenna_name", np.full(n, DISH_DIAMETER_M), metres),
        "ANTENNA_RECEPTOR_ANGLE": (
            ("antenna_name", "receptor_label"),
            np.zeros((n, 1)),
            {"type": "quantity", "units": "rad"},
        ),
    }
    attrs = {"type": "antenna", "overall_telescope_name": TELESCOPE_NAME, "relocatable_antennas": False}
    return xr.Dataset(data_vars, coords, attrs)


def field_and_source_xds(names: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> xr.Dataset:
    """Build ``field_and_source_base_xds``: one field per snapshot at that snapshot's zenith (ra, dec)."""
    coords = {
        "field_name": ("field_name", names),
        "source_name": ("field_name", names),
        "sky_dir_label": ("sky_dir_label", np.array(["ra", "dec"])),
    }
    direction = np.stack([ra, dec], axis=-1)
    data_vars = {"FIELD_PHASE_CENTER_DIRECTION": (("field_name", "sky_dir_label"), direction, _DIRECTION_ATTRS)}
    return xr.Dataset(data_vars, coords, {"type": "field_and_source"})


def gain_xds(
    gains: np.ndarray,
    phases: np.ndarray,
    antenna_enu: np.ndarray,
    time: np.ndarray,
    scan_index: np.ndarray,
    names: np.ndarray,
) -> xr.Dataset:
    """Build the ``gain_xds`` TART extension: each file's calibration snapshot (gains and ENU antenna
    positions) on a ``gain_time`` axis."""
    amp, phase = np.asarray(gains, np.float64), np.asarray(phases, np.float64)
    gain_time = np.array([np.median(time[scan_index == s]) for s in range(amp.shape[0])])
    dims = ("gain_time", "antenna_name")
    coords = {
        "gain_time": ("gain_time", gain_time, _TIME_ATTRS),
        "antenna_name": ("antenna_name", names),
        "enu_label": ("enu_label", np.array(["e", "n", "u"])),
    }
    data_vars = {
        "GAIN": (dims, (amp * np.exp(1j * phase)).astype(np.complex64)),
        "GAIN_AMPLITUDE": (dims, amp),
        "GAIN_PHASE": (dims, phase, {"units": "rad"}),
        "ANTENNA_FLAG": (dims, amp == 0),
        "ANTENNA_POSITION_ENU": ((*dims, "enu_label"), antenna_enu, {"units": "m", "frame": "ENU"}),
    }
    attrs = {
        "type": "extension:gain.tart_msv4",
        "tart_msv4_extension": True,
        "description": "TART's per-file calibration snapshot: antenna gains (not applied to VISIBILITY) and "
        "local East/North/Up antenna positions relative to the site. gain_time is the median snapshot time "
        "of each file.",
    }
    return xr.Dataset(data_vars, coords, attrs)


def _quantity(value: float, units: str) -> dict[str, Any]:
    """Encode a scalar quantity attribute the way xarray-ms and xradio expect."""
    return {"attrs": {"type": "quantity", "units": units}, "data": value}
