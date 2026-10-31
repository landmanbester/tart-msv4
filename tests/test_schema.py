"""MSv4 schema compliance, the field/UVW model and the geometry invariants, all on synthetic data."""

import erfa
import numpy as np
import pytest
import xarray as xr
from conftest import DEAD_ANTENNA, ENU, SITE, write_tart_hdf

from tart_msv4 import geometry
from tart_msv4.backend import parse_timestamps

pytestmark = pytest.mark.unit
ARCSEC = np.pi / (180 * 3600)


@pytest.fixture
def dt(tart_file) -> xr.DataTree:
    return xr.open_datatree(tart_file, engine="tart")


@pytest.fixture
def part(dt) -> xr.DataTree:
    (name,) = dt.children
    return dt[name]


def _node(part: xr.DataTree, name: str) -> xr.Dataset:
    return part[name].to_dataset(inherit=False)


def assert_schema_ok(tree: xr.DataTree) -> None:
    """Assert that xradio's MSv4 checker reports no issues."""
    check = pytest.importorskip("xradio.schema.check")
    import xradio.measurement_set  # noqa: F401  (registers the MSv4 schemas with the checker)

    issues = check.check_datatree(tree)
    assert not issues, str(issues)


def test_check_datatree_passes(dt) -> None:
    assert_schema_ok(dt)


def test_check_datatree_passes_multi_file(tart_files) -> None:
    import tart_msv4

    assert_schema_ok(tart_msv4.open_tart(tart_files))


def test_one_field_per_snapshot(part) -> None:
    ds, fields = part.ds, _node(part, "field_and_source_base_xds")
    names = ds.field_name.values
    assert len(set(names)) == ds.sizes["time"] == fields.sizes["field_name"]
    assert set(names) == set(fields.field_name.values)
    assert names[0] == "zenith_2026-06-09T08:10:42.993515"
    np.testing.assert_array_equal(fields.source_name.values, fields.field_name.values)
    assert ds.attrs["data_groups"]["base"]["field_and_source"] == f"{part.name}/field_and_source_base_xds"
    assert fields.FIELD_PHASE_CENTER_DIRECTION.attrs == {"type": "sky_coord", "units": "rad", "frame": "fk5"}


def test_phase_centre_is_the_zenith(part) -> None:
    ds, fields = part.ds, _node(part, "field_and_source_base_xds")
    ra, dec = fields.FIELD_PHASE_CENTER_DIRECTION.values.T
    days = np.floor(ds.time.values / 86400)
    utc1, utc2 = 2440587.5 + days, ds.time.values / 86400 - days
    lon, lat = np.radians(SITE["lon"]), np.radians(SITE["lat"])
    _, zenith_distance, *_ = erfa.atco13(ra, dec, 0, 0, 0, 0, utc1, utc2, 0, lon, lat, SITE["alt"], 0, 0, 0, 0, 0, 0)
    assert np.all(zenith_distance < 1 * ARCSEC)


def test_matches_astropy(part) -> None:
    """Zenith (ra, dec) and UVW agree with an independent astropy computation using the same UT1.

    astropy applies IERS polar motion (|xp|, |yp| < 0.5 arcsec), which tart-msv4 neglects, so directions agree
    to 0.5 arcsec and UVW to 10 um.
    """
    pytest.importorskip("astropy")
    import astropy.units as u
    from astropy.coordinates import GCRS, ICRS, ITRS, AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    ds = part.ds
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        t = Time(ds.time.values, format="unix", scale="utc")
        t.delta_ut1_utc = 0.0
        loc = EarthLocation(lat=SITE["lat"] * u.deg, lon=SITE["lon"] * u.deg, height=SITE["alt"] * u.m)
        zen = SkyCoord(AltAz(az=0 * u.deg, alt=90 * u.deg, obstime=t, location=loc, pressure=0)).transform_to(ICRS())
        pos = _node(part, "antenna_xds").ANTENNA_POSITION.values
        a1 = [int(n[3:]) for n in ds.baseline_antenna1_name.values]
        a2 = [int(n[3:]) for n in ds.baseline_antenna2_name.values]
        b = pos[a1] - pos[a2]
        uvw = np.empty((t.size, b.shape[0], 3))
        for i in range(t.size):
            itrs = ITRS(
                x=b[:, 0] * u.m, y=b[:, 1] * u.m, z=b[:, 2] * u.m, obstime=t[i], representation_type="cartesian"
            )
            g = itrs.transform_to(GCRS(obstime=t[i])).cartesian.xyz.to_value(u.m).T
            # Independent (u, v, w) basis: w toward the phase centre, u = pole x w (east), v = w x u (north).
            w = zen[i].cartesian.xyz.value
            east = np.cross([0.0, 0.0, 1.0], w)
            east /= np.linalg.norm(east)
            uvw[i] = g @ np.stack([east, np.cross(w, east), w]).T
    direction = _node(part, "field_and_source_base_xds").FIELD_PHASE_CENTER_DIRECTION.values
    dra = (direction[:, 0] - zen.ra.rad + np.pi) % (2 * np.pi) - np.pi
    assert np.all(np.abs(dra * np.cos(zen.dec.rad)) < 0.5 * ARCSEC)
    assert np.all(np.abs(direction[:, 1] - zen.dec.rad) < 0.5 * ARCSEC)
    np.testing.assert_allclose(ds.UVW.values, uvw, atol=1e-5)


def test_uvw_geometry_invariants(part) -> None:
    """w is the ENU 'up' component (0 here) to aberration level; (u, v) is (east, north) rotated about w."""
    ds = part.ds
    a1 = [int(n[3:]) for n in ds.baseline_antenna1_name.values]
    a2 = [int(n[3:]) for n in ds.baseline_antenna2_name.values]
    enu = ENU[a1].astype(float) - ENU[a2]  # pos(ant1) - pos(ant2)
    uvw = ds.UVW.values
    assert np.all(np.abs(uvw[..., 2] - enu[:, 2]) < 1e-3)
    np.testing.assert_allclose(
        np.hypot(uvw[..., 0], uvw[..., 1]), np.broadcast_to(np.hypot(*enu[:, :2].T), uvw.shape[:2]), atol=1e-3
    )
    # The sign convention: u has the sign of (east1 - east2) for an east-west baseline (0, 1).
    assert uvw[0, 0, 0] < -0.99
    assert ds.UVW.attrs == {"type": "uvw", "units": "m", "frame": "fk5"}


def test_dead_antenna_flags_and_weights(part) -> None:
    ds = part.ds
    dead = f"ant{DEAD_ANTENNA:02d}"
    touches = (ds.baseline_antenna1_name == dead) | (ds.baseline_antenna2_name == dead)
    assert bool((ds.FLAG.where(touches, drop=True) == 1).all())
    assert bool((ds.WEIGHT.where(touches, drop=True) == 0).all())
    assert bool((ds.FLAG.where(~touches, drop=True) == 0).all())
    assert bool((ds.WEIGHT.where(~touches, drop=True) == 1).all())


def test_antenna_positions_are_geocentric_itrs(part) -> None:
    ant = _node(part, "antenna_xds")
    pos = ant.ANTENNA_POSITION.values
    # Independent check: the WGS84 geocentric radius at the site latitude, plus height.
    a, b = 6378137.0, 6356752.314245
    phi = np.radians(SITE["lat"])
    radius = np.sqrt(
        ((a**2 * np.cos(phi)) ** 2 + (b**2 * np.sin(phi)) ** 2) / ((a * np.cos(phi)) ** 2 + (b * np.sin(phi)) ** 2)
    )
    assert np.all(np.abs(np.linalg.norm(pos, axis=1) - radius - SITE["alt"]) < 50.0)
    np.testing.assert_allclose(
        geometry.itrs_to_enu(pos, **{k + "_deg" if k != "alt" else "alt_m": v for k, v in SITE.items()}), ENU, atol=1e-6
    )
    assert ant.ANTENNA_POSITION.attrs["frame"] == "ITRS"


def test_timestamps_naive_and_offset_are_utc(tmp_path) -> None:
    aware = parse_timestamps([b"2026-06-09T08:10:42.993515+00:00"])
    assert parse_timestamps(["2026-06-09T08:10:42.993515"]) == aware
    assert parse_timestamps(["2026-06-09T10:10:42.993515+02:00"]) == aware
    naive_file = write_tart_hdf(tmp_path / "naive.hdf", naive=True)
    aware_file = write_tart_hdf(tmp_path / "aware.hdf")
    t_naive = xr.open_dataset(naive_file, engine="tart").time.values
    np.testing.assert_array_equal(t_naive, xr.open_dataset(aware_file, engine="tart").time.values)
    assert t_naive[0] == aware[0]


def test_dtypes_and_attrs(part) -> None:
    ds = part.ds
    assert ds.VISIBILITY.dtype == np.complex64
    assert ds.FLAG.dtype == np.uint8
    assert ds.WEIGHT.dtype == np.float32
    assert ds.UVW.dtype == ds.time.dtype == ds.TIME_CENTROID.dtype == np.float64
    assert ds.VISIBILITY.dims == ("time", "baseline_id", "frequency", "polarization")
    assert ds.VISIBILITY.attrs["units"] == "dimensionless"
    assert ds.time.attrs["scale"] == "utc" and ds.time.attrs["format"] == "unix"
    assert ds.time.attrs["integration_time"]["data"] == pytest.approx(1.025)
    assert list(ds.polarization.values) == ["RR"]
    assert ds.frequency.attrs["channel_width"]["data"] == 2.5e6


def test_gain_extension(part) -> None:
    gains = _node(part, "gain_xds")
    assert gains.attrs["type"] == "extension:gain.tart_msv4"
    assert gains.attrs["tart_msv4_extension"] is True
    assert gains.sizes["gain_time"] == 1
    assert bool(gains.ANTENNA_FLAG.values[0, DEAD_ANTENNA])
    np.testing.assert_allclose(gains.ANTENNA_POSITION_ENU.values[0], ENU)


def test_deterministic_and_creation_date(tart_file) -> None:
    first = xr.open_datatree(tart_file, engine="tart")
    xr.testing.assert_identical(first, xr.open_datatree(tart_file, engine="tart"))
    (name,) = first.children
    assert first[name].ds.attrs["creation_date"] == "2026-06-09T08:10:45.043515"
    other = xr.open_datatree(tart_file, engine="tart", creation_date="2030-01-01T00:00:00")
    attrs = other[name].ds.attrs
    assert attrs["creation_date"] == attrs["data_groups"]["base"]["date"] == "2030-01-01T00:00:00"


def test_leap_second_day_dates() -> None:
    """On a day with a leap second, ERFA's quasi-JD must read back the same clock time (not 1 s later)."""
    from datetime import datetime, timezone

    from tart_msv4.geometry import _utc_two_part

    t = datetime(2016, 12, 31, 23, 59, 59, 500000, tzinfo=timezone.utc).timestamp()
    utc1, utc2 = _utc_two_part(np.array([t]))
    *_, ihmsf = erfa.d2dtf("UTC", 3, utc1, utc2)
    assert tuple(ihmsf[0]) == (23, 59, 59, 500)
