"""Comparison with the tart2ms reference Measurement Set (real data; skipped if the bundle is absent).

The reference ``vis_2026-06-09_08_11_43.476804_nocal.ms`` was written with::

    tart2ms --hdf <stem>.hdf --ms <stem>_nocal.ms --rephase obs-midpoint --single-field \\
        --write-model-catalog --uncalibrated

and is opened here with ``xr.open_datatree(ms, engine="xarray-ms:msv2")``. It holds a single field: the zenith
at the middle snapshot ``time[n // 2]``. That snapshot therefore has the same phase centre in both datasets,
which allows a direct value comparison with no rephasing:

- **Conjugation:** VISIBILITY is bit-identical to the MS DATA, **not conjugated**.
- **UVW sign:** ``pos(ant1) - pos(ant2)``, as in the MS. ``w`` agrees to < 1 mm. ``u`` and ``v`` agree only
  to ~2 cm, because tart2ms places antennas with a spherical-Earth ``offset_by`` (baseline lengths change by
  ~0.3%) while tart-msv4 uses the exact WGS84 tangent plane. ``test_schema.test_matches_astropy`` checks
  tart-msv4's UVW to 10 um against an independent computation.
- **Phase centre:** < 1 arcsec, with UT1 = UTC.
- **FLAG:** tart2ms flags nothing. tart-msv4 flags the baselines of the dead antenna (gain 0).
"""

import numpy as np
import pytest
import xarray as xr

import tart_msv4
from tart_msv4 import geometry

pytestmark = pytest.mark.integration
ARCSEC = np.pi / (180 * 3600)


def _partition(dt: xr.DataTree) -> xr.DataTree:
    (name,) = dt.children
    return dt[name]


@pytest.fixture(scope="module")
def reference(ref_ms) -> xr.DataTree:
    pytest.importorskip("xarray_ms")
    with pytest.warns():  # xarray-ms warns about imputed STATE/PROCESSOR rows and the ITRF UVW frame
        return _partition(xr.open_datatree(str(ref_ms), engine="xarray-ms:msv2"))


@pytest.fixture(scope="module")
def generated(ref_hdf) -> xr.DataTree:
    return _partition(xr.open_datatree(ref_hdf, engine="tart"))


@pytest.fixture(scope="module")
def mid(generated) -> int:
    return generated.ds.sizes["time"] // 2


def test_check_datatree(generated) -> None:
    check = pytest.importorskip("xradio.schema.check")
    assert not check.check_datatree(generated.root)


def test_check_datatree_all_bundled_files(bundle_hdfs) -> None:
    check = pytest.importorskip("xradio.schema.check")
    dt = tart_msv4.open_tart(bundle_hdfs)
    assert not check.check_datatree(dt)
    ds = _partition(dt).ds
    assert ds.sizes["time"] == 60 * len(bundle_hdfs)
    assert len(set(ds.field_name.values)) == ds.sizes["time"]


# --- structure (ported from kremetart's test_read_tart_hdf_msv4.py) ---------------------------------------


def test_subnodes(reference, generated) -> None:
    assert {"antenna_xds", "field_and_source_base_xds"} <= set(reference.children) <= set(generated.children)


def test_main_dimensions_match(reference, generated) -> None:
    assert dict(generated.ds.sizes) == dict(reference.ds.sizes)


def test_main_coords_and_vars_are_supersets(reference, generated) -> None:
    gen, ref = generated.to_dataset(inherit=False), reference.to_dataset(inherit=False)
    assert set(ref.coords) <= set(gen.coords)
    assert set(ref.data_vars) - {"SIGMA_SPECTRUM"} <= set(gen.data_vars)


def test_core_dtypes_dims_and_attrs(reference, generated) -> None:
    g, r = generated.ds, reference.ds
    for name in ("VISIBILITY", "FLAG", "WEIGHT", "UVW", "TIME_CENTROID", "EFFECTIVE_INTEGRATION_TIME"):
        assert g[name].dtype == r[name].dtype, name
        assert g[name].dims == r[name].dims, name
    assert g.UVW.attrs == r.UVW.attrs
    assert g.TIME_CENTROID.attrs == r.TIME_CENTROID.attrs
    for key in ("type", "units", "format", "scale"):
        assert g.time.attrs[key] == r.time.attrs[key]
    assert g.attrs["schema_version"] == r.attrs["schema_version"]
    assert set(g.attrs["data_groups"]["base"]) == set(r.attrs["data_groups"]["base"])


def test_antenna_xds(reference, generated) -> None:
    r = reference["antenna_xds"].to_dataset(inherit=False)
    g = generated["antenna_xds"].to_dataset(inherit=False)
    assert dict(g.sizes) == dict(r.sizes)
    assert set(r.coords) <= set(g.coords) and set(r.data_vars) <= set(g.data_vars)
    assert g.ANTENNA_POSITION.attrs == r.ANTENNA_POSITION.attrs
    np.testing.assert_allclose(g.ANTENNA_DISH_DIAMETER.values, r.ANTENNA_DISH_DIAMETER.values)


def test_field_and_source_attrs(reference, generated) -> None:
    r = reference["field_and_source_base_xds"].to_dataset(inherit=False)
    g = generated["field_and_source_base_xds"].to_dataset(inherit=False)
    assert g.FIELD_PHASE_CENTER_DIRECTION.attrs == r.FIELD_PHASE_CENTER_DIRECTION.attrs
    assert g.sizes["field_name"] == generated.ds.sizes["time"]  # one field per snapshot
    assert r.sizes["field_name"] == 1  # the reference was rephased to a single field


# --- values -----------------------------------------------------------------------------------------------


def test_time(reference, generated) -> None:
    np.testing.assert_allclose(generated.ds.time.values, reference.ds.time.values, rtol=0, atol=2e-6)


def test_phase_centre_at_mid_snapshot(reference, generated, mid) -> None:
    ref = reference["field_and_source_base_xds"].ds.FIELD_PHASE_CENTER_DIRECTION.values[0]
    ours = generated["field_and_source_base_xds"].ds.FIELD_PHASE_CENTER_DIRECTION.values[mid]
    assert abs(ours[0] - ref[0]) * np.cos(ref[1]) < 1 * ARCSEC
    assert abs(ours[1] - ref[1]) < 1 * ARCSEC


def test_uvw_at_mid_snapshot(reference, generated, mid) -> None:
    ours, ref = generated.ds.UVW.values[mid], reference.ds.UVW.values[mid]
    assert np.abs(ours[:, 2] - ref[:, 2]).max() < 1e-3
    assert np.abs(ours[:, :2] - ref[:, :2]).max() < 3e-2  # tart2ms spherical antenna placement, see docstring
    # The sign is unambiguous: flipping it would give errors of order the baseline length (metres).
    assert np.abs(-ours[:, :2] - ref[:, :2]).max() > 0.5


def test_visibility_is_not_conjugated(reference, generated, mid) -> None:
    ours, ref = generated.ds.VISIBILITY.values, reference.ds.VISIBILITY.values
    np.testing.assert_array_equal(ours[mid], ref[mid])
    np.testing.assert_allclose(np.abs(ours), np.abs(ref), rtol=1e-6)  # rephasing changes only the phase


def test_uvw_matches_reference_geometry_with_its_positions(reference, generated, mid, ref_hdf) -> None:
    """Given tart2ms's own (ITRS) antenna positions, our projection reproduces its UVW to < 1 mm."""
    pos = reference["antenna_xds"].ds.ANTENNA_POSITION.values
    a1 = [int(n[3:]) for n in generated.ds.baseline_antenna1_name.values]
    a2 = [int(n[3:]) for n in generated.ds.baseline_antenna2_name.values]
    ra, dec = reference["field_and_source_base_xds"].ds.FIELD_PHASE_CENTER_DIRECTION.values[0]
    t = generated.ds.time.values[mid : mid + 1]
    uvw = geometry.baseline_uvw(pos[a1] - pos[a2], t, np.array([ra]), np.array([dec]))[0]
    assert np.abs(uvw - reference.ds.UVW.values[mid]).max() < 1e-3


def test_flags(reference, generated) -> None:
    assert not reference.ds.FLAG.values.any()
    gains = generated["gain_xds"].ds
    dead = set(gains.antenna_name.values[gains.ANTENNA_FLAG.values[0]])
    touches = generated.ds.baseline_antenna1_name.isin(list(dead)) | generated.ds.baseline_antenna2_name.isin(
        list(dead)
    )
    assert dead == {"ant15"}
    np.testing.assert_array_equal(generated.ds.FLAG.values[..., 0, 0].any(axis=0), touches.values)
