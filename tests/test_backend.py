"""Backend behaviour: engine registration, guessing, laziness, pickling, chunking and multi-file opens."""

import json
import pickle
from datetime import timedelta

import h5py
import numpy as np
import pytest
import xarray as xr
from conftest import CONFIG, ENU, START, synthetic_vis, write_tart_hdf

import tart_msv4
from tart_msv4.backend import TartBackendEntrypoint, TartVisibilityArray

pytestmark = pytest.mark.unit


def _partition(dt: xr.DataTree) -> xr.DataTree:
    (name,) = dt.children
    return dt[name]


def test_engine_is_registered() -> None:
    assert "tart" in xr.backends.list_engines()


def test_open_methods_agree(tart_file) -> None:
    dt = xr.open_datatree(tart_file, engine="tart")
    groups = TartBackendEntrypoint().open_groups_as_dict(tart_file)
    ds = xr.open_dataset(tart_file, engine="tart")
    xr.testing.assert_identical(dt, xr.DataTree.from_dict(groups))
    xr.testing.assert_identical(ds, _partition(dt).to_dataset())
    assert set(groups) == {"/", *(node.path for node in dt.subtree if node.path != "/")}


def test_partition_name(tart_file) -> None:
    assert list(xr.open_datatree(tart_file, engine="tart").children) == [f"{tart_file.stem}_partition_000"]
    assert list(xr.open_datatree(tart_file, engine="tart", partition_name="p").children) == ["p"]


def test_guess_can_open(tart_file, tmp_path) -> None:
    entry = TartBackendEntrypoint()
    assert entry.guess_can_open(tart_file)
    assert entry.guess_can_open(str(tart_file))

    plain = tmp_path / "plain.h5"
    with h5py.File(plain, "w") as f:
        f["vis"] = np.zeros(3)
    no_antennas = tmp_path / "almost.hdf"
    with h5py.File(no_antennas, "w") as f:
        for key in ("vis", "baselines", "timestamp"):
            f[key] = np.zeros(2)
        f.create_dataset("config", data=[json.dumps({"name": "x"})], dtype=h5py.string_dtype())
    text = tmp_path / "notes.hdf"
    text.write_text("not hdf5")
    for candidate in (plain, no_antennas, text, tmp_path / "missing.hdf", tmp_path, 42):
        assert not entry.guess_can_open(candidate)


def test_engine_guessing_picks_tart(tart_file) -> None:
    assert "tart" in xr.backends.plugins.guess_engine(str(tart_file))


def test_rejects_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="open_tart"):
        xr.open_datatree(tmp_path, engine="tart")


def test_open_does_not_read_vis(tart_file, monkeypatch) -> None:
    calls = []
    original = TartVisibilityArray._raw_getitem
    monkeypatch.setattr(TartVisibilityArray, "_raw_getitem", lambda self, key: calls.append(key) or original(self, key))
    vis = _partition(xr.open_datatree(tart_file, engine="tart")).ds.VISIBILITY
    assert calls == []
    assert vis.variable._in_memory is False
    vis.values  # noqa: B018
    assert calls


@pytest.mark.parametrize(
    "indexers",
    [
        {"time": [0, 2], "baseline_id": slice(1, 4)},
        {"time": [2, 0], "baseline_id": [5, 0, 3]},
        {"time": 1, "baseline_id": 4},
        {"time": slice(None, None, 2)},
    ],
)
def test_outer_indexing(tart_file, indexers) -> None:
    vis = _partition(xr.open_datatree(tart_file, engine="tart")).ds.VISIBILITY
    expected = xr.DataArray(synthetic_vis(3, 6)[:, :, None, None], dims=vis.dims).isel(indexers)
    np.testing.assert_array_equal(vis.isel(indexers).values, expected.values)
    assert vis.isel(indexers).dtype == np.complex64


def test_pickle_roundtrip(tart_file) -> None:
    dt = pickle.loads(pickle.dumps(xr.open_datatree(tart_file, engine="tart")))
    np.testing.assert_array_equal(_partition(dt).ds.VISIBILITY.values[..., 0, 0], synthetic_vis(3, 6))


def test_drop_variables(tart_file) -> None:
    ds = xr.open_dataset(tart_file, engine="tart", drop_variables=["TIME_CENTROID"])
    assert "TIME_CENTROID" not in ds and "VISIBILITY" in ds


@pytest.mark.filterwarnings("ignore:The specified chunks separate")
def test_chunks_give_dask(tart_file) -> None:
    da = pytest.importorskip("dask.array")
    dt = xr.open_datatree(tart_file, engine="tart", chunks={"time": 1})
    vis = _partition(dt).ds.VISIBILITY
    assert isinstance(vis.data, da.Array)
    assert vis.chunks[0] == (1, 1, 1)
    np.testing.assert_array_equal(vis.values[..., 0, 0], synthetic_vis(3, 6))
    pickle.loads(pickle.dumps(vis.data)).compute()


def test_open_tart_concatenates_in_time_order(tart_files) -> None:
    dt = tart_msv4.open_tart(tart_files[::-1])
    ds = _partition(dt).ds
    assert ds.sizes["time"] == 9
    assert np.all(np.diff(ds.time.values) > 0)
    assert ds.VISIBILITY.variable._in_memory is False
    np.testing.assert_array_equal(ds.VISIBILITY.values[..., 0, 0], synthetic_vis(9, 6))
    assert list(dt.children) == ["file0_partition_000"]
    assert len(set(ds.scan_name.values)) == 3


def test_open_tart_keeps_one_chunk_per_file(tart_files) -> None:
    da = pytest.importorskip("dask.array")
    ds = _partition(tart_msv4.open_tart(tart_files, chunks={})).ds
    assert isinstance(ds.VISIBILITY.data, da.Array)
    assert ds.VISIBILITY.chunks[0] == (3, 3, 3)
    np.testing.assert_array_equal(ds.VISIBILITY.values[..., 0, 0], synthetic_vis(9, 6))


def test_open_tart_gains_and_flags_per_file(tart_files) -> None:
    dt = tart_msv4.open_tart(tart_files)
    ds = _partition(dt)
    gains = ds["gain_xds"].to_dataset(inherit=False)
    assert gains.sizes["gain_time"] == 3
    np.testing.assert_array_equal(gains.ANTENNA_FLAG.values, np.eye(4, dtype=bool)[:3])
    # File i has antenna i dead: its snapshots flag exactly the baselines touching antenna i.
    a1, a2 = ds.ds.baseline_antenna1_name.values, ds.ds.baseline_antenna2_name.values
    for i in range(3):
        touches = (a1 == f"ant{i:02d}") | (a2 == f"ant{i:02d}")
        flags = ds.ds.FLAG.values[3 * i : 3 * i + 3, :, 0, 0]
        assert np.array_equal(flags, np.broadcast_to(touches, flags.shape).astype(np.uint8))


def test_open_tart_rejects_mismatched_config(tart_files, tmp_path) -> None:
    other = write_tart_hdf(tmp_path / "other.hdf", start=START + timedelta(seconds=20), config={**CONFIG, "lat": 0.0})
    with pytest.raises(ValueError, match="'config' of .*other.hdf"):
        tart_msv4.open_tart([*tart_files, other])


def test_open_tart_position_tolerance(tart_files, tmp_path) -> None:
    moved = ENU.copy()
    moved[1, 0] += 0.003
    shifted = write_tart_hdf(tmp_path / "shifted.hdf", start=START + timedelta(seconds=20), t0=9, enu=moved)
    ds = _partition(tart_msv4.open_tart([*tart_files, shifted]))
    enu = ds["gain_xds"].to_dataset(inherit=False).ANTENNA_POSITION_ENU.values
    np.testing.assert_allclose(enu[-1], moved)
    # The last file's UVW uses its own positions: baseline (0, 1) is 3 mm longer.
    uvw = ds.ds.UVW.values
    np.testing.assert_allclose(np.hypot(*uvw[-1, 0, :2].T), 1.003, atol=1e-6)
    np.testing.assert_allclose(np.hypot(*uvw[0, 0, :2].T), 1.0, atol=1e-6)

    far = ENU.copy()
    far[1, 0] += 0.05
    bad = write_tart_hdf(tmp_path / "far.hdf", start=START + timedelta(seconds=30), enu=far)
    with pytest.raises(ValueError, match="antenna_positions"):
        tart_msv4.open_tart([*tart_files, bad])


def test_open_tart_needs_paths() -> None:
    with pytest.raises(ValueError, match="no TART files"):
        tart_msv4.open_tart([])


def test_engine_resolution_does_not_import_erfa() -> None:
    import subprocess
    import sys

    code = "import sys, xarray as xr; xr.backends.list_engines(); import tart_msv4; print('erfa' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
