"""Test fixtures: tiny synthetic TART files (always available) and the optional real-data bundle.

The synthetic writer copies the real TART HDF5 layout recorded from
``vis_2026-06-09_08_11_43.476804.hdf``. The root has no groups and no attributes, and every dataset is
contiguous with no attributes:

    vis                (n_time, n_baseline) complex64, stored as an HDF5 compound {r, i}
    baselines          (n_baseline, 2) int32, i < j in row-major order
    config             (1,) variable-length ASCII string holding JSON
    timestamp          (n_time,) variable-length UTF-8 ISO 8601 strings with a "+00:00" offset
    antenna_positions  (n_antenna, 3) float32, local East/North/Up in metres
    gains              (n_antenna,) float32 amplitudes (0 = dead antenna)
    phases             (n_antenna,) float32 radians
    phase_elaz         (2,) float64 [90, 0]

The real bundle (nine TART files plus tart2ms Measurement Sets) is downloaded once into ``tests/data``.
``TART_MSV4_OFFLINE=1`` skips the download, ``TART_MSV4_DATA=<dir>`` points at an existing copy, and
``TART_MSV4_REQUIRE_DATA=1`` (set in CI) makes the real-data tests fail instead of skip when it is missing.
"""

import json
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

DATA_DIR = Path(os.environ.get("TART_MSV4_DATA", Path(__file__).resolve().parent / "data"))
BUNDLE_ID = "1vENbb8wQatHCNwO6LODXgPDwchcHzo7m"  # https://drive.google.com/file/d/<id>/view
REF_STEM = "vis_2026-06-09_08_11_43.476804"
_BUNDLE_MARKER = f"{REF_STEM}_nocal.ms"

SITE = {"lat": -20.2587508, "lon": 57.7591989, "alt": 20.0}
CONFIG = {
    "name": "Synthetic - Test",
    "frequency": 1575420000.0,
    "L0_frequency": 1571328000.0,
    "baseband_frequency": 4092000.0,
    "sampling_frequency": 16368000.0,
    "bandwidth": 2500000.0,
    **SITE,
    "num_antenna": 4,
    "orientation": 0.0,
    "axes": ["East", "North", "Up"],
}
ENU = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.5, -0.5, 0.0]], dtype=np.float32)
DEAD_ANTENNA = 2
START = datetime(2026, 6, 9, 8, 10, 42, 993515, tzinfo=timezone.utc)


def synthetic_vis(n_time: int, n_baseline: int, t0: int = 0) -> np.ndarray:
    """The known visibility pattern: real part = global time index + 1, imaginary part = baseline + 1."""
    t = np.arange(t0, t0 + n_time)[:, None] + 1.0
    b = np.arange(n_baseline)[None, :] + 1.0
    return (t + 1j * b).astype(np.complex64)


def write_tart_hdf(
    path: Path,
    *,
    start: datetime = START,
    n_time: int = 3,
    t0: int = 0,
    config: dict | None = None,
    enu: np.ndarray = ENU,
    gains: np.ndarray | None = None,
    naive: bool = False,
) -> Path:
    """Write a synthetic TART HDF5 file using the real layout (see the module docstring)."""
    config = config or CONFIG
    n_ant = int(config["num_antenna"])
    baselines = np.array([(i, j) for i in range(n_ant) for j in range(i + 1, n_ant)], dtype=np.int32)
    if gains is None:
        gains = np.ones(n_ant, dtype=np.float32)
        gains[DEAD_ANTENNA] = 0.0
    stamps = [start + timedelta(seconds=1.025 * i) for i in range(n_time)]
    text = [(s.replace(tzinfo=None) if naive else s).isoformat() for s in stamps]
    with h5py.File(path, "w") as f:
        f["vis"] = synthetic_vis(n_time, baselines.shape[0], t0)
        f["baselines"] = baselines
        f.create_dataset("config", data=[json.dumps(config)], dtype=h5py.string_dtype("ascii"))
        f.create_dataset("timestamp", data=text, dtype=h5py.string_dtype("utf-8"))
        f["antenna_positions"] = np.asarray(enu, dtype=np.float32)
        f["gains"] = np.asarray(gains, dtype=np.float32)
        f["phases"] = np.linspace(0.0, 1.0, n_ant, dtype=np.float32)
        f["phase_elaz"] = np.array([90.0, 0.0])
    return path


@pytest.fixture
def tart_file(tmp_path: Path) -> Path:
    """A synthetic TART file: 4 antennas (antenna 2 dead), 6 baselines, 3 snapshots."""
    return write_tart_hdf(tmp_path / "vis_2026-06-09_08_10_45.043515.hdf")


@pytest.fixture
def tart_files(tmp_path: Path) -> list[Path]:
    """Three consecutive synthetic files of one observation (3 snapshots each), with differing gains."""
    paths = []
    for i in range(3):
        gains = np.ones(4, dtype=np.float32)
        gains[i] = 0.0  # a different dead antenna per file
        start = START + timedelta(seconds=4.0 * i)
        paths.append(write_tart_hdf(tmp_path / f"file{i}.hdf", start=start, t0=3 * i, gains=gains))
    return paths


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fetch the real-data bundle once, best effort. Tests that need it skip when it is absent."""
    if os.environ.get("TART_MSV4_OFFLINE") == "1" or (DATA_DIR / _BUNDLE_MARKER).exists():
        return
    try:
        import gdown

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATA_DIR / "_tart_test_data.zip"
        gdown.download(id=BUNDLE_ID, output=str(zip_path), quiet=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink(missing_ok=True)
    except Exception as exc:  # the dependent tests skip
        print(f"[tart-msv4 tests] test-data download failed ({exc}); real-data tests will skip")


def _missing(reason: str) -> None:
    """Skip a real-data test, or fail it when ``TART_MSV4_REQUIRE_DATA=1``."""
    if os.environ.get("TART_MSV4_REQUIRE_DATA") == "1":
        pytest.fail(f"{reason} (TART_MSV4_REQUIRE_DATA=1)")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def ref_hdf() -> Path:
    """The bundled reference TART file (skips if absent)."""
    path = DATA_DIR / f"{REF_STEM}.hdf"
    if not path.exists():
        _missing("real-data bundle not available")
    return path


@pytest.fixture(scope="session")
def ref_ms() -> Path:
    """The tart2ms reference MS (uncalibrated, rephased to the zenith at the middle snapshot)."""
    path = DATA_DIR / _BUNDLE_MARKER
    if not path.exists():
        _missing("reference Measurement Set not available")
    return path


@pytest.fixture(scope="session")
def bundle_hdfs(ref_hdf: Path) -> list[Path]:
    """All bundled TART files."""
    return sorted(ref_hdf.parent.glob("*.hdf"))
