"""Site geometry, instantaneous-zenith phase centres and UVW, computed with pyerfa.

All functions are pure and vectorised over time. Times are float64 unix seconds (UTC). Without IERS data,
UT1 - UTC is taken from the ``dut1`` argument (default 0 s). That approximation shifts right ascensions by
about 15 arcsec per second of |UT1 - UTC| (|UT1 - UTC| < 0.9 s by definition). It is small compared with
TART's resolution of about 7 degrees.
"""

import erfa
import numpy as np

_SECONDS_PER_DAY = 86400.0


def geodetic_to_itrs(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert WGS84 geodetic coordinates to a geocentric ITRS position.

    Args:
        lat_deg: Geodetic latitude in degrees.
        lon_deg: Longitude in degrees, east positive.
        alt_m: Height above the WGS84 ellipsoid in metres.

    Returns:
        The ``(3,)`` ITRS position in metres.
    """
    return erfa.gd2gc(erfa.WGS84, np.radians(lon_deg), np.radians(lat_deg), alt_m)


def enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the ``(3, 3)`` matrix whose columns are the site's East, North and Up unit vectors in ITRS."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0.0, np.cos(lat), np.sin(lat)],
        ]
    )


def enu_to_itrs(enu: np.ndarray, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Map ``(n, 3)`` local East/North/Up offsets (m) at a site to geocentric ITRS positions (m).

    Uses the exact tangent plane at the WGS84 site position.
    """
    site = geodetic_to_itrs(lat_deg, lon_deg, alt_m)
    return site + np.asarray(enu, dtype=np.float64) @ enu_rotation(lat_deg, lon_deg).T


def itrs_to_enu(itrs: np.ndarray, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Invert :func:`enu_to_itrs`: ``(n, 3)`` ITRS positions (m) to local East/North/Up offsets (m)."""
    site = geodetic_to_itrs(lat_deg, lon_deg, alt_m)
    return (np.asarray(itrs, dtype=np.float64) - site) @ enu_rotation(lat_deg, lon_deg)


def _utc_two_part(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert unix seconds to an ERFA two-part UTC quasi Julian date.

    Built from calendar fields with ``erfa.dtf2d``, so that a day containing a leap second correctly has
    86401 s in ERFA's quasi-JD convention (dividing seconds-of-day by 86400 would be off by up to 1 s).
    """
    time = np.asarray(time, dtype=np.float64)
    days = np.floor(time / _SECONDS_PER_DAY)
    seconds = time - days * _SECONDS_PER_DAY
    date = days.astype(np.int64).astype("datetime64[D]")
    month = date.astype("datetime64[M]")
    year = month.astype("datetime64[Y]").astype(np.int32) + 1970
    return erfa.dtf2d(
        "UTC",
        year,
        month.astype(np.int32) % 12 + 1,
        (date - month).astype(np.int32) + 1,
        (seconds // 3600).astype(np.int32),
        (seconds % 3600 // 60).astype(np.int32),
        seconds % 60,
    )


def zenith_radec(
    time: np.ndarray, lat_deg: float, lon_deg: float, alt_m: float, dut1: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the astrometric ICRS (ra, dec) of the site zenith at each time.

    Inverts the full observed-to-ICRS chain (``erfa.atoc13``), which includes aberration and light
    deflection, with refraction disabled. This matches astropy's ``AltAz -> ICRS`` and casacore's
    ``AZEL -> J2000`` conversions.

    Args:
        time: Unix seconds (UTC), shape ``(n,)``.
        lat_deg: Geodetic latitude in degrees.
        lon_deg: Longitude in degrees, east positive.
        alt_m: Height above the WGS84 ellipsoid in metres.
        dut1: UT1 - UTC in seconds.

    Returns:
        ``(ra, dec)`` in radians, each of shape ``(n,)``, with ra in ``[0, 2pi)``.
    """
    utc1, utc2 = _utc_two_part(time)
    # Azimuth 0, zenith distance 0; pressure 0 disables refraction. Polar motion is neglected.
    ra, dec = erfa.atoc13(
        "A", 0.0, 0.0, utc1, utc2, dut1, np.radians(lon_deg), np.radians(lat_deg), alt_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    return np.mod(ra, 2.0 * np.pi), dec


def celestial_to_terrestrial(time: np.ndarray, dut1: float = 0.0) -> np.ndarray:
    """Return the ``(n, 3, 3)`` IAU 2006/2000A GCRS-to-ITRS rotation at each time (polar motion neglected)."""
    utc1, utc2 = _utc_two_part(time)
    tai1, tai2 = erfa.utctai(utc1, utc2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    ut11, ut12 = erfa.utcut1(utc1, utc2, dut1)
    return erfa.c2t06a(tt1, tt2, ut11, ut12, 0.0, 0.0)


def uvw_basis(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Return ``(n, 3, 3)`` matrices whose rows are the celestial (u, v, w) unit vectors toward (ra, dec)."""
    sra, cra, sdec, cdec = np.sin(ra), np.cos(ra), np.sin(dec), np.cos(dec)
    zero = np.zeros_like(ra)
    return np.stack(
        [
            np.stack([-sra, cra, zero], axis=-1),
            np.stack([-sdec * cra, -sdec * sra, cdec], axis=-1),
            np.stack([cdec * cra, cdec * sra, sdec], axis=-1),
        ],
        axis=-2,
    )


def uvw_rotation(time: np.ndarray, ra: np.ndarray, dec: np.ndarray, dut1: float = 0.0) -> np.ndarray:
    """Return the ``(n_time, 3, 3)`` matrices mapping an ITRS vector to (u, v, w) toward each (ra, dec).

    The vector is rotated from ITRS into the celestial frame at its time (Earth rotation plus precession and
    nutation), then projected onto the (u, v, w) basis of that time's phase centre.
    """
    # v_itrs = R @ v_gcrs, so the (u, v, w) components of an ITRS vector are basis @ R.T @ v_itrs.
    return uvw_basis(ra, dec) @ np.swapaxes(celestial_to_terrestrial(time, dut1), -1, -2)


def baseline_uvw(
    baselines_itrs: np.ndarray, time: np.ndarray, ra: np.ndarray, dec: np.ndarray, dut1: float = 0.0
) -> np.ndarray:
    """Project ITRS baseline vectors onto each time's (u, v, w) basis (see :func:`uvw_rotation`).

    Args:
        baselines_itrs: ``(n_baseline, 3)`` ITRS baseline vectors in metres.
        time: ``(n_time,)`` unix seconds (UTC).
        ra: ``(n_time,)`` phase-centre right ascension in radians.
        dec: ``(n_time,)`` phase-centre declination in radians.
        dut1: UT1 - UTC in seconds.

    Returns:
        ``(n_time, n_baseline, 3)`` UVW in metres.
    """
    rot = uvw_rotation(time, ra, dec, dut1)
    return np.einsum("tij,bj->tbi", rot, np.asarray(baselines_itrs, dtype=np.float64))


def uvw_rows(
    rot: np.ndarray, baselines_itrs: np.ndarray, scan_index: np.ndarray, t: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Return UVW for snapshot indices ``t`` and baseline indices ``b``.

    Args:
        rot: ``(n_time, 3, 3)`` matrices from :func:`uvw_rotation`.
        baselines_itrs: ``(n_scan, n_baseline, 3)`` ITRS baselines of each file, in metres.
        scan_index: ``(n_time,)`` file index of each snapshot.
        t: Snapshot indices.
        b: Baseline indices.

    Returns:
        ``(t.size, b.size, 3)`` UVW in metres.
    """
    return np.einsum("tij,tbj->tbi", rot[t], baselines_itrs[scan_index[t]][:, b])
