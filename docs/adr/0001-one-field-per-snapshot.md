# 1. One field per snapshot, one partition per observation

## Status

Accepted

## Context

TART points at the zenith and does not fringe-stop, so each snapshot's visibilities are phased to the zenith
*at that instant*. Because the sky rotates, every snapshot has a different phase centre on the sky. The earlier
kremetart reader labelled all snapshots with a single field `"zenith"` and gave its direction only as el/az.
MSv4 cannot express that, and it hides the fact that the phase centre moves.

## Decision

- Each snapshot is its own field. It is named `zenith_<UTC ISO time with microseconds>`. The name comes from
  the timestamp, so names stay unique when files are concatenated.
- `FIELD_PHASE_CENTER_DIRECTION` holds the (ra, dec) of the zenith at each snapshot.
- All fields go into **one** partition per observation. This is the default in xradio (`partition_scheme=[]`)
  and xarray-ms (whose partition columns do not include FIELD_ID).
- Each file is one scan.

## Consequences

- `field_and_source_base_xds` has `n_time` entries. That is cheap.
- Rephasing every snapshot to one common centre is a processing step. If it is done, its result is a separate
  `field_and_source` data group, not the default.
