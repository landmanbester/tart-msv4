# 2. UVW = position(antenna 1) − position(antenna 2)

## Status

Accepted

## Context

There are two candidate baseline sign conventions, `pos(ant2) − pos(ant1)` and `pos(ant1) − pos(ant2)`, and
the sign must agree with the phase convention of the visibilities. The reference Measurement Set that tart2ms
wrote from the same HDF file (opened through xarray-ms) uses UVW ≈ `pos(ant1) − pos(ant2)`. For baseline
(0, 1) it gives (−0.2245, −0.5415, 0) m, against an ENU difference pos0 − pos1 = (−0.226, −0.544). TART
visibilities with that UVW sign have been imaged successfully.

## Decision

UVW is `pos(ant1) − pos(ant2)`, projected onto the (u, v, w) basis of each snapshot's phase centre.
Visibilities are passed through unconjugated. `tests/test_reference_ms.py` checks both the sign and the
conjugation against the reference MS, at the snapshot whose phase centre equals the MS phase centre.

## Consequences

Consumers must not flip the sign. The convention is stated in the README and the glossary.
