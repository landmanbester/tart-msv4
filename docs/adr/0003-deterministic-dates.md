# 3. Deterministic creation dates

## Status

Accepted

## Context

MSv4 requires `creation_date` on the visibility dataset and a `date` on each data group. xarray-ms fills both
with `datetime.now()`. With that choice, two opens of the same file are not equal, and the schema builders
stop being pure functions.

## Decision

Both dates default to the UTC time of the observation's last snapshot. That is when the last HDF file was
written, and TART names files by it. A `creation_date` keyword overrides the default.

## Consequences

Repeated opens produce identical trees, which helps tests and caching. The date describes the data, not the
moment the file was opened.
