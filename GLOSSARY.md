# Glossary

**Snapshot** — The correlations of every baseline at a single timestamp. One snapshot is one MSv4 `time`
sample. (Not "frame": that word is reserved for coordinate frames.)

**Field** — The phase centre of one snapshot: the direction of the site's zenith at that instant. TART does
not track or fringe-stop, so every snapshot is its own field.

**File** — One TART HDF5 file, holding roughly a minute of consecutive snapshots plus one gain snapshot.

**Scan** — The snapshots of one file. Scan and file are one-to-one.

**Observation** — A time-ordered set of files that share one telescope configuration (site, antenna layout,
baselines, frequency). An observation is one MSv4 partition. Small recalibrations of antenna positions between
files do not start a new observation.

**Calibration snapshot** — The per-antenna complex gains and antenna positions recorded in a file by TART's own
calibration. Both may change from file to file within an observation.

**Dead antenna** — An antenna whose gain amplitude in a file is exactly zero. Every baseline it touches is
unusable for that file's snapshots.

**Baseline** — An ordered antenna pair (antenna 1, antenna 2) with antenna 1 < antenna 2. Its UVW is
position(antenna 1) − position(antenna 2).
