# Coding standards (review)

Read during review. Mechanical rules (formatting, lint, module size, `backend.py` imports, schema compliance)
are enforced by ruff, pre-commit and `tests/test_structure.py`, so they are not repeated here. These are the
judgement calls.

**Spec first.** Check every change against [docs/spec.md](docs/spec.md). A change that contradicts a settled
decision needs a new or updated ADR in `docs/adr/` in the same change. Otherwise, flag it.

**MSv4 is the authority.** Where xradio's MSv4 schema and kremetart disagree, the schema wins. Where the schema
is silent, follow xarray-ms. Every deliberate deviation is listed in the spec and the README. A new one needs a
reason recorded there. [docs/msv4-notes.md](docs/msv4-notes.md) has the schema gotchas.

**Lazy grid.** Anything that scales with `n_time * n_baseline` is a `lazy.GridArray` built from small tables,
including derived arrays. Look for any `np.broadcast_to`, `np.repeat` or full-grid allocation that runs at open
time rather than on read.

**Pure core.** `schema.py` and `geometry.py` are functions of their arguments: no file access, no clock, no
globals that change. Determinism is part of the contract (ADR 3): two opens of the same files are identical.

**Geometry claims are verified, not argued.** A change to phase centres, UVW, time scales or positions comes
with a test against an independent computation (astropy in tests) or the tart2ms reference MS. Tolerances
state their physical cause in the test docstring. A tolerance loosened to make a test pass is a finding.

**Tests prove behaviour.** Each new branch or edge case has a test that would fail without the change. Watch
for tests that reuse the code under test as their oracle.

**Docs track code.** README, MIGRATION.md, the spec and the ADRs are updated in the same change as the
behaviour they describe. Numbers quoted in docs are measured, not estimated.

**Vocabulary.** Use the terms in [GLOSSARY.md](GLOSSARY.md) (snapshot, field, file, scan, observation,
calibration snapshot) in code, docs and messages. Use "snapshot", never "frame", for one timestamp.
