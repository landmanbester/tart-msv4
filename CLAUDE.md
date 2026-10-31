# CLAUDE.md

- Requirements and settled decisions: [docs/spec.md](docs/spec.md). Rationale: `docs/adr/`. Terms: [GLOSSARY.md](GLOSSARY.md).
- Review rules: [CODING_STANDARDS.md](CODING_STANDARDS.md).
- MSv4 schema and pyerfa gotchas: [docs/msv4-notes.md](docs/msv4-notes.md). Read it before changing `schema.py` or `geometry.py`.
- The real TART HDF5 layout is documented in `tests/conftest.py`. The real-data bundle is in `tests/data/`; point `TART_MSV4_DATA=<dir>` at an existing copy.
- Run commands from this directory with `env -u VIRTUAL_ENV uv run ...`, so that `uv` uses this project's venv rather than an outer one.
