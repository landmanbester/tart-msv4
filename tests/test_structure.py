"""Guardrails for the package's structural rules (see CODING_STANDARDS.md and docs/spec.md)."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SRC = Path(__file__).resolve().parents[1] / "src" / "tart_msv4"
MAX_MODULE_LINES = 300
# xarray imports every registered backend on each engine lookup, so these stay light.
LIGHT_IMPORTS = {"backend.py": {"numpy", "xarray", "h5py", "tart_msv4"}, "lazy.py": {"numpy", "xarray"}}


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_module_size(path: Path) -> None:
    n = len(path.read_text().splitlines())
    assert n <= MAX_MODULE_LINES, f"{path.name} has {n} lines (limit {MAX_MODULE_LINES}): simplify it"


@pytest.mark.parametrize("name", sorted(LIGHT_IMPORTS))
def test_light_module_imports(name: str) -> None:
    """Module-level imports are the stdlib plus an allow-list (deferred imports inside functions are fine)."""
    allowed = LIGHT_IMPORTS[name] | set(sys.stdlib_module_names)
    tree = ast.parse((SRC / name).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert roots <= allowed, f"{name}:{node.lineno} imports {sorted(roots - allowed)} at module level"


def test_lazy_imports_only_light_tart_modules() -> None:
    """backend.py may import tart_msv4.lazy at module level, but not schema/geometry (they pull in pyerfa)."""
    tree = ast.parse((SRC / "backend.py").read_text())
    modules = {n.module for n in tree.body if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("tart")}
    assert modules <= {"tart_msv4.lazy"}


def test_engine_resolution_does_not_import_erfa() -> None:
    code = "import sys, xarray as xr; xr.backends.list_engines(); import tart_msv4; print('erfa' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
