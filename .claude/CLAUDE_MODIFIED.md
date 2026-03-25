# CLAUDE_MODIFIED.md — Project State

This file tracks the evolving state of the project. It supplements CLAUDE_INITIAL.md (immutable).

## Session: 2026-03-25 — Initial Scaffold

### What was done
1. Updated `.claude/settings.local.json` with permissions for pytest, sphinx-build, python, make, Write, Edit
2. Created `.claude/DESIGN.md` with architecture decisions and conventions
3. Scaffold code created:
   - `src/ddp_binarize/binarize.py` — `OtsuBinarizer` functor + `main_binarize_offline`
   - `src/ddp_resolution/resolution.py` — `HeuristicResolutionEstimator` + `main_resolution_offline`
   - `src/ddp_recto/recto_verso.py` — updated: fixed import bug, added `RectoSelector`/`HeuristicRectoSelector`
   - `src/ddp_cv_preprocess/fsdb.py` — `FSDBCharter`, `iter_fsdb`
   - `src/ddp_cv_preprocess/offline.py` — `main_cv_preprocess_offline`
4. Project files: `setup.py`, `pyproject.toml`, `requirements.txt`, `README.md`, `Makefile`, `.coveragerc`
5. Tests: `test/conftest.py` (session fixture creating fake FSDB), `test/unittest/test_*.py`
6. Docs: `docs/conf.py`, `docs/index.md`, `docs/quickstart.md`, `docs/cli.md`, `docs/api.md`
7. Static `test/fake_fsdb_root/` created with a single charter (PIL-generated image)

### Bug fixes applied
- `src/ddp_recto/recto_verso.py`: fixed `from .util import` → `from ddp_cv_preprocess.util import`
- `src/ddp_binarize/binarize.py`: fixed Otsu threshold `gray < threshold` → `gray <= threshold` (dark pixels at exactly the threshold were wrongly classified as background)
- `pyproject.toml`: fixed build backend `setuptools.backends.legacy:build` → `setuptools.build_meta` (pip install -e . was failing)
- `pyproject.toml`: added `pythonpath = ["src"]` to pytest config so modules are importable without install

### Design decisions made
- Functor interface: ABC with `__call__` for all three tasks
- Fake FSDB for tests created by `test/conftest.py` session fixture AND static `test/fake_fsdb_root/` for reference
- Package layout: `src/` layout with `package_dir={'': 'src'}`
- License: AGPL-3.0
- Linting: `ruff format` only (whitespace-safe); `ruff check` issues in existing files (dibco.py, resresnet.py) are not enforced by `make testlint`

### Makefile targets
| Target | Command | Description |
|---|---|---|
| `clean` | find + rm | Remove build artefacts, .pyc, egg-info, docs build |
| `build` | setup.py sdist bdist_wheel | Build for PyPI |
| `htmldoc` / `doc` | sphinx-build -b html | Build HTML docs to docs/_build/html |
| `pdfdoc` | sphinx-build -b latex | Build PDF docs |
| `test` | pytest test -x | Run all tests, stop on first failure |
| `testfull` | pytest test | Run all tests, never stop |
| `unitest` | pytest test/unittest --cov | Unittests with coverage report |
| `testlint` | ruff format --check src | Passively check formatting would be a no-op |
| `autolint_spaces` | ruff format src | Auto-fix whitespace/formatting only |

### Known TODOs / Next Steps
- Implement ML-based functors in `binet.py` (binarize) and `resresnet.py` (resolution)
- Implement layout-based heuristic in `HeuristicRectoSelector` using `.layout.pred.json` files
- Add docstrings when explicitly requested
- Add `dibco.py` gdown fallback for broken URLs
- Consider adding `ddp_binarize/train.py` for training entry point
- Fix remaining `ruff check` issues (unused imports, F401 re-exports in `__init__.py` files) when code matures beyond scaffold
